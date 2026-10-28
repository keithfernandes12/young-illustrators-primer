"""Illustrated picture books from African Storybook and StoryWeaver (Pratham Books).

Both come from the Global Storybooks project's GitHub mirrors
(github.com/global-asp), which hold every story as Markdown plus an image bank:

    <text repo>/<lang>/0004_goat-dog-and-cow.md      "# Title", pages split by "##",
                                                      a final "* License: ..." credits block
    <image repo>/medium/0004/01.jpg, 02.jpg, ...      01 is the cover; an extra last image is a back cover
    <image repo>/metadata/0004.md                     (Pratham only) a description of each illustration

These are snapshot sources keyed to the two repos' latest commits: a new batch
is made only when either repo changes. Repo archives are streamed from GitHub
straight into the bucket, so nothing is written to the local disk.

Each batch produces two datasets:
- ``storybook_pages``: one row per page with its text, illustration, and caption
- ``stories_text``: one row per book with the full story, for story LLM training
"""

from __future__ import annotations

import json
import logging
import re
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import pandas as pd

from primer.data import checks
from primer.data.manifest import Check, StageRun
from primer.data.schemas import STORIES_TEXT, STORYBOOK_PAGES, conform
from primer.data.sources.base import USER_AGENT, Source
from primer.data.text import normalize_text, record_id, sentence_count, text_hash, word_count

log = logging.getLogger(__name__)

LatestCommit = Callable[[str, str], str]
RepoFiles = Callable[[str, str, Callable[[str], bool]], Iterator[tuple[str, bytes]]]

WRITE_CHUNK = 200
STORY_FILENAME = re.compile(r"\d{4}_[^/]+\.md")  # e.g. 0004_goat-dog-and-cow.md


def github_latest_commit(repo: str, branch: str) -> str:
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/commits/{branch}",
                                     headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["sha"]


def github_repo_files(repo: str, ref: str, wanted: Callable[[str], bool]) -> Iterator[tuple[str, bytes]]:
    """Stream a repo archive and yield (path in repo, bytes) for wanted files, without saving it."""
    request = urllib.request.Request(f"https://codeload.github.com/{repo}/tar.gz/{ref}",
                                     headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response, \
            tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = member.name.split("/", 1)[-1]  # drop the "<repo>-<sha>/" top folder
            if wanted(path):
                content = archive.extractfile(member)
                if content is None:  # not a regular file; isfile() above already skips these
                    continue
                yield path, content.read()


# ------------------------------------------------------------------- parsing

def parse_story(markdown: str) -> dict:
    """Split a Global Storybooks Markdown story into title, pages, and credits."""
    sections = re.split(r"^##\s*$", markdown, flags=re.MULTILINE)
    title = next((line[2:].strip() for line in sections[0].splitlines() if line.startswith("# ")), "")
    pages, credits = [], {}
    for section in sections[1:]:
        lines = [line.strip() for line in section.strip().splitlines() if line.strip()]
        if not lines:
            continue
        if any(line.lower().startswith("* license") for line in lines):
            # Credits block. Some stories add a back-cover blurb after it; that isn't a page, so drop it.
            for line in lines:
                if line.startswith("* "):
                    key, _, value = line[2:].partition(":")
                    credits[key.strip().lower()] = value.strip().strip("[]")
            continue
        pages.append(normalize_text(" ".join(lines)))  # source pages are hard-wrapped PDF lines
    return {"title": title, "pages": pages, "credits": credits}


def parse_captions(markdown: str) -> dict[str, str]:
    """Image file -> description, from a Pratham image-bank metadata table."""
    captions = {}
    for line in markdown.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.fullmatch(r"\d+\.jpg", cells[0]):
            captions[cells[0]] = cells[1]
    return captions


def align_images(n_pages: int, images: list[str]) -> tuple[str, list[str | None]]:
    """Match page images to pages. Returns (alignment label, image file per page)."""
    if not images:
        return "no_images", [None] * n_pages
    extra = len(images) - n_pages
    if extra == 1:
        return "cover+pages", list(images[1:])
    if extra == 2:
        return "cover+pages+back", list(images[1:-1])
    if extra == 0:
        return "pages", list(images)
    return "unaligned", [None] * n_pages


# -------------------------------------------------------------------- source

class Storybooks(Source):
    dataset = "storybook_pages"
    snapshot = True

    def __init__(self, config, store, latest_commit: LatestCommit | None = None,
                 repo_files: RepoFiles | None = None):
        super().__init__(config, store)
        self.latest_commit = latest_commit or github_latest_commit
        self.repo_files = repo_files or github_repo_files

    @property
    def language(self) -> str:
        return self.options.get("language", "en")

    # ------------------------------------------------------------------ ingest

    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        text_repo, image_repo = self.options["text_repo"], self.options["image_repo"]
        branch = self.options.get("branch", "master")
        text_sha, image_sha = self.latest_commit(text_repo, branch), self.latest_commit(image_repo, branch)

        state = self.store.read_json(self.state_path, default={})
        run.records_in = 0
        if (state.get("text_sha"), state.get("image_sha")) == (text_sha, image_sha):
            log.info("%s repos unchanged since batch %s; nothing new", self.key, state.get("latest_batch"))
            run.records_out = 0
            return False

        prefix = self.raw_prefix(run.batch_id)
        lang = self.language
        log.info("Downloading %s stories from %s@%s", lang, text_repo, text_sha[:7])
        texts = dict(self.repo_files(  # story files only; skips README.md and the like
            text_repo, text_sha,
            lambda p: p.count("/") == 1 and p.startswith(f"{lang}/") and bool(STORY_FILENAME.fullmatch(p[len(lang) + 1:]))))
        story_ids = {p.split("/")[1][:4] for p in texts}
        self.store.write_many({f"{prefix}/text/{p.split('/')[1]}": data for p, data in texts.items()})

        size = self.options.get("image_size", "medium")

        def wanted_image(path: str) -> bool:
            parts = path.split("/")
            if parts[0] == size and len(parts) == 3:
                return parts[1] in story_ids and parts[2].endswith(".jpg")
            return parts[0] == "metadata" and len(parts) == 2 and parts[1][:4] in story_ids

        log.info("Streaming images for %d stories from %s@%s", len(story_ids), image_repo, image_sha[:7])
        pending: dict[str, bytes] = {}
        n_images = 0
        for path, data in self.repo_files(image_repo, image_sha, wanted_image):
            parts = path.split("/")
            if parts[0] == "metadata":
                pending[f"{prefix}/image_metadata/{parts[1]}"] = data
            else:
                pending[f"{prefix}/images/{parts[1]}/{parts[2]}"] = data
                n_images += 1
            if len(pending) >= WRITE_CHUNK:
                self.store.write_many(pending)
                pending = {}
                log.info("  %s images uploaded", f"{n_images:,}")
        self.store.write_many(pending)

        self.store.write_json(f"{prefix}/_batch.json", {
            "batch_id": run.batch_id,
            "text_repo": text_repo, "text_sha": text_sha,
            "image_repo": image_repo, "image_sha": image_sha,
            "language": lang, "image_size": size,
            "stories": len(texts), "images": n_images,
            "fetched_at": datetime.now(UTC).isoformat(),
        })
        self.store.write_json(self.state_path, {"text_sha": text_sha, "image_sha": image_sha,
                                                "latest_batch": run.batch_id})
        run.records_out = len(texts)
        run.output_uri = self.store.url(prefix)
        return True

    # ------------------------------------------------------------- read raw

    def _books(self, batch_id: str) -> list[dict]:
        prefix = self.raw_prefix(batch_id)
        texts = self.store.read_many(self.store.glob(f"{prefix}/text/*.md"))
        captions = {p.rsplit("/", 1)[-1][:4]: parse_captions(d.decode("utf-8"))
                    for p, d in self.store.read_many(self.store.glob(f"{prefix}/image_metadata/*.md")).items()}
        images: dict[str, list[str]] = {}
        for path in self.store.glob(f"{prefix}/images/*/*.jpg"):
            _, story_id, name = path.rsplit("/", 2)
            images.setdefault(story_id, []).append(name)

        books = []
        for path, data in sorted(texts.items()):
            filename = path.rsplit("/", 1)[-1]
            if not STORY_FILENAME.fullmatch(filename):
                continue
            story_id = filename[:4]
            story = parse_story(data.decode("utf-8"))
            files = sorted(images.get(story_id, []))
            alignment, page_images = align_images(len(story["pages"]), files)
            books.append({
                "story_id": story_id, "slug": filename[5:-3], **story,
                "alignment": alignment, "page_images": page_images, "n_images": len(files),
                "captions": captions.get(story_id, {}),
            })
        return books

    # ---------------------------------------------------------------- validate

    def validate(self, run: StageRun) -> None:
        books = self._books(run.batch_id)
        run.records_in = run.records_out = len(books)
        run.output_uri = self.store.url(self.raw_prefix(run.batch_id))

        with_images = [b for b in books if b["n_images"]]
        aligned = [b for b in with_images if b["alignment"] != "unaligned"]
        image_share = len(with_images) / len(books) if books else 0.0
        aligned_share = len(aligned) / len(with_images) if with_images else 0.0
        lang = self.language
        run.checks += [
            checks.count_between("stories", len(books), self.options.get("min_stories", 300)),
            checks.unique(books, "story_id"),
            checks.max_ratio("parse_errors", books, lambda b: not b["title"], 0.01),
            # A few mirrored stories are just a title and credits; preprocessing skips them.
            checks.max_ratio("empty_stories", books, lambda b: not b["pages"], 0.02),
            checks.max_ratio("missing_credits", books, lambda b: not b["credits"].get("license"), 0.05),
            checks.max_ratio(f"language_is_{lang}", books,
                             lambda b: b["credits"].get("language", lang) != lang, 0.02),
            Check("image_coverage", image_share >= 0.9, f"{image_share:.1%} of stories have images (minimum 90%)"),
            Check("page_image_alignment", aligned_share >= 0.85,
                  f"{aligned_share:.1%} of illustrated stories line up with their pages (minimum 85%); "
                  + ", ".join(f"{k}: {v}" for k, v in Counter(b["alignment"] for b in books).most_common())),
        ]

    # -------------------------------------------------------------- preprocess

    def preprocess(self, run: StageRun) -> None:
        books = self._books(run.batch_id)
        prefix = self.raw_prefix(run.batch_id)
        source_name = self.config.name
        now = pd.Timestamp.now(tz="UTC")
        run.records_in = len(books)

        pages, stories = [], []
        for book in books:
            if not book["pages"]:
                continue
            credits = book["credits"]
            book_id = f"{self.key}-{book['story_id']}"
            book_license = credits.get("license") or self.config.license
            attribution = "; ".join(filter(None, [
                f"Text: {credits['text']}" if credits.get("text") else None,
                f"Illustration: {credits['illustration']}" if credits.get("illustration") else None,
                f"Translation: {credits['translation']}" if credits.get("translation") else None,
                source_name,
            ]))
            n_pages = len(book["pages"])
            for number, (text, image) in enumerate(zip(book["pages"], book["page_images"]), start=1):
                pages.append({
                    "record_id": record_id(self.key, f"{book['story_id']}:{number}"),
                    "book_id": book_id, "title": book["title"],
                    "page_number": number, "n_pages": n_pages,
                    "text": text, "word_count": word_count(text),
                    "image_uri": self.store.url(f"{prefix}/images/{book['story_id']}/{image}") if image else None,
                    "image_caption": book["captions"].get(image) if image else None,
                    "image_alignment": book["alignment"],
                    "author": credits.get("text"), "illustrator": credits.get("illustration"),
                    "translator": credits.get("translation"),
                    "language": credits.get("language", self.language),
                    "license": book_license, "attribution": attribution,
                })
            full_text = "\n\n".join(book["pages"])
            stories.append({
                "record_id": record_id(self.key, book["story_id"]),
                "source_id": book["story_id"], "title": book["title"], "text": full_text,
                "word_count": word_count(full_text), "sentence_count": sentence_count(full_text),
                "text_hash": text_hash(full_text), "language": credits.get("language", self.language),
                "license": book_license, "attribution": attribution,
            })

        common = {"source": self.key, "batch_id": run.batch_id, "ingested_at": now}
        pages_df = conform(pd.DataFrame(pages).assign(reading_level=pd.NA, **common), STORYBOOK_PAGES)
        stories_df = pd.DataFrame(stories).drop_duplicates("text_hash").assign(**common)
        stories_df = conform(stories_df, STORIES_TEXT)

        pages_path = self.processed_path(run.batch_id)
        self.store.write_parquet(pages_path, pages_df)
        self.store.write_parquet(self.processed_path(run.batch_id, "stories_text"), stories_df)
        run.records_out = run.records_total = len(pages_df)
        run.output_uri = self.store.url(pages_path)
        log.info("%s: %d books, %d pages (%d with images)", self.key, len(stories_df), len(pages_df),
                 int(pages_df["image_uri"].notna().sum()))


class AfricanStorybook(Storybooks):
    key = "african_storybook"


class StoryWeaver(Storybooks):
    key = "storyweaver"
