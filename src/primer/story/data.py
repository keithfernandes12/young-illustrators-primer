"""Books, splits and validation requests for the story stage, from a frozen data version.

The steps follow the plan's "Building the pairs and the splits":

1. **One pair per book.** Picture books come only from `storybook_pages` (grouped by source and
   book_id, pages in order); from `stories_text`, only `source == "tinystories"` rows. A picture book's
   `stories_text` row is the same text as its pages, so reading both would count it twice.
2. (The TinyStories hash pool is for training pairs; validation requests do not need it.)
3. **Cross-source exact dedupe** on `text_hash` over the full story, keeping a fixed source order so a
   duplicate keeps its picture-book attribution.
4. **The split is a pure hash** of `book_key = f"{source}/{book_id}"`: `split_of`'s hash
   (`illustration/consistency/experiment.py`), inlined because model areas do not import each other,
   at 80/10/10. Level is never part of the key. It moves to `src/primer/data` under KAN-120.

Validation requests are then drawn from the **val** split only, for ages 3-5. Offline only: this
module reads `primer.data`, which nothing on the request path may do.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from primer.data.text import text_hash
from primer.story.interface import StoryRequest
from primer.story.levels import FK_VERSION, fk_grade, words
from primer.story.presets import LEARN_TO_READ

#: Dedupe keeps the first source in this order, so a duplicate keeps its picture-book attribution.
SOURCE_ORDER = ("african_storybook", "storyweaver", "tinystories")
SPLITS = (("train", 0.8), ("val", 0.9), ("test", 1.0))

#: Places a 3-5 story can be "in". A request's setting is the first one its book mentions, or None.
#: Each must read naturally as "in the <place>"; "home" is left out because "in the home" does not.
SETTINGS = ("garden", "park", "forest", "farm", "beach", "school", "house", "kitchen", "river", "village",
            "sea", "zoo", "market", "jungle", "pond", "tree", "sky")

_STOPWORDS = frozenset(
    ["the", "a", "an", "and", "or", "but", "so", "then", "when", "was", "were", "is", "are", "am", "be", "been", "to", "of", "in", "on", "at", "for", "with", "by", "from", "up", "down", "out", "into", "over", "under", "they", "them", "their", "there", "this", "that", "these", "those", "he", "she", "it", "his", "her", "its", "we", "you", "i", "me", "my", "our", "your", "said", "says", "did", "does", "do", "not", "no", "yes", "very", "too", "all", "one", "two", "can", "could", "would", "will", "just", "like", "went", "came", "got", "get", "had", "has", "have", "what", "who", "where", "why", "how", "him"]
)
_CAPITALISED = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")


def book_key(source: str, book_id: str) -> str:
    """The split key: `source/book_id`, the format `split_of` already uses."""
    return f"{source}/{book_id}"


def hash_fraction(key: str, salt: str = "") -> float:
    """A stable number in [0, 1) for a key: the first 8 bytes of SHA-256, as in `split_of`."""
    return int.from_bytes(hashlib.sha256((salt + key).encode()).digest()[:8], "big") / 2**64


def split_of_book(key: str) -> str:
    """train / val / test at 80/10/10, from the key alone. Never depends on other books or on level."""
    fraction = hash_fraction(key)
    return next(name for name, upper in SPLITS if fraction < upper)


def books_from_frames(pages: pd.DataFrame, stories: pd.DataFrame) -> pd.DataFrame:
    """One row per book: picture books from `pages`, TinyStories from `stories` (steps 1 and 3).

    Returns columns book_key, source, book_id, title, pages (tuple of str), text, text_hash, license,
    attribution, split. Dropped exact duplicates are counted in ``attrs``: ``dropped_duplicates`` (all),
    ``dropped_cross_source`` and ``dropped_within_source``.
    """
    rows: list[dict[str, Any]] = []
    if not pages.empty:
        ordered = pages.sort_values(["source", "book_id", "page_number"])
        for (source, book_id), group in ordered.groupby(["source", "book_id"], sort=False):
            texts = tuple(str(t) for t in group["text"] if isinstance(t, str) and t.strip())
            if not texts:
                continue
            first = group.iloc[0]
            rows.append({"source": source, "book_id": str(book_id), "title": first.get("title"),
                         "pages": texts, "text": "\n\n".join(texts),
                         "license": first.get("license"), "attribution": first.get("attribution")})
    if not stories.empty:
        tiny = stories[stories["source"] == "tinystories"]
        for record in tiny.itertuples(index=False):
            rows.append({"source": record.source, "book_id": str(record.source_id), "title": record.title,
                         "pages": (str(record.text),), "text": str(record.text),
                         "license": record.license, "attribution": record.attribution})
    books = pd.DataFrame(rows, columns=["source", "book_id", "title", "pages", "text", "license", "attribution"])
    if books.empty:
        books = books.assign(book_key=[], text_hash=[], split=[])
        books.attrs.update(dropped_duplicates=0, dropped_cross_source=0, dropped_within_source=0)
        return books
    books["book_key"] = [book_key(s, b) for s, b in zip(books["source"], books["book_id"])]
    books["text_hash"] = books["text"].map(text_hash)
    rank = {source: i for i, source in enumerate(SOURCE_ORDER)}
    books["_rank"] = books["source"].map(lambda s: rank.get(s, len(rank)))
    books = books.sort_values(["_rank", "book_key"], kind="stable")
    # Count the two kinds of duplicate apart (review F10): one whose kept copy is in another source,
    # and one repeated within its own source (two TinyStories with the same text).
    duplicate = books.duplicated("text_hash", keep="first")
    kept_source = books.drop_duplicates("text_hash", keep="first").set_index("text_hash")["source"]
    cross = duplicate & (books["text_hash"].map(kept_source) != books["source"])
    books = books[~duplicate].drop(columns="_rank")
    books["split"] = books["book_key"].map(split_of_book)
    books = books.reset_index(drop=True)
    books.attrs["dropped_duplicates"] = int(duplicate.sum())
    books.attrs["dropped_cross_source"] = int(cross.sum())
    books.attrs["dropped_within_source"] = int(duplicate.sum() - cross.sum())
    return books


@dataclass(frozen=True)
class ValidationRequest:
    """One request for model comparison, with where it came from (for credit and slicing)."""

    request_id: str
    book_key: str
    source: str
    age: int
    target_words: tuple[str, ...]
    character_name: str | None
    setting: str | None
    #: The **target** level for the story: from the child's age, by the golden set's rule
    #: (`target_level_for_age`), not from the source book.
    reading_level: float
    #: The source book's own `fk_grade`, kept for transparency and slicing.
    book_level: float
    #: How `book_level` was computed.
    level_source: str
    license: str | None
    attribution: str | None
    #: Where each target word came from, in the same order: a word list's `list_name` ("dolch",
    #: "fry"), or "book" when no vocabulary was given and the word is from the book's own text. With a
    #: vocabulary (the hard Learn to Read filter) it is never "book".
    target_word_lists: tuple[str, ...] = ()

    def to_request(self) -> StoryRequest:
        return StoryRequest(
            prompt="", age=self.age, target_words=self.target_words, character_name=self.character_name,
            setting=self.setting, age_mode=LEARN_TO_READ.name, reading_level=self.reading_level,
        )


def character_of(text: str) -> str | None:
    """The most frequent capitalised word not at a sentence start, if it appears at least twice."""
    counts = Counter(m for m in _CAPITALISED.findall(text) if m.lower() not in _STOPWORDS)
    if not counts:
        return None
    name, n = counts.most_common(1)[0]
    return name if n >= 2 else None


def setting_of(text: str) -> str | None:
    lowered = {w.lower() for w in words(text)}
    for place in SETTINGS:
        if place in lowered:
            return f"the {place}"
    return None


#: The Learn to Read word lists (plan, "Vocabulary-aware prompts"): Dolch pre-primer to grade 1, plus
#: Fry's first 300. These are the `level` values `primer.data.sources.word_lists` writes.
LEARN_TO_READ_LEVELS = frozenset({"pre-primer", "primer", "grade1", "fry_100", "fry_200", "fry_300"})


def learn_to_read_vocabulary(lists: pd.DataFrame) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """The 3-5 vocabulary from the `word_lists` sight-word rows (review F2).

    Returns `{word: list_name}` for the Learn to Read levels only, and one credit row per list used
    (`list_name`, `license`, `attribution`, the levels taken, and the word count), for the manifest.
    """
    if lists.empty:
        return {}, []
    chosen = lists[lists["level"].isin(LEARN_TO_READ_LEVELS)]
    vocabulary: dict[str, str] = {}
    for word, list_name in zip(chosen["word"], chosen["list_name"]):
        vocabulary.setdefault(str(word).lower(), str(list_name))
    credits = []
    for list_name, group in chosen.groupby("list_name", sort=True):
        credits.append({"list_name": str(list_name), "license": group["license"].iloc[0],
                        "attribution": group["attribution"].iloc[0],
                        "levels": sorted(str(v) for v in group["level"].unique()), "words": len(group)})
    return vocabulary, credits


def target_words_of(text: str, key: str, n: int, vocabulary: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Up to `n` content words the story uses, chosen by a hash of the book key (deterministic).

    With a `vocabulary`, it is a **hard filter** (Diya, PR #36 review question): only words in the
    vocabulary qualify. A book with fewer than `n` of them yields fewer than `n` words, and the caller
    skips it, so a 3-5 request never falls back to a harder word from its source book. Without a
    vocabulary, any content word qualifies. Because the words come from the book's own text, a story
    that uses them exists: inclusion is satisfiable by construction.
    """
    seen: dict[str, None] = {}
    for token in words(text):
        lowered = token.lower()
        if len(lowered) >= 3 and lowered.isalpha() and lowered not in _STOPWORDS and token[0].islower():
            seen.setdefault(lowered, None)
    candidates = [w for w in seen if w in vocabulary] if vocabulary else list(seen)
    candidates.sort(key=lambda word: hash_fraction(word, salt=key))
    return tuple(candidates[:n])


#: Source books above this whole-story FK grade are not used for 3-5 requests. Measured on v1.1 val
#: (2026-11-15): only 12 of 264 val books are at <= 1.5, but 57 are at <= 3.0. The book only supplies
#: the target words, the character and the setting; the target level comes from the child's age.
MAX_SOURCE_BOOK_LEVEL = 3.0


def target_level_for_age(age: int) -> float:
    """The story's target FK grade for a child of `age`, by the golden set's rule.

    `scripts/gen_golden_scenarios.py` uses `max(0, age - 4)` plus a random ±0.3 jitter; validation
    requests use the centre of that rule, so ages 3, 4 and 5 target 0.0, 0.0 and 1.0.
    """
    return max(0.0, float(age) - 4.0)


def build_validation_requests(
    books: pd.DataFrame, n: int = 30, vocabulary: Mapping[str, str] | None = None,
    max_book_level: float = MAX_SOURCE_BOOK_LEVEL,
) -> list[ValidationRequest]:
    """`n` requests for ages 3-5 from the **val** split, balanced across sources, deterministic.

    - Source books qualify at whole-story FK grade <= `max_book_level`.
    - The target level comes from the child's age (`target_level_for_age`), not from the book.
    - With a `vocabulary` (the Learn to Read lists), target words come **only** from it: a hard
      filter, so a book with fewer than two list words is skipped. Words always come from the book's
      own text.
    - Books are taken round-robin across sources in hash order, so one source cannot fill the set; a
      book that yields fewer than two target words is skipped and the next one is taken.
    - Ages cycle 3, 4, 5 within each source (starting at a different age per source), so age and
      source are not confounded.
    """
    val = books[books["split"] == "val"].copy()
    val["level"] = val["text"].map(fk_grade)
    val = val[val["level"] <= max_book_level]
    val["_order"] = val["book_key"].map(lambda key: hash_fraction(key, salt="requests:"))
    queues = {source: list(group.sort_values("_order").itertuples(index=False))
              for source, group in val.groupby("source", sort=True)}
    ordered: list[Any] = []
    while any(queues.values()):
        for source in sorted(queues):
            if queues[source]:
                ordered.append(queues[source].pop(0))
    requests: list[ValidationRequest] = []
    # Age cycles *within each source*, from a different start per source (review F1). A single cycle
    # over the round-robin would lock age to source -- african_storybook always 3, storyweaver 4,
    # tinystories 5 -- and the `age` and `source` slices would measure the same thing.
    offset = {source: i for i, source in enumerate(sorted(queues))}
    per_source: Counter[str] = Counter()
    for book in ordered:
        if len(requests) >= n:
            break
        index = len(requests)
        target = target_words_of(book.text, book.book_key, 2 + index % 2, vocabulary)
        if len(target) < 2:
            continue
        age = 3 + (per_source[book.source] + offset[book.source]) % 3
        per_source[book.source] += 1
        requests.append(ValidationRequest(
            request_id=hashlib.sha1(book.book_key.encode()).hexdigest()[:12],
            book_key=book.book_key, source=book.source, age=age, target_words=target,
            character_name=character_of(book.text), setting=setting_of(book.text),
            reading_level=target_level_for_age(age), book_level=float(book.level), level_source=FK_VERSION,
            license=book.license, attribution=book.attribution,
            target_word_lists=tuple((vocabulary or {}).get(word, "book") for word in target),
        ))
    return requests


def write_requests(requests: list[ValidationRequest], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(asdict(request), sort_keys=True) + "\n")


def read_requests(path: Path) -> list[ValidationRequest]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            raw["target_words"] = tuple(raw["target_words"])
            raw["target_word_lists"] = tuple(raw.get("target_word_lists", ()))
            out.append(ValidationRequest(**raw))
    return out


def smallest_files(version_record: dict[str, Any], dataset: str, source: str, n: int) -> list[dict[str, Any]]:
    """The `n` smallest files of one source in a version record (`n <= 0` means all), smallest first.

    Used to read a small, exactly-pinned TinyStories subset for validation requests instead of all
    ~931 MB of it (Diya's choice, 2026-11-15). Each file keeps its GCS generation, so the subset is as
    frozen as the version itself. Splits are unaffected: they are a pure hash of each book's key.
    """
    files = version_record["datasets"][dataset]["sources"][source]["files"]
    ordered = sorted(files, key=lambda f: (int(f.get("size") or 0), f["path"]))
    return ordered if n <= 0 else ordered[:n]


def load_books(store: Any, version: str, tinystories_files: int = 1) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Books at a frozen data version: every picture book, plus the `tinystories_files` smallest
    TinyStories files (0 = all, ~931 MB), each read at its pinned GCS generation.

    Returns the books, the TinyStories files read, and their row count. `store` is required: there is
    no default, so nothing here can fall back to `PRIMER_DATA_URI` or the bucket (review F4).
    """
    from primer.data.catalog import read_dataset as read
    from primer.data.catalog import read_version

    record = read_version(version, store)
    tiny_files = smallest_files(record, "stories_text", "tinystories", tinystories_files)
    tiny_frames = [store.read_parquet(f["path"], generation=f.get("generation")) for f in tiny_files]
    tiny = pd.concat(tiny_frames, ignore_index=True) if tiny_frames else pd.DataFrame()
    return books_from_frames(read("storybook_pages", store, version=version), tiny), tiny_files, len(tiny)


def read_dataset(version: str, split: str, store: Any, tinystories_files: int = 1) -> pd.DataFrame:
    """The books of one split at a frozen data version (the model-development section 2 entry point).

    `store` is required (review F4), and TinyStories is the same pinned subset `main` reads.
    """
    books, _, _ = load_books(store, version, tinystories_files)
    return books[books["split"] == split].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Build the validation requests: `python -m primer.story.data --data-uri gs://... --out ...`."""
    import argparse

    from primer.data.catalog import read_dataset as read
    from primer.data.storage import Store

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data-uri", required=True, help="where the frozen version lives (read only)")
    parser.add_argument("--version", default="v1.1")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--tinystories-files", type=int, default=1,
                        help="read only the N smallest TinyStories files of the version (0 = all, ~931 MB)")
    parser.add_argument("--out", type=Path, default=Path(".local-data/story/validation_requests.jsonl"))
    args = parser.parse_args(argv)

    store = Store(args.data_uri)
    books, tiny_files, tiny_rows = load_books(store, args.version, args.tinystories_files)
    lists = read("word_lists", store, sources=["sight_words"], version=args.version)
    vocabulary, list_credits = learn_to_read_vocabulary(lists)
    if not vocabulary:
        # An empty list would silently turn the hard Learn to Read filter off and let requests take
        # any word from their (up to grade 3) source books. Refuse instead.
        parser.error(f"no Learn to Read words in word_lists/sight_words at {args.version}; "
                     "refusing to build 3-5 requests without the hard filter")
    requests = build_validation_requests(books, n=args.n, vocabulary=vocabulary)
    write_requests(requests, args.out)

    val = books[books["split"] == "val"]
    manifest = {
        "data_version": args.version,
        "level_source": FK_VERSION,
        "tinystories_files": [{k: f.get(k) for k in ("path", "generation", "size")} for f in tiny_files],
        "tinystories_rows_read": tiny_rows,
        "vocabulary": {"levels": sorted(LEARN_TO_READ_LEVELS), "words": len(vocabulary), "lists": list_credits},
        "target_words_by_list": dict(sorted(Counter(src for r in requests for src in r.target_word_lists).items())),
        "books": len(books),
        "books_by_source": {str(k): int(v) for k, v in books["source"].value_counts().sort_index().items()},
        "dropped_cross_source_duplicates": int(books.attrs["dropped_cross_source"]),
        "dropped_within_source_duplicates": int(books.attrs["dropped_within_source"]),
        "val_books_by_source": {str(k): int(v) for k, v in val["source"].value_counts().sort_index().items()},
        "requests": len(requests),
        "requests_by_source": dict(sorted(Counter(r.source for r in requests).items())),
        "requests_by_age": {str(k): v for k, v in sorted(Counter(r.age for r in requests).items())},
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {len(requests)} requests to {args.out} and the manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
