"""African Storybook / StoryWeaver sources, run against fake GitHub repos and a temp store."""

import pytest

from primer.data import pipeline
from primer.data.catalog import read_dataset
from primer.data.config import SourceConfig
from primer.data.manifest import ValidationFailed
from primer.data.sources.storybooks import StoryWeaver, align_images, parse_captions, parse_story

STORY = """# Goat, Dog, and Cow

##
Goat, Dog, and Cow
were great friends.

##

##
When they reached the
end of their journey.

##
* License: [CC-BY]
* Text: Fabian Wakholi
* Illustration: Marleen Visser
* Language: en

A back-cover blurb that is not a page.
"""

CAPTIONS = """Image metadata for illustrations.

Image | Description | Illustrator | License
----- | ----------- | ----------- | -------
01.jpg | Cover | A | CC-BY
02.jpg | Goat and dog | A | CC-BY
03.jpg | A taxi | A | CC-BY
"""


def test_parse_story_joins_wrapped_lines_and_reads_credits():
    story = parse_story(STORY)
    assert story["title"] == "Goat, Dog, and Cow"
    assert story["pages"] == ["Goat, Dog, and Cow were great friends.", "When they reached the end of their journey."]
    assert story["credits"] == {"license": "CC-BY", "text": "Fabian Wakholi",
                                "illustration": "Marleen Visser", "language": "en"}


def test_parse_captions():
    assert parse_captions(CAPTIONS) == {"01.jpg": "Cover", "02.jpg": "Goat and dog", "03.jpg": "A taxi"}


@pytest.mark.parametrize("n_images, label, expected", [
    (3, "cover+pages", ["02.jpg", "03.jpg"]),
    (4, "cover+pages+back", ["02.jpg", "03.jpg"]),
    (2, "pages", ["01.jpg", "02.jpg"]),
    (6, "unaligned", [None, None]),
    (0, "no_images", [None, None]),
])
def test_align_images(n_images, label, expected):
    images = [f"{i:02d}.jpg" for i in range(1, n_images + 1)]
    assert align_images(2, images) == (label, expected)


def story_md(title, n_pages, lang="en"):
    pages = "\n".join(f"##\nPage {i} of {title} is here.\n" for i in range(1, n_pages + 1))
    return f"# {title}\n\n{pages}\n##\n* License: [CC-BY]\n* Text: Writer\n* Illustration: Artist\n* Language: {lang}\n"


def fake_repos(n_stories=4, images_per_story=None, shas=("t1", "i1")):
    """Text repo with English and French stories; image repo with medium/ and large/ images."""
    images_per_story = images_per_story or {}
    text = {"en/README.md": b"# English stories\n\n##\nNot a story.\n"}
    for i in range(1, n_stories + 1):
        text[f"en/{i:04d}_story-{i}.md"] = story_md(f"Story {i}", 2).encode()
        text[f"fr/{i:04d}_histoire-{i}.md"] = story_md(f"Histoire {i}", 2, "fr").encode()
    image = {"README.md": b"readme", "metadata/0001.md": CAPTIONS.encode()}
    for i in range(1, n_stories + 1):
        for j in range(1, images_per_story.get(i, 3) + 1):
            image[f"medium/{i:04d}/{j:02d}.jpg"] = f"img {i}-{j}".encode()
            image[f"large/{i:04d}/{j:02d}.jpg"] = b"big"
    repos = {"o/text": text, "o/images": image}
    current = {"o/text": shas[0], "o/images": shas[1]}

    def latest_commit(repo, branch):
        return current[repo]

    def repo_files(repo, ref, wanted):
        assert ref == current[repo]
        return ((p, d) for p, d in repos[repo].items() if wanted(p))

    return latest_commit, repo_files, current


def make_source(store, n_stories=4, **kw):
    latest_commit, repo_files, current = fake_repos(n_stories, **kw)
    cfg = SourceConfig(key="storyweaver", name="StoryWeaver", license="CC-BY-4.0", attribution="SW",
                       ingest={"text_repo": "o/text", "image_repo": "o/images", "min_stories": 3})
    return StoryWeaver(cfg, store, latest_commit=latest_commit, repo_files=repo_files), current


@pytest.fixture
def batch_ids(monkeypatch):
    ids = iter(f"2026101{i}T000000Z" for i in range(10))
    monkeypatch.setattr(pipeline, "new_batch_id", lambda: next(ids))


def test_storybooks_end_to_end(store, batch_ids):
    # story 2 has a back cover, story 3 has too many images to line up
    source, current = make_source(store, n_stories=10, images_per_story={2: 4, 3: 7})
    batch = pipeline.run_all(source)

    raw = f"raw/storyweaver/{batch}"
    assert len(store.glob(f"{raw}/text/*.md")) == 10                # English only
    assert len(store.glob(f"{raw}/images/*/*.jpg")) == 3 * 8 + 4 + 7  # medium only
    assert store.glob(f"{raw}/image_metadata/*.md") == [f"{raw}/image_metadata/0001.md"]

    pages = store.read_parquet(source.processed_path(batch))
    assert len(pages) == 20
    first = pages[pages["book_id"] == "storyweaver-0001"].sort_values("page_number")
    assert [uri.rsplit("/", 2)[-2:] for uri in first["image_uri"]] == [["0001", "02.jpg"], ["0001", "03.jpg"]]
    assert first["image_caption"].tolist() == ["Goat and dog", "A taxi"]
    assert set(pages.groupby("book_id")["image_alignment"].first()) == {"cover+pages", "cover+pages+back", "unaligned"}
    assert pages.loc[pages["book_id"] == "storyweaver-0003", "image_uri"].isna().all()
    assert pages["attribution"].iloc[0] == "Text: Writer; Illustration: Artist; StoryWeaver"

    books = read_dataset("stories_text", store)
    assert len(books) == 10 and books["text"].iloc[0] == "Page 1 of Story 1 is here.\n\nPage 2 of Story 1 is here."

    assert pipeline.ingest(source) is None      # repos unchanged
    current["o/images"] = "i2"                   # image repo moved on
    assert pipeline.run_all(source) is not None
    assert len(read_dataset("storybook_pages", store)) == 20  # latest snapshot only


def test_storybooks_validation_fails_when_images_missing(store, batch_ids):
    source, _ = make_source(store, images_per_story={1: 0, 2: 0, 3: 0})
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed, match="image_coverage"):
        pipeline.validate(source, batch)
