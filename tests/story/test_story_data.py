"""Books, the pure-hash split and validation requests, on synthetic frames (never the bucket)."""

from __future__ import annotations

import pandas as pd
import pytest

from primer.data.schemas import STORIES_TEXT, STORYBOOK_PAGES, conform
from primer.illustration.consistency.experiment import split_of as illustration_split_of
from primer.story.data import (
    SPLITS,
    book_key,
    books_from_frames,
    build_validation_requests,
    character_of,
    hash_fraction,
    read_requests,
    setting_of,
    split_of_book,
    write_requests,
)
from primer.story.levels import FK_VERSION, fk_grade

SIMPLE = ["Mia has a red ball.", "Mia and Sam play in the park.", "The ball goes up.", "Mia is glad."]


def pages_frame(books: dict[tuple[str, str], list[str]]) -> pd.DataFrame:
    rows = []
    for (source, book_id), texts in books.items():
        for number, text in enumerate(texts, 1):
            rows.append({"record_id": f"{book_id}:{number}", "source": source, "book_id": book_id,
                         "title": f"Book {book_id}", "page_number": number, "n_pages": len(texts), "text": text,
                         "word_count": len(text.split()), "language": "en", "license": "CC-BY-4.0",
                         "attribution": f"{source} {book_id}"})
    return conform(pd.DataFrame(rows), STORYBOOK_PAGES)


def stories_frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """(source, source_id, text) rows."""
    return conform(pd.DataFrame([
        {"record_id": f"{s}:{i}", "source": s, "source_id": i, "title": None, "text": t,
         "word_count": len(t.split()), "sentence_count": 1, "text_hash": "", "language": "en",
         "license": "CDLA-Sharing-1.0", "attribution": "TinyStories"} for s, i, t in rows
    ]), STORIES_TEXT)


def test_a_picture_book_is_one_pair_even_though_it_is_in_both_datasets():
    """Its stories_text row is its pages joined; reading both would count it twice (review F3)."""
    pages = pages_frame({("storyweaver", "storyweaver-1"): SIMPLE})
    stories = stories_frame([("storyweaver", "1", "\n\n".join(SIMPLE)), ("tinystories", "7", "Tom has a dog.")])
    books = books_from_frames(pages, stories)
    assert sorted(books["book_key"]) == ["storyweaver/storyweaver-1", "tinystories/7"]
    assert books.loc[books["source"] == "storyweaver", "pages"].iloc[0] == tuple(SIMPLE)


def test_an_exact_duplicate_across_sources_keeps_the_picture_book():
    pages = pages_frame({("african_storybook", "asp-9"): SIMPLE})
    stories = stories_frame([("tinystories", "3", "\n\n".join(SIMPLE)), ("tinystories", "4", "A cat naps.")])
    books = books_from_frames(pages, stories)
    assert books.attrs["dropped_duplicates"] == 1
    assert set(books["source"]) == {"african_storybook", "tinystories"}
    assert "tinystories/3" not in set(books["book_key"])


def test_the_split_uses_split_ofs_exact_hash():
    """Same hash as the illustration area's split_of, so a book lands in the same place in both."""
    for key in ("storyweaver/storyweaver-1", "african_storybook/asp-2", "tinystories/99"):
        fraction = hash_fraction(key)
        assert illustration_split_of(key, fraction + 1e-12) == "val"
        assert illustration_split_of(key, max(fraction - 1e-12, 0.0)) == "test"


def test_a_books_split_never_depends_on_other_books():
    """A pure hash, no strata: adding books or changing levels cannot move a book (review F1)."""
    small = books_from_frames(pages_frame({("storyweaver", "storyweaver-1"): SIMPLE}), stories_frame([]))
    many = books_from_frames(
        pages_frame({("storyweaver", "storyweaver-1"): SIMPLE,
                     **{("storyweaver", f"storyweaver-{i}"): [f"Book {i} is about a frog."] for i in range(2, 60)}}),
        stories_frame([("tinystories", str(i), f"Story {i} has a cat named Bo.") for i in range(200)]),
    )
    key = "storyweaver/storyweaver-1"
    assert small.set_index("book_key").loc[key, "split"] == many.set_index("book_key").loc[key, "split"]


def test_split_ratios_are_close_to_80_10_10():
    keys = [book_key("tinystories", str(i)) for i in range(1000)]
    counts = pd.Series([split_of_book(k) for k in keys]).value_counts(normalize=True)
    for name, expected in zip((s for s, _ in SPLITS), (0.8, 0.1, 0.1)):
        assert abs(counts.get(name, 0.0) - expected) <= 0.02, counts.to_dict()


def _corpus(n_tiny: int = 300) -> pd.DataFrame:
    pages = pages_frame({("storyweaver", f"storyweaver-{i}"): SIMPLE[:-1] + [f"Mia sees frog {i}."]
                         for i in range(40)}
                        | {("african_storybook", f"asp-{i}"): [f"Ama runs to the farm {i}.", "Ama feeds the hen.",
                                                               "The hen is happy."] for i in range(40)})
    stories = stories_frame([("tinystories", str(i), f"Tom has a big dog {i}. The dog runs to Tom. Tom pats it.")
                             for i in range(n_tiny)])
    return books_from_frames(pages, stories)


def test_requests_come_only_from_val_and_are_well_formed():
    books = _corpus()
    requests = build_validation_requests(books, n=30)
    assert 0 < len(requests) <= 30
    val_keys = set(books.loc[books["split"] == "val", "book_key"])
    texts = books.set_index("book_key")["text"]
    for request in requests:
        assert request.book_key in val_keys
        assert request.age in (3, 4, 5)
        assert 2 <= len(request.target_words) <= 3
        for word in request.target_words:
            assert word in texts[request.book_key].lower(), "target words come from the book itself"
        assert request.reading_level == fk_grade(texts[request.book_key]) <= 1.5
        assert request.level_source == FK_VERSION
        assert request.license and request.attribution
        story_request = request.to_request()
        assert story_request.age_mode == "learn_to_read" and story_request.target_words == request.target_words


def test_requests_are_balanced_across_sources_and_deterministic():
    """Round-robin by source: the first request from each source comes before any source's second.

    TinyStories outnumbers the picture books here 300 to 80, so taking requests in plain hash order
    would fill the set with TinyStories; this fails if the round-robin is removed.
    """
    books = _corpus()
    first = build_validation_requests(books, n=12)
    assert first == build_validation_requests(books, n=12)
    sources_in_val = set(books.loc[books["split"] == "val", "source"])
    assert len(sources_in_val) == 3, "the synthetic corpus should put every source in val"
    assert {r.source for r in first[:3]} == sources_in_val


def test_requests_round_trip_through_jsonl(tmp_path):
    requests = build_validation_requests(_corpus(), n=6)
    path = tmp_path / "requests.jsonl"
    write_requests(requests, path)
    assert read_requests(path) == requests


@pytest.mark.parametrize("text,name", [("Then Zip flew. Zip was happy. Look, said Zip.", "Zip"),
                                       ("The cat sat.", None)])
def test_character_of(text, name):
    assert character_of(text) == name


def test_setting_of():
    assert setting_of("They walk to the Park today.") == "the park"
    assert setting_of("They talk.") is None
