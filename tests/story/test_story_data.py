"""Books, the pure-hash split and validation requests, on synthetic frames (never the bucket)."""

from __future__ import annotations

import pandas as pd
import pytest

from primer.data.schemas import STORIES_TEXT, STORYBOOK_PAGES, conform
from primer.illustration.consistency.experiment import split_of as illustration_split_of
from primer.story.data import (
    MAX_SOURCE_BOOK_LEVEL,
    SPLITS,
    book_key,
    books_from_frames,
    build_validation_requests,
    character_of,
    hash_fraction,
    read_requests,
    setting_of,
    smallest_files,
    split_of_book,
    target_level_for_age,
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
    # PR #36 review F7: the picture book's stories_text row must never be read, not read and then
    # removed as a duplicate. Without this, the test passed only through the dedupe's tie-break.
    assert books.attrs["dropped_duplicates"] == 0


def test_an_exact_duplicate_across_sources_keeps_the_picture_book():
    pages = pages_frame({("african_storybook", "asp-9"): SIMPLE})
    stories = stories_frame([("tinystories", "3", "\n\n".join(SIMPLE)), ("tinystories", "4", "A cat naps.")])
    books = books_from_frames(pages, stories)
    assert books.attrs["dropped_duplicates"] == 1
    assert books.attrs["dropped_cross_source"] == 1 and books.attrs["dropped_within_source"] == 0
    assert set(books["source"]) == {"african_storybook", "tinystories"}
    assert "tinystories/3" not in set(books["book_key"])


def test_a_duplicate_within_one_source_is_not_counted_as_cross_source():
    """Review F10: two TinyStories with the same text are a within-source duplicate."""
    stories = stories_frame([("tinystories", "1", "A cat naps."), ("tinystories", "2", "A cat naps.")])
    books = books_from_frames(pages_frame({}), stories)
    assert len(books) == 1
    assert books.attrs["dropped_within_source"] == 1 and books.attrs["dropped_cross_source"] == 0


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


def test_every_books_split_is_the_unsalted_hash_of_its_key():
    """PR #36 review F1: pins the split to `book_key` alone, row by row. Fails if the level, the text
    or a salt enters the split hash inside books_from_frames."""
    books = _corpus()
    assert (books["split"] == books["book_key"].map(split_of_book)).all()


@pytest.mark.parametrize(
    "text,level_changes",
    [("Pip naps.", False),  # different text, same level (0.0)
     ("Extraordinarily sophisticated vocabulary demonstrates complexity.", True)],  # level 0.0 -> 43.0
    ids=["text-only", "text-and-level"],
)
def test_changing_a_books_text_or_level_never_moves_it(text, level_changes):
    """The plan's S2 acceptance: the same key with different text, or a different level, keeps its split."""
    key = "storyweaver/storyweaver-1"
    base = books_from_frames(pages_frame({("storyweaver", "storyweaver-1"): SIMPLE}), stories_frame([]))
    changed = books_from_frames(pages_frame({("storyweaver", "storyweaver-1"): [text]}), stories_frame([]))
    assert (fk_grade(text) != fk_grade("\n\n".join(SIMPLE))) is level_changes, "the fixture must do what its id says"
    assert base.set_index("book_key").loc[key, "split"] == changed.set_index("book_key").loc[key, "split"]


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
        assert request.book_level == fk_grade(texts[request.book_key]) <= MAX_SOURCE_BOOK_LEVEL
        assert request.reading_level == target_level_for_age(request.age), "the target comes from age"
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


@pytest.mark.parametrize("age,level", [(3, 0.0), (4, 0.0), (5, 1.0)])
def test_target_level_follows_the_golden_age_rule(age, level):
    """The centre of `max(0, age - 4) ± 0.3` from scripts/gen_golden_scenarios.py."""
    assert target_level_for_age(age) == level


def test_books_above_the_level_cut_are_never_used():
    hard = ["Extraordinarily complicated vocabulary demonstrates considerable sophistication throughout."] * 3
    books = books_from_frames(pages_frame({("storyweaver", f"storyweaver-{i}"): hard for i in range(60)}),
                              stories_frame([]))
    assert (books["text"].map(fk_grade) > MAX_SOURCE_BOOK_LEVEL).all()
    assert build_validation_requests(books, n=30) == []


def test_age_is_not_locked_to_source():
    """Review F1: one age cycle over the round-robin gave each source a single age."""
    requests = build_validation_requests(_corpus(), n=18)
    ages_by_source: dict[str, list[int]] = {}
    for request in requests:
        ages_by_source.setdefault(request.source, []).append(request.age)
    assert len(ages_by_source) == 3
    # A source with at least three requests must see every age; one with two must see two ages.
    for source, ages in ages_by_source.items():
        assert len(set(ages)) == min(3, len(ages)), (source, ages)


def test_a_book_without_enough_words_is_skipped_not_counted():
    """Review F5: books with fewer than two usable target words must be skipped, and the next book
    taken, so they cannot shrink the set below n. The short books here have no eligible word at all
    ("The end." / "I am."), so the skip path really runs; turning `continue` into `break` fails this."""
    # Distinct texts (the digits keep them apart under the exact dedupe) with at most one usable word:
    # "end" is the only eligible word in "The end 7.", and "I am 7." has none.
    pages = pages_frame({("storyweaver", f"storyweaver-{i}"): [f"The end {i}."] for i in range(60)}
                        | {("african_storybook", f"asp-{i}"): ["Ama runs to the farm.", f"Ama feeds hen {i}.",
                                                               "The hen is happy."] for i in range(60)})
    stories = stories_frame([("tinystories", str(i), f"I am {i}.") for i in range(60)]
                            + [("tinystories", str(100 + i), f"Tom has a big dog {i}. The dog runs to Tom.")
                               for i in range(60)])
    books = books_from_frames(pages, stories)
    short = {key for key, text in zip(books["book_key"], books["text"])
             if text.startswith(("The end ", "I am "))}
    val_short = short & set(books.loc[books["split"] == "val", "book_key"])
    assert val_short, "the fixture must put some short books in val, or this test proves nothing"
    requests = build_validation_requests(books, n=8)
    assert len(requests) == 8
    assert not {r.book_key for r in requests} & short


def test_smallest_files_picks_by_size_and_keeps_generations():
    record = {"datasets": {"stories_text": {"sources": {"tinystories": {"files": [
        {"path": "b.parquet", "size": 26_000_000, "generation": "2"},
        {"path": "a.parquet", "size": 1_012_081, "generation": "1"},
        {"path": "c.parquet", "size": 1_018_684, "generation": "3"},
    ]}}}}}
    assert [f["path"] for f in smallest_files(record, "stories_text", "tinystories", 1)] == ["a.parquet"]
    assert [f["generation"] for f in smallest_files(record, "stories_text", "tinystories", 2)] == ["1", "3"]
    assert len(smallest_files(record, "stories_text", "tinystories", 0)) == 3, "0 means every file"


@pytest.mark.parametrize("text,name", [("Then Zip flew. Zip was happy. Look, said Zip.", "Zip"),
                                       ("The cat sat.", None)])
def test_character_of(text, name):
    assert character_of(text) == name


def test_setting_of():
    assert setting_of("They walk to the Park today.") == "the park"
    assert setting_of("They talk.") is None
