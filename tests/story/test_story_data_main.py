"""The request builder's CLI, its vocabulary and its bucket boundary, against a fake local store.

PR #36 review F2, F3 and F4. Everything runs on `Store(tmp_path)` with a hand-written
`metadata/versions/v1.1.json`, so the version-pinned read, the smallest-file TinyStories subset, the
vocabulary build and the manifest are all exercised without the bucket or the network. A local
Store ignores GCS generations, which is what makes this possible.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from primer.data.schemas import STORIES_TEXT, STORYBOOK_PAGES, WORD_LISTS, conform
from primer.data.storage import Store
from primer.story.data import (
    LEARN_TO_READ_LEVELS,
    learn_to_read_vocabulary,
    main,
    read_dataset,
    read_requests,
    target_words_of,
)

#: Rows for the fake sight-word lists: eight words at Learn to Read levels (Dolch pre-primer and
#: grade 1, Fry 100 and 300), and four outside them (Dolch grades 2 and 3, Fry 400 and 900).
WORDS = [
    ("dolch", "play", "pre-primer"), ("dolch", "jump", "pre-primer"), ("dolch", "funny", "pre-primer"),
    ("dolch", "little", "pre-primer"), ("dolch", "yellow", "pre-primer"), ("dolch", "ride", "grade1"),
    ("dolch", "always", "grade2"), ("dolch", "because", "grade3"),
    ("fry", "water", "fry_100"), ("fry", "under", "fry_300"),
    ("fry", "example", "fry_400"), ("fry", "machine", "fry_900"),
]


def word_lists_frame() -> pd.DataFrame:
    return conform(pd.DataFrame([
        {"record_id": f"{name}:{word}", "source": "sight_words", "list_name": name, "word": word, "level": level,
         "rank": i, "license": f"{name}-license", "attribution": f"{name} list"}
        for i, (name, word, level) in enumerate(WORDS)
    ]), WORD_LISTS)


def test_the_vocabulary_is_only_the_learn_to_read_levels():
    """F2: Dolch pre-primer to grade 1 and Fry's first 300, never grade 2-3 or Fry beyond 300."""
    vocabulary, credits = learn_to_read_vocabulary(word_lists_frame())
    assert set(vocabulary) == {"play", "jump", "funny", "little", "yellow", "ride", "water", "under"}
    assert {"always", "because", "example", "machine"}.isdisjoint(vocabulary)
    assert vocabulary["ride"] == "dolch" and vocabulary["under"] == "fry"
    by_list = {c["list_name"]: c for c in credits}
    assert by_list["dolch"]["license"] == "dolch-license" and by_list["fry"]["attribution"] == "fry list"
    assert set(by_list["dolch"]["levels"]) | set(by_list["fry"]["levels"]) <= LEARN_TO_READ_LEVELS


def test_target_words_are_a_hard_learn_to_read_filter():
    """PR #36 review question: with a vocabulary, only list words qualify. A book with one list word
    ("play") and many other content words yields only "play", so the request builder skips it."""
    vocabulary, _ = learn_to_read_vocabulary(word_lists_frame())
    text = "The dragon will play near the enormous mountain castle."
    assert target_words_of(text, "storyweaver/x", 2, vocabulary) == ("play",)
    assert set(target_words_of(text, "storyweaver/x", 2, None)) <= {"dragon", "play", "near", "enormous",
                                                                   "mountain", "castle", "will"}
    assert len(target_words_of(text, "storyweaver/x", 2, None)) == 2, "no vocabulary: any content word"


def test_an_empty_word_list_gives_no_vocabulary():
    assert learn_to_read_vocabulary(pd.DataFrame()) == ({}, [])


def _write(store: Store, rel: str, df: pd.DataFrame) -> dict:
    store.write_parquet(rel, df)
    return {"path": rel, "size": len(df) * 100, "generation": "1"}


def fake_bucket(tmp_path, word_lists: pd.DataFrame | None = None) -> Store:
    """A local store holding a v1.1 version: 120 picture books, two TinyStories files and the lists.

    The picture books use list words ("play", "jump", "under", ...) so requests can draw from the
    vocabulary. The *large* TinyStories file's texts all say "BIGFILE", so a request built from it
    is easy to spot.
    """
    store = Store(str(tmp_path / "bucket"))
    pages = conform(pd.DataFrame([
        {"record_id": f"sw{i}:{p}", "source": "storyweaver", "book_id": f"storyweaver-{i}", "title": f"B{i}",
         "page_number": p, "n_pages": 2, "text": text, "word_count": 5, "language": "en",
         "license": "CC-BY-4.0", "attribution": f"storyweaver {i}"}
        for i in range(120)
        for p, text in enumerate([f"Mia likes to play and jump {i}.", "Mia is funny under the yellow sun."], 1)
    ]), STORYBOOK_PAGES)

    def tiny(prefix: str, n: int, text: str) -> pd.DataFrame:
        return conform(pd.DataFrame([
            {"record_id": f"{prefix}{i}", "source": "tinystories", "source_id": f"{prefix}{i}", "title": None,
             "text": text.format(i=i), "word_count": 8, "sentence_count": 2, "text_hash": "", "language": "en",
             "license": "CDLA-Sharing-1.0", "attribution": "TinyStories"} for i in range(n)
        ]), STORIES_TEXT)

    small = _write(store, "processed/stories_text/source=tinystories/batch=small.parquet",
                   tiny("s", 60, "Tom can play with the little dog {i}. They jump."))
    big = _write(store, "processed/stories_text/source=tinystories/batch=big.parquet",
                 tiny("b", 200, "BIGFILE story {i} where Tom can play and jump."))
    big["size"] = 10**9
    record = {"tag": "v1.1", "datasets": {
        "storybook_pages": {"sources": {"storyweaver": {"files": [
            _write(store, "processed/storybook_pages/source=storyweaver/batch=1.parquet", pages)]}}},
        "stories_text": {"sources": {"tinystories": {"files": [big, small]}}},
        "word_lists": {"sources": {"sight_words": {"files": [
            _write(store, "processed/word_lists/source=sight_words/batch=1.parquet",
                   word_lists_frame() if word_lists is None else word_lists)]}}},
    }}
    store.write_json("metadata/versions/v1.1.json", record)
    return store


def test_main_builds_requests_from_the_pinned_subset_and_writes_the_manifest(tmp_path):
    """F3: the CLI end to end. Reads only the smallest TinyStories file, takes words from the Learn to
    Read lists, and records the lists' credits and each word's list in the manifest."""
    fake_bucket(tmp_path)
    out = tmp_path / "out" / "requests.jsonl"
    assert main(["--data-uri", str(tmp_path / "bucket"), "--version", "v1.1", "--n", "6",
                 "--out", str(out)]) == 0

    requests = read_requests(out)
    assert len(requests) == 6
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert [f["path"] for f in manifest["tinystories_files"]] == [
        "processed/stories_text/source=tinystories/batch=small.parquet"], "only the smallest file"
    assert manifest["tinystories_rows_read"] == 60
    assert manifest["vocabulary"]["levels"] == sorted(LEARN_TO_READ_LEVELS)
    assert {c["list_name"] for c in manifest["vocabulary"]["lists"]} == {"dolch", "fry"}
    assert sum(manifest["target_words_by_list"].values()) == sum(len(r.target_words) for r in requests)
    for request in requests:
        assert len(request.target_word_lists) == len(request.target_words)
        assert set(request.target_word_lists) <= {"dolch", "fry"}, "hard filter: never a book word"
        assert not {"always", "because", "example", "machine"} & set(request.target_words)
        assert "/b" not in request.book_key, "nothing from the large TinyStories file"


def test_main_refuses_to_build_requests_without_the_learn_to_read_words(tmp_path):
    """An empty vocabulary would silently turn the hard filter off; main must stop instead."""
    only_hard_words = word_lists_frame()
    only_hard_words = only_hard_words[~only_hard_words["level"].isin(LEARN_TO_READ_LEVELS)]
    fake_bucket(tmp_path, word_lists=only_hard_words)
    out = tmp_path / "out" / "requests.jsonl"
    with pytest.raises(SystemExit) as stopped:
        main(["--data-uri", str(tmp_path / "bucket"), "--version", "v1.1", "--n", "6", "--out", str(out)])
    assert stopped.value.code != 0
    assert not out.exists(), "no requests may be written without the filter"


def test_read_dataset_needs_an_explicit_store_and_reads_the_same_subset(tmp_path):
    """F4: no default store, so it can never fall back to PRIMER_DATA_URI or the bucket; and it reads
    the same smallest-file subset as main, not all of TinyStories."""
    with pytest.raises(TypeError):
        read_dataset("v1.1", "val")  # no store: the missing argument is the point
    store = fake_bucket(tmp_path)
    books = pd.concat([read_dataset("v1.1", split, store) for split in ("train", "val", "test")])
    tiny = books[books["source"] == "tinystories"]
    assert len(tiny) == 60 and not tiny["text"].str.contains("BIGFILE").any()
