import pytest

from primer.data import pipeline
from primer.data.manifest import ValidationFailed
from primer.data.sources.tinystories import PROCESSED_DIR, TinyStories
from primer.data.text import normalize_text, word_count

STORIES = [
    "Once upon a time there was a cat.",
    "The   dog ran to the park and played.",
    "",                                          # empty: dropped
    "once upon a time there was a CAT.",         # duplicate of the first, ignoring case
    "Lily found a shiny red ball in the garden.",
    "Tom and Sam built a big sandcastle today.",
    "Once upon a time there was a cat.",         # duplicate of an earlier batch
]


def fake_hub(rows):
    def fetch(offset, limit):
        return [{"text": t} for t in rows[offset:offset + limit]]
    return fetch


def make_source(config, store, rows=STORIES):
    return TinyStories(config, store, fetch=fake_hub(rows))


def test_text_helpers():
    assert normalize_text("  Hi   there.\r\n\r\n\r\nBye  ") == "Hi there.\n\nBye"
    assert word_count("The cat's hat, isn't it?") == 5


def test_ingest_writes_raw_and_advances_offset(tinystories_config, store):
    source = make_source(tinystories_config, store)
    batch = pipeline.ingest(source, batch_size=4)
    assert batch is not None
    assert len(source.read_raw(batch)) == 4
    assert store.read_json(source.state_path) == {"next_offset": 4}
    assert store.read_json(f"raw/tinystories/{batch}/_batch.json")["offset_end"] == 4


def test_two_batches_continue_and_dedupe(tinystories_config, store, monkeypatch):
    source = make_source(tinystories_config, store)
    ids = iter(["20261014T030000Z", "20261015T030000Z", "20261016T030000Z"])
    monkeypatch.setattr(pipeline, "new_batch_id", lambda: next(ids))

    first = pipeline.ingest(source, batch_size=4)
    pipeline.preprocess(source, first)
    df1 = store.read_parquet(f"{PROCESSED_DIR}/batch={first}.parquet")
    # empty and case-duplicate dropped
    assert df1["text"].tolist() == [
        "Once upon a time there was a cat.",
        "The dog ran to the park and played.",
    ]
    assert set(df1["license"]) == {"CDLA-Sharing-1.0"}

    second = pipeline.ingest(source, batch_size=4)
    pipeline.preprocess(source, second)
    df2 = store.read_parquet(f"{PROCESSED_DIR}/batch={second}.parquet")
    # rows 4-6; the repeat of the first story is dropped against batch one
    assert df2["source_id"].tolist() == ["4", "5"]

    assert pipeline.ingest(source, batch_size=4) is None  # source exhausted


def test_validate_fails_on_bad_batch(tinystories_config, store):
    source = make_source(tinystories_config, store, rows=["", "", "", "tiny"])
    batch = pipeline.ingest(source, batch_size=4)
    with pytest.raises(ValidationFailed):
        pipeline.validate(source, batch)
    manifests = [store.read_json(p) for p in store.glob("metadata/runs/*-validate.json")]
    assert manifests[0]["status"] == "failed"
    assert {c["name"] for c in manifests[0]["validation"]["checks"] if not c["passed"]} == {
        "empty_text", "word_count_3_to_50"
    }


def test_validate_passes_on_good_batch(tinystories_config, store):
    rows = ["The cat sat on the mat.", "A dog ran home fast.", "Sun is up today now.", "We like to read."]
    source = make_source(tinystories_config, store, rows=rows)
    batch = pipeline.ingest(source, batch_size=4)
    pipeline.validate(source, batch)
    [manifest] = [store.read_json(p) for p in store.glob("metadata/runs/*-validate.json")]
    assert manifest["status"] == "success"
    assert all(c["passed"] for c in manifest["validation"]["checks"])


def test_ingests_in_the_same_second_get_separate_batches(tinystories_config, store, monkeypatch):
    """Regression: same-second ingests shared a batch id, so the second overwrote the first's raw files."""
    monkeypatch.setattr(pipeline, "new_batch_id", lambda: "20261026T000000Z")
    source = make_source(tinystories_config, store)
    first = pipeline.ingest(source, batch_size=2)
    second = pipeline.ingest(source, batch_size=2)
    assert (first, second) == ("20261026T000000Z", "20261026T000000Z-2")
    assert [r["source_id"] for r in source.read_raw(first)] == ["0", "1"]   # untouched
    assert [r["source_id"] for r in source.read_raw(second)] == ["2", "3"]
    assert pipeline.latest_batch(source) == second
