"""The functions the Airflow DAGs call, run against fakes and a temporary local store."""

import pytest

from primer.data import orchestration
from primer.data.manifest import ValidationFailed
from primer.data.sources.tinystories import TinyStories
from primer.data.storage import Store

STORIES = [f"Story {i}: the little cat found a big red ball." for i in range(6)]


@pytest.fixture
def fake_tinystories(monkeypatch, tinystories_config):
    """Make orchestration build a TinyStories source with a fake fetcher on the default store."""
    rows = list(STORIES)

    def build(key, store=None):
        assert key == "tinystories"
        return TinyStories(tinystories_config, store or Store(), fetch=lambda off, n: [
            {"text": t} for t in rows[off:off + n]])

    monkeypatch.setattr(orchestration, "build_source", build)
    return rows


def test_pipeline_sources_all_have_pipelines():
    from primer.data.sources import SOURCES

    assert set(orchestration.PIPELINE_SOURCES) <= set(SOURCES)


def test_build_source_rejects_unknown_and_unimplemented():
    with pytest.raises(KeyError, match="Unknown source"):
        orchestration.build_source("nope")
    with pytest.raises(KeyError, match="no pipeline"):
        orchestration.build_source("myst")      # registered in sources.yaml, not implemented


def test_ingest_validate_preprocess_then_nothing_new(fake_tinystories):
    batch = orchestration.ingest("tinystories", batch_size=4)
    assert batch
    orchestration.validate("tinystories", batch)
    orchestration.preprocess("tinystories", batch)

    second = orchestration.ingest("tinystories", batch_size=4)
    assert second and second != batch
    assert orchestration.ingest("tinystories", batch_size=4) is None   # the DAG skips on None


def test_validate_raises_on_bad_batch(fake_tinystories):
    fake_tinystories[:] = ["", "", "", "x"]
    batch = orchestration.ingest("tinystories", batch_size=4)
    with pytest.raises(ValidationFailed):
        orchestration.validate("tinystories", batch)


def test_summarize_counts_rows_and_saves_report(fake_tinystories):
    batch = orchestration.ingest("tinystories", batch_size=4)
    orchestration.validate("tinystories", batch)
    orchestration.preprocess("tinystories", batch)

    outcomes = {"tinystories": {"status": "success", "stage": None},
                "quickdraw": {"status": "failed", "stage": "validate"}}
    report = orchestration.summarize("manual__2026-10-26T00:00:00+00:00", outcomes)
    assert report["datasets"]["stories_text"] == {"rows": 4, "sources": {"tinystories": 4}}
    assert report["failed"] == ["quickdraw"]
    assert report["sources"]["quickdraw"] == {"status": "failed", "stage": "validate"}
    saved = Store().glob("metadata/reports/*.json")
    assert saved == ["metadata/reports/manual__2026-10-26T00_00_00_00_00.json"]


def test_check_data_emails_drift_only_when_there_are_no_errors(store, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    from primer.data import alerts, profiling

    def frame(seed, shift=0.0, source="clear_corpus"):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"score": rng.normal(-1.0 + shift, 0.5, 200),
                             "source": pd.array([source] * 200, dtype="string")})

    def write(df, batch):   # clear_corpus is a snapshot source: the newest batch replaces the others
        store.write_parquet(f"processed/readability/source=clear_corpus/batch={batch}.parquet", df)

    schemas = tmp_path / "schemas"
    real_check = profiling.check_datasets
    monkeypatch.setattr(profiling, "check_datasets", lambda store, **kw: real_check(
        store, schemas_dir=schemas, datasets=("readability",), **kw))
    emails = []
    monkeypatch.setattr(alerts, "send_alert", lambda subject, body: emails.append((subject, body)))

    write(frame(0), "20261001T000000Z")
    profiling.write_schemas(store, schemas_dir=schemas, datasets=("readability",))
    clean = orchestration.check_data("run1", store=store)
    assert (clean["errors"], clean["warnings"], emails) == (0, 0, [])

    write(frame(1, shift=2.0), "20261002T000000Z")            # mean moves 4 sd: drift, contract holds
    drifted = orchestration.check_data("run2", store=store)
    assert drifted["errors"] == 0 and drifted["warnings"] > 0
    [(subject, body)] = emails
    assert subject == f"Data drift: {drifted['warnings']} warnings" and body == drifted["summary"]

    write(frame(2, source="mystery"), "20261003T000000Z")     # breaks the allowed set: an error
    broken = orchestration.check_data("run3", store=store)
    assert broken["errors"] > 0 and len(emails) == 1           # the failing task's alert carries it
    assert "ERROR readability [source]" in broken["summary"]


def test_freeze_version(fake_tinystories):
    batch = orchestration.ingest("tinystories", batch_size=4)
    orchestration.validate("tinystories", batch)
    orchestration.preprocess("tinystories", batch)
    assert orchestration.freeze_version("v9", "test") == {"tag": "v9", "datasets": {"stories_text": 4}}


def test_source_outcomes_from_task_states():
    states = {
        "tinystories.ingest": "success", "tinystories.validate": "success", "tinystories.preprocess": "success",
        "quickdraw.ingest": "success", "quickdraw.validate": "failed", "quickdraw.preprocess": "upstream_failed",
        "clear_corpus.ingest": "skipped", "clear_corpus.validate": "skipped", "clear_corpus.preprocess": "skipped",
        "sight_words.ingest": "failed", "sight_words.validate": "upstream_failed",
        "sight_words.preprocess": "upstream_failed",
        "kuperman_aoa.ingest": "running",
    }
    outcomes = orchestration.source_outcomes(states)
    assert outcomes["tinystories"] == {"status": "success", "stage": None}
    assert outcomes["quickdraw"] == {"status": "failed", "stage": "validate"}
    assert outcomes["clear_corpus"] == {"status": "skipped", "stage": None}
    assert outcomes["sight_words"] == {"status": "failed", "stage": "ingest"}
    assert outcomes["kuperman_aoa"] == {"status": "unknown", "stage": None}
    assert outcomes["storyweaver"] == {"status": "unknown", "stage": None}   # no states at all
