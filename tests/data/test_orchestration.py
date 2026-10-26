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

    report = orchestration.summarize("manual__2026-10-26T00:00:00+00:00")
    assert report["datasets"]["stories_text"] == {"rows": 4, "sources": {"tinystories": 4}}
    saved = Store().glob("metadata/reports/*.json")
    assert saved == ["metadata/reports/manual__2026-10-26T00_00_00_00_00.json"]


def test_freeze_version(fake_tinystories):
    batch = orchestration.ingest("tinystories", batch_size=4)
    orchestration.validate("tinystories", batch)
    orchestration.preprocess("tinystories", batch)
    assert orchestration.freeze_version("v9", "test") == {"tag": "v9", "datasets": {"stories_text": 4}}
