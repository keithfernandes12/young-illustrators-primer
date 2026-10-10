import pandas as pd
import pytest

from primer.data import pipeline
from primer.data.catalog import read_dataset, read_version
from primer.data.sources.tinystories import TinyStories
from primer.data.versioning import VersionExists, create_version, list_versions

ROWS = [f"Story number {i} is a short tale about a cat." for i in range(8)]


@pytest.fixture
def source(tinystories_config, store, monkeypatch):
    ids = iter(f"2026101{i}T000000Z" for i in range(10))
    monkeypatch.setattr(pipeline, "new_batch_id", lambda: next(ids))
    return TinyStories(tinystories_config, store, fetch=lambda off, n: [{"text": t} for t in ROWS[off:off + n]])


def test_version_freezes_current_files(source, store):
    pipeline.run_all(source, batch_size=4)
    version = create_version(store, "v1", "first cut", dvc=False)

    assert version["datasets"]["stories_text"]["rows"] == 4
    assert version["datasets"]["stories_text"]["sources"]["tinystories"]["files"][0]["path"].startswith(
        "processed/stories_text/source=tinystories/")
    assert read_version("v1", store)["message"] == "first cut"

    manifests = [store.read_json(p) for p in store.glob("metadata/runs/*-version.json")]
    assert [(m["source"], m["dataset_version"], m["records_total"]) for m in manifests] == [("tinystories", "v1", 4)]

    # New data after the freeze shows up in the current dataset but not in v1.
    pipeline.run_all(source, batch_size=4)
    assert len(read_dataset("stories_text", store)) == 8
    assert len(read_dataset("stories_text", store, version="v1")) == 4


def test_versions_are_immutable_and_listed(source, store):
    pipeline.run_all(source, batch_size=4)
    create_version(store, "v1", dvc=False)
    with pytest.raises(VersionExists):
        create_version(store, "v1", dvc=False)
    assert [v["tag"] for v in list_versions(store)] == ["v1"]
    with pytest.raises(KeyError):
        read_dataset("stories_text", store, version="nope")


def test_dashboard_shows_version(source, store, dashboard_loaders):
    pipeline.run_all(source, batch_size=4)
    create_version(store, "v1", dvc=False)
    loaders = dashboard_loaders
    versions = loaders.load_versions(store.url("metadata"))
    assert [v["tag"] for v in versions] == ["v1"]

    runs = loaders.load_runs(store.url("metadata"))
    sources = pd.DataFrame({"name": ["TinyStories"]}, index=pd.Index(["tinystories"], name="key"))
    feed = loaders.activity({}, runs, sources, versions)
    assert sum("Dataset version v1" in i["text"] for i in feed) == 1
    assert not any("version done" in i["text"] for i in feed)
    assert loaders.source_status(sources, runs).iloc[0]["Status"] == "✅ processed, in v1"
