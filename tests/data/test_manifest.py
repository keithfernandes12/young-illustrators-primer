import pytest

from primer.data.manifest import Check, ValidationFailed, record_stage


def test_success_manifest(store):
    with record_stage(store, "tinystories", "ingest", "B1") as run:
        run.records_out = 10
    [path] = store.glob("metadata/runs/*.json")
    manifest = store.read_json(path)
    assert manifest["status"] == "success"
    assert manifest["records_out"] == 10
    assert manifest["batch_id"] == "B1"
    assert manifest["finished_at"]


def test_exception_marks_run_failed_and_reraises(store):
    with pytest.raises(RuntimeError), record_stage(store, "tinystories", "ingest", "B1"):
        raise RuntimeError("network down")
    manifest = store.read_json(store.glob("metadata/runs/*.json")[0])
    assert manifest["status"] == "failed"
    assert manifest["error"] == "RuntimeError: network down"


def test_failed_checks_fail_the_run(store):
    with pytest.raises(ValidationFailed), record_stage(store, "tinystories", "validate", "B1") as run:
        run.checks = [Check("a", True), Check("b", False, "bad")]
        run.fail_on_checks()
    manifest = store.read_json(store.glob("metadata/runs/*.json")[0])
    assert manifest["status"] == "failed"
    assert manifest["error"] == "Validation failed: b"
    assert manifest["validation"]["checks"][1] == {"name": "b", "passed": False, "detail": "bad"}


def test_dashboard_reads_pipeline_manifests(store, dashboard_loaders):
    """The manifests the pipeline writes must match what the dashboard expects."""
    with record_stage(store, "tinystories", "ingest", "B1") as run:
        run.records_out = run.records_total = 7
    runs = dashboard_loaders.load_runs(store.url("metadata"))
    assert len(runs) == 1
    row = runs.iloc[0]
    assert (row["source"], row["stage"], row["status"], row["records_total"]) == (
        "tinystories", "ingest", "success", 7
    )
