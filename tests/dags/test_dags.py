"""The Airflow DAGs load and are wired as documented.

Needs Airflow installed, which runs on Linux/macOS only (or inside the docker-compose
containers: see dags/README.md -> Tests).
Skipped elsewhere.
"""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

try:  # Airflow 3.1+
    from airflow.dag_processing.dagbag import DagBag
except ImportError:  # pragma: no cover
    from airflow.models.dagbag import DagBag

from primer.data.orchestration import PIPELINE_SOURCES

# The repo's dags/ folder, or Airflow's when the tests run inside the docker-compose containers.
DAGS_DIR = next(p for p in (Path(__file__).resolve().parents[2] / "dags", Path("/opt/airflow/dags")) if p.is_dir())


@pytest.fixture(scope="module")
def dagbag():
    # Airflow 3.3 dropped `include_examples`; examples come from config (off in docker-compose.yaml).
    return DagBag(dag_folder=str(DAGS_DIR))


def test_dags_import_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert {"primer_data_pipeline", "primer_freeze_version"} <= set(dagbag.dag_ids)


def test_each_source_runs_ingest_validate_preprocess(dagbag):
    dag = dagbag.get_dag("primer_data_pipeline")
    summarize = dag.get_task("summarize")
    for key in PIPELINE_SOURCES:
        ingest, validate, preprocess = (dag.get_task(f"{key}.{t}") for t in ("ingest", "validate", "preprocess"))
        assert ingest.downstream_task_ids == {validate.task_id}
        assert validate.downstream_task_ids == {preprocess.task_id}
        assert summarize.task_id in preprocess.downstream_task_ids
        assert ingest.retries == 3 and validate.retries == 0


def test_summarize_always_runs_after_every_source(dagbag):
    dag = dagbag.get_dag("primer_data_pipeline")
    summarize = dag.get_task("summarize")
    assert summarize.upstream_task_ids == {f"{key}.preprocess" for key in PIPELINE_SOURCES}
    assert summarize.trigger_rule == "all_done"   # writes the report even when a source failed


def test_check_data_runs_after_every_source(dagbag):
    dag = dagbag.get_dag("primer_data_pipeline")
    check = dag.get_task("check_data")
    assert check.upstream_task_ids == {f"{key}.preprocess" for key in PIPELINE_SOURCES}
    assert check.trigger_rule == "all_done"     # profiles what exists even when a source failed
    assert check.downstream_task_ids == set()   # a leaf: its failure fails the run


def test_failures_email_except_the_summary(dagbag):
    from primer.data import alerts

    for dag_id in ("primer_data_pipeline", "primer_freeze_version"):
        dag = dagbag.get_dag(dag_id)
        for t in dag.tasks:
            callbacks = t.on_failure_callback if isinstance(t.on_failure_callback, list) else [t.on_failure_callback]
            expected = "summarize_failure" if t.task_id == "summarize" else alerts.notify_task_failure.__name__
            assert [c.__name__ for c in callbacks] == [expected], (dag_id, t.task_id)


def test_summarize_emails_its_own_failures_but_not_a_source_failure(dagbag, monkeypatch):
    from types import SimpleNamespace

    from primer.data import alerts

    task = dagbag.get_dag("primer_data_pipeline").get_task("summarize")
    [callback] = task.on_failure_callback if isinstance(task.on_failure_callback, list) else [task.on_failure_callback]
    emailed = []
    monkeypatch.setattr(alerts, "notify_task_failure", emailed.append)
    ti = SimpleNamespace(dag_id="primer_data_pipeline", task_id="summarize", try_number=1)

    callback({"ti": ti, "run_id": "r", "exception": callback.__globals__["SourcesFailed"]("Sources failed: x")})
    assert emailed == []                       # the failed source already sent its alert
    callback({"ti": ti, "run_id": "r", "exception": OSError("store write failed")})
    assert len(emailed) == 1                   # summarize broke by itself: nobody else will say so


def test_pipeline_params_and_concurrency(dagbag):
    dag = dagbag.get_dag("primer_data_pipeline")
    assert dag.max_active_runs == 1
    assert list(dag.params["sources"]) == list(PIPELINE_SOURCES)
    assert dag.params["tinystories_batch_size"] == 2000


def test_freeze_version_requires_a_tag(dagbag):
    dag = dagbag.get_dag("primer_freeze_version")
    assert dag.task_ids == ["freeze"]
    assert "tag" in dag.params
