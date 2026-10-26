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


def test_summarize_waits_for_every_source_and_fails_on_failure(dagbag):
    dag = dagbag.get_dag("primer_data_pipeline")
    summarize = dag.get_task("summarize")
    assert summarize.upstream_task_ids == {f"{key}.preprocess" for key in PIPELINE_SOURCES}
    assert summarize.trigger_rule == "none_failed_min_one_success"


def test_pipeline_params_and_concurrency(dagbag):
    dag = dagbag.get_dag("primer_data_pipeline")
    assert dag.max_active_runs == 1
    assert list(dag.params["sources"]) == list(PIPELINE_SOURCES)
    assert dag.params["tinystories_batch_size"] == 2000


def test_freeze_version_requires_a_tag(dagbag):
    dag = dagbag.get_dag("primer_freeze_version")
    assert dag.task_ids == ["freeze"]
    assert "tag" in dag.params
