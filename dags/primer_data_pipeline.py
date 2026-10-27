"""
### Young Illustrator's Primer: data pipeline

One task group per data source, all running in parallel. Inside each group:

`ingest` → `validate` → `preprocess`

- **ingest** fetches the next batch into `raw/`. If the source has nothing new
  (an unchanged snapshot, or an incremental source that is fully ingested), the
  task is *skipped* and so are the rest of that source's tasks.
- **validate** runs the source's checks; a failed check fails the task, which
  stops that source before anything reaches `processed/`.
- **preprocess** cleans the batch into `processed/` in the dataset's schema.

`summarize` runs once every source has finished, whatever happened. It records
each source's outcome and row counts per dataset in `metadata/reports/`, then
**fails the run** if any source failed, naming which ones; a partial run still
leaves a report of what landed.

Data goes to `PRIMER_DATA_URI` (a local folder by default; the bucket when the
GCS compose override is used). See `dags/README.md`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.sdk import Param, dag, get_current_context, task, task_group

try:  # Airflow 3 task SDK
    from airflow.sdk.exceptions import AirflowFailException, AirflowSkipException
except ImportError:  # pragma: no cover - older Airflow
    from airflow.exceptions import AirflowFailException, AirflowSkipException

from primer.data import orchestration

log = logging.getLogger(__name__)

DEFAULT_SOURCES = list(orchestration.PIPELINE_SOURCES)


def log_failure(context) -> None:
    """Log enough to diagnose a failed task. Email/Slack alerts are added on top of this."""
    ti = context["ti"]
    log.error("Task failed: dag=%s task=%s run=%s try=%s error=%r", ti.dag_id, ti.task_id,
              context.get("run_id"), ti.try_number, context.get("exception"))


@dag(
    dag_id="primer_data_pipeline",
    description="Ingest, validate, and preprocess every data source into the data root",
    schedule=None,  # triggered by hand for now; set e.g. "@weekly" once runs go to the bucket
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,  # incremental sources keep a position; two runs at once would race on it
    tags=["primer", "data"],
    doc_md=__doc__,
    default_args={
        "owner": "data-engineering",
        "retries": 0,
        "on_failure_callback": log_failure,
    },
    params={
        "sources": Param(DEFAULT_SOURCES, type="array", items={"type": "string", "enum": DEFAULT_SOURCES},
                         description="Sources to run; the others are skipped."),
        "tinystories_batch_size": Param(2000, type="integer", minimum=1,
                                        description="Stories fetched per run."),
        "quickdraw_categories": Param(5, type="integer", minimum=1, maximum=345,
                                      description="Quick, Draw! categories fetched per run (of 345)."),
    },
)
def primer_data_pipeline():
    groups = []
    for key in orchestration.PIPELINE_SOURCES:

        @task_group(group_id=key, tooltip=f"ingest → validate → preprocess for {key}")
        def source_group(key: str = key):

            # Downloads are the only step that talks to the outside world, so only they retry.
            @task(retries=3, retry_delay=timedelta(minutes=1), retry_exponential_backoff=True,
                  execution_timeout=timedelta(hours=2))
            def ingest() -> str:
                params = get_current_context()["params"]
                if key not in params["sources"]:
                    raise AirflowSkipException(f"{key} not selected for this run")
                batch_size = {"tinystories": params["tinystories_batch_size"],
                              "quickdraw": params["quickdraw_categories"]}.get(key)
                batch_id = orchestration.ingest(key, batch_size)
                if batch_id is None:
                    raise AirflowSkipException(f"{key}: nothing new to ingest")
                return batch_id

            @task(execution_timeout=timedelta(minutes=30))
            def validate(batch_id: str) -> str:
                orchestration.validate(key, batch_id)
                return batch_id

            @task(retries=1, retry_delay=timedelta(minutes=1), execution_timeout=timedelta(hours=1))
            def preprocess(batch_id: str) -> None:
                orchestration.preprocess(key, batch_id)

            preprocess(validate(ingest()))

        groups.append(source_group())

    # Always runs, so every run leaves a report; then fails the run if any source failed.
    @task(trigger_rule="all_done")
    def summarize() -> dict:
        context = get_current_context()
        ti, run_id = context["ti"], context["run_id"]
        states = ti.get_task_states(dag_id=ti.dag_id, run_ids=[run_id]).get(run_id, {})
        report = orchestration.summarize(run_id, orchestration.source_outcomes(states))
        if report["failed"]:
            failed = ", ".join(f"{k} ({report['sources'][k]['stage'] or 'upstream'})" for k in report["failed"])
            raise AirflowFailException(f"Sources failed: {failed}. Report: {report['report_uri']}")
        return report

    groups >> summarize()


primer_data_pipeline()
