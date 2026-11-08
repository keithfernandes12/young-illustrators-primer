"""
### Young Illustrator's Primer: freeze a dataset version

Records every file that currently makes up each processed dataset, with its
exact object generation, in `metadata/versions/<tag>.json`. Versions are
permanent: a tag that already exists fails the run.

Trigger it by hand with a `tag` (e.g. `v1`) and a short `message`. Airflow has
no git clone, so afterwards a person refreshes the DVC pointer files and tags
the commit:

    primer-data dvc-pointers
    git add data/*.dvc && git commit -m "chore(data): data version v1 (KAN-<n>)"
    git tag data-v1
"""

from __future__ import annotations

from datetime import datetime

from airflow.sdk import Param, dag, get_current_context, task

from primer.data import alerts, orchestration


@dag(
    dag_id="primer_freeze_version",
    description="Freeze the current processed datasets as a named version",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["primer", "data", "versioning"],
    doc_md=__doc__,
    default_args={"owner": "data-engineering", "retries": 0, "on_failure_callback": alerts.notify_task_failure},
    params={
        "tag": Param(type="string", pattern=r"^v[0-9]+(\.[0-9]+)*$", description="Version name, e.g. v1"),
        "message": Param("", type="string", description="What this version contains"),
    },
)
def primer_freeze_version():
    @task
    def freeze() -> dict:
        params = get_current_context()["params"]
        return orchestration.freeze_version(params["tag"], params["message"])

    freeze()


primer_freeze_version()
