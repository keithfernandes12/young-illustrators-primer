"""Entry points the Airflow DAGs call, kept free of Airflow so they can be tested anywhere.

Each function builds the source from ``configs/sources.yaml`` and a ``Store`` on
``PRIMER_DATA_URI``, then runs one pipeline stage. The DAGs in ``dags/`` only
wire these together; all behaviour lives here and in ``pipeline.py``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from primer.data import alerts, pipeline
from primer.data.catalog import DATASETS, dataset_files
from primer.data.config import load_sources
from primer.data.sources import SOURCES, get_source
from primer.data.sources.base import Source
from primer.data.storage import Store
from primer.data.versioning import create_version

log = logging.getLogger(__name__)

# Order the DAG lays the source groups out in (all run in parallel).
PIPELINE_SOURCES = ("tinystories", "quickdraw", "clear_corpus", "sight_words", "kuperman_aoa",
                    "storyweaver", "african_storybook")
REPORTS_DIR = "metadata/reports"
STAGE_ORDER = ("ingest", "validate", "preprocess")
FAILED_STATES = {"failed", "upstream_failed"}


def build_source(key: str, store: Store | None = None) -> Source:
    configs = load_sources()
    if key not in configs:
        raise KeyError(f"Unknown source {key!r}; known: {', '.join(configs)}")
    if key not in SOURCES:
        raise KeyError(f"Source {key!r} has no pipeline yet")
    return get_source(configs[key], store or Store())


def ingest(key: str, batch_size: int | None = None) -> str | None:
    """Fetch the next batch. Returns its batch id, or None when there was nothing new."""
    source = build_source(key)
    log.info("Ingesting %s into %s (batch size %s)", key, source.store.uri, batch_size or "default")
    return pipeline.ingest(source, batch_size)


def validate(key: str, batch_id: str) -> None:
    """Check a raw batch. Raises ``ValidationFailed`` if any check fails."""
    pipeline.validate(build_source(key), batch_id)


def preprocess(key: str, batch_id: str) -> None:
    pipeline.preprocess(build_source(key), batch_id)


def source_outcomes(task_states: dict[str, str | None]) -> dict[str, dict[str, str | None]]:
    """Each source's result in a DAG run, from its tasks' states (task ids like "quickdraw.validate").

    failed  - a stage failed ("stage" names the first one that did)
    success - preprocess finished
    skipped - ingest was skipped (not selected, or nothing new)
    unknown - anything else, e.g. the run was cancelled mid-way
    """
    outcomes: dict[str, dict[str, str | None]] = {}
    for key in PIPELINE_SOURCES:
        states = {stage: task_states.get(f"{key}.{stage}") for stage in STAGE_ORDER}
        failed = next((s for s in STAGE_ORDER if states[s] == "failed"), None)
        if failed or any(state in FAILED_STATES for state in states.values()):
            outcomes[key] = {"status": "failed", "stage": failed}
        elif states["preprocess"] == "success":
            outcomes[key] = {"status": "success", "stage": None}
        elif states["ingest"] == "skipped":
            outcomes[key] = {"status": "skipped", "stage": None}
        else:
            outcomes[key] = {"status": "unknown", "stage": None}
    return outcomes


def summarize(run_id: str, outcomes: dict[str, dict[str, str | None]] | None = None,
              store: Store | None = None) -> dict[str, Any]:
    """Count rows per dataset and source after a run, and save them with each source's outcome.

    Written whether or not sources failed, so a partial run still records what landed.
    """
    store = store or Store()
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        sources = {key: sum(store.parquet_rows(p) for p in paths)
                   for key, paths in dataset_files(store, dataset).items()}
        if sources:
            datasets[dataset] = {"rows": sum(sources.values()), "sources": sources}
    outcomes = outcomes or {}
    report: dict[str, Any] = {
        "run_id": run_id, "created_at": datetime.now(UTC).isoformat(), "data_uri": store.uri,
        "sources": outcomes,
        "failed": sorted(k for k, v in outcomes.items() if v["status"] == "failed"),
        "datasets": datasets,
    }
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    store.write_json(f"{REPORTS_DIR}/{safe_id}.json", report)
    for name, entry in datasets.items():
        log.info("%-16s %10s rows  %s", name, f"{entry['rows']:,}",
                 ", ".join(f"{k}={v:,}" for k, v in entry["sources"].items()))
    for key, outcome in outcomes.items():
        log.info("%-18s %s%s", key, outcome["status"], f" at {outcome['stage']}" if outcome["stage"] else "")
    report["report_uri"] = store.url(f"{REPORTS_DIR}/{safe_id}.json")
    return report


def check_data(run_id: str, store: Store | None = None) -> dict[str, Any]:
    """Statistics, schema validation and drift for every dataset; saves profiles and the anomaly report.

    Drift-only results (warnings, no errors) are emailed from here. Schema errors are left to the
    caller to fail the task, so the failure alert carries them (``report["summary"]``).
    """
    from primer.data import profiling  # Great Expectations loads only for this task

    store = store or Store()
    log.info("Checking datasets in %s", store.uri)
    report = profiling.check_datasets(store, run_id=run_id, save=True)
    report["summary"] = profiling.format_anomalies(report)
    log.info("%s", report["summary"])
    if report["warnings"] and not report["errors"]:
        alerts.send_alert(f"Data drift: {report['warnings']} warnings", report["summary"])
    return report


def freeze_version(tag: str, message: str = "") -> dict[str, Any]:
    """Record a dataset version. DVC pointer files are refreshed separately, from a git clone."""
    version = create_version(Store(), tag, message, dvc=False)
    return {"tag": version["tag"], "datasets": {k: v["rows"] for k, v in version["datasets"].items()}}
