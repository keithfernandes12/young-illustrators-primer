"""Entry points the Airflow DAGs call, kept free of Airflow so they can be tested anywhere.

Each function builds the source from ``configs/sources.yaml`` and a ``Store`` on
``PRIMER_DATA_URI``, then runs one pipeline stage. The DAGs in ``dags/`` only
wire these together; all behaviour lives here and in ``pipeline.py``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from primer.data import pipeline
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


def summarize(run_id: str, store: Store | None = None) -> dict[str, Any]:
    """Count rows per dataset and source after a run, and save the counts as a report."""
    store = store or Store()
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        sources = {key: sum(store.parquet_rows(p) for p in paths)
                   for key, paths in dataset_files(store, dataset).items()}
        if sources:
            datasets[dataset] = {"rows": sum(sources.values()), "sources": sources}
    report = {"run_id": run_id, "created_at": datetime.now(UTC).isoformat(), "data_uri": store.uri,
              "datasets": datasets}
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    store.write_json(f"{REPORTS_DIR}/{safe_id}.json", report)
    for name, entry in datasets.items():
        log.info("%-16s %10s rows  %s", name, f"{entry['rows']:,}",
                 ", ".join(f"{k}={v:,}" for k, v in entry["sources"].items()))
    return report


def freeze_version(tag: str, message: str = "") -> dict[str, Any]:
    """Record a dataset version. DVC pointer files are refreshed separately, from a git clone."""
    version = create_version(Store(), tag, message, dvc=False)
    return {"tag": version["tag"], "datasets": {k: v["rows"] for k, v in version["datasets"].items()}}
