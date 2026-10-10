"""Run source stages with run manifests. The CLI and Airflow both call these."""

from __future__ import annotations

import logging

from primer.data.manifest import compact, record_stage, utc_now
from primer.data.sources.base import Source

log = logging.getLogger(__name__)


def new_batch_id() -> str:
    return compact(utc_now())


def latest_batch(source: Source) -> str:
    batches = source.raw_batches()
    if not batches:
        raise FileNotFoundError(f"No raw batches for {source.key}; run ingest first")
    return batches[-1]


def ingest(source: Source, batch_size: int | None = None) -> str | None:
    """Ingest a new batch. Returns its batch id, or None if there was nothing new."""
    batch_id = new_batch_id()
    with record_stage(source.store, source.key, "ingest", batch_id) as run:
        fetched = source.ingest(run, batch_size)
    return batch_id if fetched else None


def validate(source: Source, batch_id: str | None = None) -> None:
    batch_id = batch_id or latest_batch(source)
    with record_stage(source.store, source.key, "validate", batch_id) as run:
        source.validate(run)
        run.fail_on_checks()


def preprocess(source: Source, batch_id: str | None = None) -> None:
    batch_id = batch_id or latest_batch(source)
    with record_stage(source.store, source.key, "preprocess", batch_id) as run:
        source.preprocess(run)


def run_all(source: Source, batch_size: int | None = None) -> str | None:
    """Ingest one new batch, validate it, then preprocess it."""
    batch_id = ingest(source, batch_size)
    if batch_id is None:
        return None
    validate(source, batch_id)
    preprocess(source, batch_id)
    return batch_id
