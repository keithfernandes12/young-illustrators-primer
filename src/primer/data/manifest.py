"""Run manifests: one JSON file per pipeline stage run, read by the tracker dashboard.

The format is documented in ``dashboards/tracker/loaders.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from primer.data.storage import Store

log = logging.getLogger(__name__)

STAGES = ("ingest", "validate", "preprocess", "split", "version")


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(ts: datetime | None) -> str | None:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None


def compact(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


class ValidationFailed(Exception):
    pass


@dataclass
class StageRun:
    store: Store
    source: str
    stage: str
    batch_id: str
    started_at: datetime = field(default_factory=utc_now)
    status: str = "running"
    finished_at: datetime | None = None
    records_in: int | None = None
    records_out: int | None = None
    records_total: int | None = None
    dataset_version: str | None = None
    output_uri: str | None = None
    checks: list[Check] = field(default_factory=list)
    error: str | None = None

    @property
    def run_id(self) -> str:
        return f"{compact(self.started_at)}-{self.source}-{self.stage}"

    def to_dict(self) -> dict:
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "source": self.source,
            "stage": self.stage,
            "batch_id": self.batch_id,
            "status": self.status,
            "started_at": iso(self.started_at),
            "finished_at": iso(self.finished_at),
            "records_in": self.records_in,
            "records_out": self.records_out,
            "records_total": self.records_total,
            "dataset_version": self.dataset_version,
            "output_uri": self.output_uri,
            "error": self.error,
        }
        if self.checks:
            record["validation"] = {"checks": [vars(c) for c in self.checks]}
        return record

    def save(self) -> None:
        self.store.write_json(f"metadata/runs/{self.run_id}.json", self.to_dict())

    def fail_on_checks(self) -> None:
        failed = [c.name for c in self.checks if not c.passed]
        if failed:
            raise ValidationFailed(f"Validation failed: {', '.join(failed)}")


@contextmanager
def record_stage(store: Store, source: str, stage: str, batch_id: str) -> Iterator[StageRun]:
    """Write a "running" manifest, then "success" or "failed" when the block ends."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    run = StageRun(store=store, source=source, stage=stage, batch_id=batch_id)
    run.save()
    log.info("%s %s started (batch %s)", source, stage, batch_id)
    try:
        yield run
    except Exception as exc:
        run.status = "failed"
        run.error = str(exc) if isinstance(exc, ValidationFailed) else f"{type(exc).__name__}: {exc}"
        run.finished_at = utc_now()
        run.save()
        log.error("%s %s failed: %s", source, stage, run.error)
        raise
    run.status = "success"
    run.finished_at = utc_now()
    run.save()
    log.info("%s %s finished: %s in, %s out", source, stage, run.records_in, run.records_out)
