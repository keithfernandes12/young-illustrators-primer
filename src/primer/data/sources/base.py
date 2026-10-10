"""Base class for data sources.

Each source implements three stages. The pipeline wraps every call in a run
manifest, so a stage only does its work and fills in counts on ``run``.

- ``ingest``: copy the next batch, unchanged, into ``raw/<source>/<batch_id>/``.
- ``validate``: check the raw batch and append ``Check`` results to ``run.checks``.
- ``preprocess``: clean the raw batch and write it to ``processed/`` in a standard schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from primer.data.config import SourceConfig
from primer.data.manifest import StageRun
from primer.data.storage import Store


class Source(ABC):
    key: str

    def __init__(self, config: SourceConfig, store: Store):
        self.config = config
        self.store = store

    @property
    def options(self) -> dict:
        return self.config.ingest

    @property
    def state_path(self) -> str:
        return f"metadata/state/{self.key}.json"

    def raw_prefix(self, batch_id: str) -> str:
        return f"raw/{self.key}/{batch_id}"

    def raw_batches(self) -> list[str]:
        return self.store.list_dirs(f"raw/{self.key}")

    def read_raw(self, batch_id: str) -> list[dict]:
        parts = self.store.glob(f"{self.raw_prefix(batch_id)}/part-*.jsonl.gz")
        if not parts:
            raise FileNotFoundError(f"No raw files for {self.key} batch {batch_id}")
        return [record for part in parts for record in self.store.read_jsonl_gz(part)]

    @abstractmethod
    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        """Fetch the next batch. Return False when there was nothing new to fetch."""

    @abstractmethod
    def validate(self, run: StageRun) -> None: ...

    @abstractmethod
    def preprocess(self, run: StageRun) -> None: ...
