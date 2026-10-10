"""Base class for data sources.

Each source implements three stages. The pipeline wraps every call in a run
manifest, so a stage only does its work and fills in counts on ``run``.

- ``ingest``: copy the next batch, unchanged, into ``raw/<source>/<batch_id>/``.
- ``validate``: check the raw batch and append ``Check`` results to ``run.checks``.
- ``preprocess``: clean the raw batch and write it to ``processed/`` in a standard schema.

Sources come in two kinds:

- **Incremental** (``snapshot = False``): each batch adds new records, and the
  dataset is all batches together. TinyStories and Quick, Draw! work this way.
- **Snapshot** (``snapshot = True``): the source is a fixed file. Each batch is
  a full copy, a new batch is made only when the file changes, and readers use
  the latest batch. CLEAR and the word lists work this way.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime

from primer.data.config import SourceConfig
from primer.data.manifest import StageRun
from primer.data.storage import Store

log = logging.getLogger(__name__)

USER_AGENT = "young-illustrators-primer-data-pipeline (+https://github.com/keithfernandes12/young-illustrators-primer)"

Download = Callable[[str], bytes]


def http_get(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class Source(ABC):
    key: str
    dataset: str  # processed/<dataset>/
    snapshot: bool = False

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

    def processed_path(self, batch_id: str, dataset: str | None = None) -> str:
        return f"processed/{dataset or self.dataset}/source={self.key}/batch={batch_id}.parquet"

    def processed_paths(self, dataset: str | None = None) -> list[str]:
        return self.store.glob(f"processed/{dataset or self.dataset}/source={self.key}/batch=*.parquet")

    def read_raw(self, batch_id: str) -> list[dict]:
        """All records from a batch's ``part-*.jsonl.gz`` files (incremental sources)."""
        parts = self.store.glob(f"{self.raw_prefix(batch_id)}/part-*.jsonl.gz")
        if not parts:
            raise FileNotFoundError(f"No raw files for {self.key} batch {batch_id}")
        return [record for part in parts for record in self.store.read_jsonl_gz(part)]

    def read_raw_file(self, batch_id: str, name: str) -> bytes:
        return self.store.read_bytes(f"{self.raw_prefix(batch_id)}/{name}")

    def save_snapshot(self, run: StageRun, files: dict[str, bytes], **details) -> bool:
        """Store downloaded files as a new raw batch unless they match the last snapshot.

        Returns False (and writes nothing) when the content hasn't changed.
        """
        digest = hashlib.sha256()
        for name in sorted(files):
            digest.update(name.encode())
            digest.update(files[name])
        sha = digest.hexdigest()

        state = self.store.read_json(self.state_path, default={})
        run.records_in = 0
        if state.get("sha256") == sha:
            log.info("%s is unchanged since batch %s; nothing new", self.key, state.get("latest_batch"))
            run.records_out = 0
            return False

        prefix = self.raw_prefix(run.batch_id)
        for name, data in files.items():
            self.store.write_bytes(f"{prefix}/{name}", data)
        self.store.write_json(f"{prefix}/_batch.json", {
            "batch_id": run.batch_id,
            "files": {n: {"bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()} for n, d in files.items()},
            "fetched_at": datetime.now(UTC).isoformat(),
            **details,
        })
        self.store.write_json(self.state_path, {"sha256": sha, "latest_batch": run.batch_id})
        run.output_uri = self.store.url(prefix)
        return True

    @abstractmethod
    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        """Fetch the next batch. Return False when there was nothing new to fetch."""

    @abstractmethod
    def validate(self, run: StageRun) -> None: ...

    @abstractmethod
    def preprocess(self, run: StageRun) -> None: ...
