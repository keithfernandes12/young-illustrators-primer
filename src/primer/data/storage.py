"""One interface for pipeline files, whether they live in GCS or a local folder.

Paths passed to ``Store`` are relative to the data root, for example
``raw/tinystories/20261014T030000Z/part-00000.jsonl.gz``. The root comes from
``PRIMER_DATA_URI`` and defaults to the project bucket.
"""

from __future__ import annotations

import gzip
import io
import json
import os
from typing import Any

import fsspec
import pandas as pd

DEFAULT_DATA_URI = "gs://young-illustrators-primer-data"


class Store:
    def __init__(self, uri: str | None = None):
        self.uri = (uri or os.environ.get("PRIMER_DATA_URI") or DEFAULT_DATA_URI).rstrip("/")
        # Don't pass project= to gcsfs: when the default credentials have no project
        # set, it silently falls back to anonymous access and every write fails.
        self.fs, self.root = fsspec.url_to_fs(self.uri)
        protocols = self.fs.protocol if isinstance(self.fs.protocol, tuple) else (self.fs.protocol,)
        self.is_local = "file" in protocols

    def __repr__(self) -> str:
        return f"Store({self.uri!r})"

    # -- paths

    def path(self, rel: str) -> str:
        return f"{self.root}/{rel.lstrip('/')}"

    def url(self, rel: str) -> str:
        """Full URI for a relative path, e.g. for manifests and logs."""
        return f"{self.uri}/{rel.lstrip('/')}"

    def _rel(self, full: str) -> str:
        full = full.replace("\\", "/")
        root = self.root.replace("\\", "/").rstrip("/")
        return full[len(root) + 1:] if full.startswith(root + "/") else full

    # -- listing

    def exists(self, rel: str) -> bool:
        self.fs.invalidate_cache()
        return self.fs.exists(self.path(rel))

    def glob(self, pattern: str) -> list[str]:
        self.fs.invalidate_cache()
        return sorted(self._rel(p) for p in self.fs.glob(self.path(pattern)))

    def list_dirs(self, rel: str) -> list[str]:
        """Names of the immediate subfolders of ``rel``."""
        self.fs.invalidate_cache()
        try:
            entries = self.fs.ls(self.path(rel), detail=True)
        except FileNotFoundError:
            return []
        return sorted(e["name"].rstrip("/").split("/")[-1] for e in entries if e["type"] == "directory")

    # -- bytes

    def write_bytes(self, rel: str, data: bytes) -> None:
        full = self.path(rel)
        if self.is_local:
            self.fs.makedirs(full.rsplit("/", 1)[0], exist_ok=True)
        self.fs.pipe_file(full, data)

    def read_bytes(self, rel: str) -> bytes:
        return self.fs.cat_file(self.path(rel))

    # -- json

    def write_json(self, rel: str, obj: Any) -> None:
        self.write_bytes(rel, json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8"))

    def read_json(self, rel: str, default: Any = None) -> Any:
        try:
            return json.loads(self.read_bytes(rel))
        except FileNotFoundError:
            return default

    # -- gzipped json lines (raw zone)

    def write_jsonl_gz(self, rel: str, records: list[dict]) -> None:
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        self.write_bytes(rel, gzip.compress(lines.encode("utf-8")))

    def read_jsonl_gz(self, rel: str) -> list[dict]:
        text = gzip.decompress(self.read_bytes(rel)).decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    # -- parquet (processed zone)

    def write_parquet(self, rel: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.write_bytes(rel, buf.getvalue())

    def read_parquet(self, rel: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.read_bytes(rel)), columns=columns)
