"""Freeze the processed datasets as a named, reproducible version.

``create_version(store, "v1")``:

1. Records every file that makes up each dataset right now, with its exact GCS
   object generation, in ``metadata/versions/v1.json``. The bucket has object
   versioning on, so those generations stay readable even if a file is later
   replaced: ``read_dataset("drawings", version="v1")`` always returns the same data.
2. Writes a "version" run manifest per source, which the tracker dashboard shows.
3. Refreshes the DVC pointer files ``data/<dataset>.dvc`` with
   ``dvc import-url --version-aware --no-download``. They record the same object
   versions without downloading or copying any data; commit them and tag the commit.

Versions are immutable: an existing tag can't be overwritten.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from primer.data.catalog import DATASETS, VERSIONS_DIR, dataset_files
from primer.data.config import REPO_ROOT
from primer.data.manifest import record_stage
from primer.data.sources import SOURCES
from primer.data.storage import Store

log = logging.getLogger(__name__)


class VersionExists(Exception):
    pass


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def update_dvc_pointers(store: Store, datasets: list[str], repo: Path = REPO_ROOT) -> list[str]:
    """Point data/<dataset>.dvc at the current object versions in GCS (no download)."""
    beside_python = Path(sys.executable).parent / ("dvc.exe" if sys.platform == "win32" else "dvc")
    dvc = str(beside_python) if beside_python.exists() else shutil.which("dvc")
    if dvc is None:
        raise RuntimeError("dvc is not installed; pip install 'dvc[gs]' or pass --no-dvc")
    (repo / "data").mkdir(exist_ok=True)
    written = []
    for dataset in datasets:
        subprocess.run([dvc, "import-url", "--version-aware", "--no-download", "--force",
                        store.url(f"processed/{dataset}"), f"data/{dataset}"],
                       cwd=repo, check=True, capture_output=True, text=True)
        written.append(f"data/{dataset}.dvc")
        log.info("DVC pointer updated: data/%s.dvc", dataset)
    return written


def create_version(store: Store, tag: str, message: str = "", dvc: bool = True,
                   repo: Path = REPO_ROOT) -> dict:
    path = f"{VERSIONS_DIR}/{tag}.json"
    if store.exists(path):
        raise VersionExists(f"Dataset version {tag!r} already exists; versions can't be overwritten")

    versions = {f["path"]: f for f in store.file_versions("processed")}
    datasets: dict[str, dict] = {}
    for dataset in DATASETS:
        sources = {}
        for key, paths in dataset_files(store, dataset).items():
            sources[key] = {
                "rows": sum(store.parquet_rows(p) for p in paths),
                "files": [versions.get(p, {"path": p}) for p in paths],
            }
        if sources:
            datasets[dataset] = {"rows": sum(s["rows"] for s in sources.values()), "sources": sources}

    version = {
        "tag": tag,
        "created_at": datetime.now(UTC).isoformat(),
        "message": message,
        "data_uri": store.uri,
        "git_commit": _git_commit(repo),
        "datasets": datasets,
    }
    store.write_json(path, version)

    # One "version" manifest per source, counted in the source's main dataset.
    for key, source_cls in SOURCES.items():
        entry = datasets.get(source_cls.dataset, {}).get("sources", {}).get(key)
        if entry is None:
            continue
        with record_stage(store, key, "version", tag) as run:
            run.records_in = run.records_out = run.records_total = entry["rows"]
            run.dataset_version = tag
            run.output_uri = store.url(path)

    version["dvc_files"] = update_dvc_pointers(store, list(datasets), repo) if dvc and not store.is_local else []
    return version


def list_versions(store: Store) -> list[dict]:
    return [store.read_json(p) for p in store.glob(f"{VERSIONS_DIR}/*.json")]
