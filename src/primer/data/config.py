"""Load the dataset registry (configs/sources.yaml)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES_FILE = REPO_ROOT / "configs" / "sources.yaml"


@dataclass
class SourceConfig:
    key: str
    name: str
    license: str
    license_verified: bool
    attribution: str
    feeds: list[str] = field(default_factory=list)
    priority: int = 99
    mvp: bool = False
    ingest: dict = field(default_factory=dict)


def load_sources(path: Path | str | None = None) -> dict[str, SourceConfig]:
    path = Path(path or os.environ.get("PRIMER_SOURCES_FILE") or DEFAULT_SOURCES_FILE)
    with open(path, encoding="utf-8") as fh:
        raw = (yaml.safe_load(fh) or {}).get("sources", {})
    return {
        key: SourceConfig(
            key=key,
            name=entry.get("name", key),
            license=entry.get("license", "unknown"),
            license_verified=bool(entry.get("license_verified", False)),
            attribution=entry.get("attribution", ""),
            feeds=entry.get("feeds") or [],
            priority=entry.get("priority", 99),
            mvp=bool(entry.get("mvp", False)),
            ingest=entry.get("ingest") or {},
        )
        for key, entry in raw.items()
    }
