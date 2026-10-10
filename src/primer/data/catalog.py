"""Read processed datasets for analysis and model training.

    from primer.data.catalog import read_dataset
    stories = read_dataset("stories_text")                     # current data
    drawings = read_dataset("drawings", version="v1")          # exactly as frozen in v1
    words = read_dataset("word_lists", sources=["sight_words"])

Incremental sources contribute every batch; snapshot sources contribute only
their latest batch, which is always the complete dataset. A named version
(see ``primer-data version``) reads the exact file versions it recorded.
"""

from __future__ import annotations

import pandas as pd

from primer.data.sources import SOURCES
from primer.data.storage import Store

DATASETS = ("stories_text", "storybook_pages", "drawings", "readability", "word_lists")
VERSIONS_DIR = "metadata/versions"


def dataset_files(store: Store, dataset: str, sources: list[str] | None = None) -> dict[str, list[str]]:
    """The processed files that currently make up a dataset, by source."""
    files = {}
    for folder in store.list_dirs(f"processed/{dataset}"):
        key = folder.removeprefix("source=")
        if sources and key not in sources:
            continue
        paths = store.glob(f"processed/{dataset}/{folder}/batch=*.parquet")
        source_cls = SOURCES.get(key)
        if source_cls is not None and source_cls.snapshot:
            paths = paths[-1:]
        if paths:
            files[key] = paths
    return files


def read_version(tag: str, store: Store | None = None) -> dict:
    store = store or Store()
    version = store.read_json(f"{VERSIONS_DIR}/{tag}.json")
    if version is None:
        raise KeyError(f"No dataset version {tag!r}; see `primer-data versions`")
    return version


def read_dataset(dataset: str, store: Store | None = None, sources: list[str] | None = None,
                 version: str | None = None) -> pd.DataFrame:
    store = store or Store()
    if version:
        recorded = read_version(version, store)["datasets"].get(dataset, {}).get("sources", {})
        frames = [store.read_parquet(f["path"], generation=f.get("generation"))
                  for key, entry in recorded.items() if not sources or key in sources
                  for f in entry["files"]]
    else:
        frames = [store.read_parquet(path)
                  for paths in dataset_files(store, dataset, sources).values() for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
