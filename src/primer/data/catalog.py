"""Read processed datasets for analysis and model training.

    from primer.data.catalog import read_dataset
    stories = read_dataset("stories_text")
    words = read_dataset("word_lists", sources=["sight_words"])

Incremental sources contribute every batch; snapshot sources contribute only
their latest batch, which is always the complete dataset.
"""

from __future__ import annotations

import pandas as pd

from primer.data.sources import SOURCES
from primer.data.storage import Store

DATASETS = ("stories_text", "drawings", "readability", "word_lists")


def read_dataset(dataset: str, store: Store | None = None, sources: list[str] | None = None) -> pd.DataFrame:
    store = store or Store()
    frames = []
    for folder in store.list_dirs(f"processed/{dataset}"):
        key = folder.removeprefix("source=")
        if sources and key not in sources:
            continue
        paths = store.glob(f"processed/{dataset}/{folder}/batch=*.parquet")
        source_cls = SOURCES.get(key)
        if source_cls is not None and source_cls.snapshot:
            paths = paths[-1:]
        frames += [store.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
