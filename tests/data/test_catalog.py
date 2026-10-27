"""Which processed files make up a dataset."""

import pandas as pd

from primer.data.catalog import dataset_files, read_dataset


def test_snapshot_latest_batch_respects_repeated_batch_ids(store):
    """Regression: batch=X-2.parquet sorted before batch=X.parquet, so the older snapshot was read."""
    folder = "processed/readability/source=clear_corpus"   # clear_corpus is a snapshot source
    store.write_parquet(f"{folder}/batch=20261026T000000Z.parquet", pd.DataFrame({"v": ["old"]}))
    store.write_parquet(f"{folder}/batch=20261026T000000Z-2.parquet", pd.DataFrame({"v": ["new"]}))

    assert dataset_files(store, "readability") == {"clear_corpus": [f"{folder}/batch=20261026T000000Z-2.parquet"]}
    assert read_dataset("readability", store)["v"].tolist() == ["new"]


def test_incremental_sources_keep_every_batch_in_order(store):
    folder = "processed/stories_text/source=tinystories"
    for batch in ("20261026T000001Z", "20261026T000000Z-2", "20261026T000000Z"):
        store.write_parquet(f"{folder}/batch={batch}.parquet", pd.DataFrame({"b": [batch]}))
    assert read_dataset("stories_text", store)["b"].tolist() == [
        "20261026T000000Z", "20261026T000000Z-2", "20261026T000001Z"]
