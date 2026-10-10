import pandas as pd


def test_json_roundtrip_and_default(store):
    store.write_json("metadata/state/x.json", {"next_offset": 5})
    assert store.read_json("metadata/state/x.json") == {"next_offset": 5}
    assert store.read_json("metadata/state/missing.json", default={}) == {}


def test_jsonl_gz_roundtrip(store):
    records = [{"source_id": "1", "text": "Hello, world"}, {"source_id": "2", "text": "Ünïcödé"}]
    store.write_jsonl_gz("raw/s/b1/part-00000.jsonl.gz", records)
    assert store.read_jsonl_gz("raw/s/b1/part-00000.jsonl.gz") == records


def test_parquet_roundtrip_with_columns(store):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    store.write_parquet("processed/t/batch=1.parquet", df)
    assert store.read_parquet("processed/t/batch=1.parquet", columns=["b"])["b"].tolist() == ["x", "y"]


def test_glob_and_list_dirs_return_relative_paths(store):
    store.write_json("raw/s/b1/_batch.json", {})
    store.write_json("raw/s/b2/_batch.json", {})
    assert store.glob("raw/s/*/_batch.json") == ["raw/s/b1/_batch.json", "raw/s/b2/_batch.json"]
    assert store.list_dirs("raw/s") == ["b1", "b2"]
    assert store.list_dirs("raw/nothing") == []
    assert store.exists("raw/s/b1/_batch.json")
