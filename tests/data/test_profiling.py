"""Statistics, schema inference, Great Expectations validation and drift, on small synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from primer.data import profiling


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "word_count": pd.array(rng.integers(20, 200, n), dtype="int32"),
        "score": rng.normal(-1.0, 0.5, n),
        "source": pd.array(rng.choice(["tinystories", "storyweaver"], n), dtype="string"),
        "recognized": pd.array(rng.random(n) < 0.9, dtype="boolean"),
        "text": pd.array([f"story {i} " + "word " * 60 for i in range(n)], dtype="string"),
        "pub_year": pd.array([1900 + i % 100 if i % 10 else None for i in range(n)], dtype="Int32"),
    })


# ------------------------------------------------------------------ statistics

def test_profile_column_kinds():
    p = profiling.profile_frame(frame())["columns"]
    assert p["word_count"]["kind"] == "int" and 20 <= p["word_count"]["min"] <= p["word_count"]["max"] < 200
    assert p["score"]["kind"] == "float" and "std" in p["score"]
    assert p["recognized"]["kind"] == "bool" and 0.8 < p["recognized"]["true_fraction"] < 1.0
    assert set(p["source"]["shares"]) == {"tinystories", "storyweaver"}          # categorical
    assert "distinct" not in p["text"] and p["text"]["mean_length"] > profiling.LONG_TEXT   # long text
    assert p["pub_year"]["null_fraction"] == pytest.approx(0.1)


# ------------------------------------------------------------------ schema

def test_infer_schema_ranges_nulls_and_allowed_values():
    schema = profiling.infer_schema("t", profiling.profile_frame(frame()))
    cols = schema["columns"]
    assert cols["word_count"]["min"] >= 0                  # non-negative stays non-negative
    assert cols["word_count"]["max"] > 199
    assert cols["source"]["allowed"] == ["storyweaver", "tinystories"]
    assert cols["text"]["max_null_fraction"] == 0.0
    assert cols["pub_year"]["max_null_fraction"] == pytest.approx(0.15)
    assert schema["min_rows"] == 100


def test_infer_schema_leaves_open_and_rare_text_columns_unrestricted():
    df = frame().assign(batch_id="20261001T000000Z", narrator=None)
    df.loc[:5, "narrator"] = ["a", "b", "c", "d", "e", "f"]   # six names on six rows: not a category
    cols = profiling.infer_schema("t", profiling.profile_frame(df))["columns"]
    assert "allowed" not in cols["batch_id"]       # every ingest adds a new batch
    assert "allowed" not in cols["narrator"]
    assert "allowed" in cols["source"]


@pytest.fixture(scope="module")
def schema():
    return profiling.infer_schema("t", profiling.profile_frame(frame()))


@pytest.fixture(scope="module")
def context():
    return profiling.gx_context()


def checks_failed(df, schema, context):
    return {(a.check, a.column) for a in profiling.validate(df, schema, context)}


def test_validate_passes_on_the_data_it_was_inferred_from(schema, context):
    assert profiling.validate(frame(), schema, context) == []


def test_validate_catches_each_contract_break(schema, context):
    df = frame()
    assert ("expect_table_columns_to_match_set", None) in checks_failed(df.drop(columns="score"), schema, context)
    assert ("expect_table_columns_to_match_set", None) in checks_failed(df.assign(extra=1), schema, context)

    nulls = df.copy()
    nulls.loc[:20, "text"] = pd.NA
    assert ("expect_column_values_to_not_be_null", "text") in checks_failed(nulls, schema, context)

    out_of_range = df.copy()
    out_of_range.loc[0, "word_count"] = 100_000
    assert ("expect_column_values_to_be_between", "word_count") in checks_failed(out_of_range, schema, context)

    new_category = df.copy()
    new_category.loc[0, "source"] = "unknown_source"
    assert ("expect_column_values_to_be_in_set", "source") in checks_failed(new_category, schema, context)

    assert ("expect_table_row_count_to_be_between", None) in checks_failed(df.head(10), schema, context)


# ------------------------------------------------------------------ drift

def test_drift_flags_large_changes_only():
    before = profiling.profile_frame(frame(seed=0))
    assert profiling.drift("t", before, profiling.profile_frame(frame(seed=1))) == []   # same distribution

    shifted = frame(seed=0)
    shifted["score"] = shifted["score"] + 2.0
    shifted.loc[:40, "pub_year"] = pd.NA
    shifted["source"] = "tinystories"
    found = profiling.drift("t", before, profiling.profile_frame(shifted.head(150)))
    checks = {(a.check, a.column) for a in found}
    assert {("row_drop", None), ("mean_shift", "score"), ("null_jump", "pub_year"), ("share_shift", "source")} <= checks
    assert ("mean_shift", "word_count") not in checks          # an unchanged column isn't flagged
    assert all(a.severity == "warning" for a in found)


# ------------------------------------------------------------------ end to end

def _write(store, df, batch):
    store.write_parquet(f"processed/readability/source=clear_corpus/batch={batch}.parquet", df)


def test_check_datasets_saves_profiles_and_reports_errors_and_drift(store, tmp_path):
    schemas = tmp_path / "schemas"
    _write(store, frame(), "20261001T000000Z")
    written = profiling.write_schemas(store, schemas_dir=schemas, datasets=("readability",))
    assert [p.name for p in written] == ["readability.json"]
    assert json.loads(written[0].read_text())["dataset"] == "readability"

    first = profiling.check_datasets(store, run_id="run1", schemas_dir=schemas, datasets=("readability",))
    assert (first["errors"], first["warnings"]) == (0, 0)
    assert store.exists("metadata/profiles/readability/run1.json")
    assert store.exists("metadata/anomalies/run1.json")

    # clear_corpus is a snapshot source: a newer batch replaces the old one. Make it break the contract.
    bad = frame(seed=3).head(60)
    bad.loc[0, "source"] = "mystery"
    _write(store, bad, "20261002T000000Z")
    second = profiling.check_datasets(store, run_id="run2", schemas_dir=schemas, datasets=("readability",))
    checks = {a["check"] for a in second["anomalies"]}
    assert "expect_column_values_to_be_in_set" in checks and "expect_table_row_count_to_be_between" in checks
    assert "row_drop" in checks                                   # drift against run1's profile
    assert second["datasets"]["readability"]["compared_with"] == "run1"
    text = profiling.format_anomalies(second)
    assert "ERROR readability [source]" in text and "WARNING readability: rows fell" in text


def test_missing_schema_is_a_warning_and_nothing_is_saved_without_save(store, tmp_path):
    _write(store, frame(), "20261001T000000Z")
    report = profiling.check_datasets(store, run_id="r", save=False, schemas_dir=tmp_path / "none",
                                      datasets=("readability",))
    assert [a["check"] for a in report["anomalies"]] == ["no_schema"]
    assert report["errors"] == 0
    assert not store.glob("metadata/profiles/*/*.json") and not store.glob("metadata/anomalies/*.json")


# ------------------------------------------------------------------ committed schemas

@pytest.mark.parametrize("dataset", profiling.DATASETS)
def test_committed_schema_matches_the_parquet_schema(dataset):
    from primer.data import schemas

    committed = profiling.load_schema(dataset)
    assert committed is not None, f"configs/schemas/{dataset}.json is missing: run primer-data profile --write-schemas"
    assert committed["dataset"] == dataset and committed["min_rows"] > 0
    assert set(committed["columns"]) == set(getattr(schemas, dataset.upper()))
    assert "allowed" not in committed["columns"]["batch_id"]
