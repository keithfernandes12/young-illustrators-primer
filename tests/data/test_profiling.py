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
    assert cols["text"]["min_length"] >= 1 and "min_length" not in cols["word_count"]
    assert schema["min_rows"] == 100


def test_infer_schema_floor_is_zero_only_if_zero_was_seen():
    df = pd.DataFrame({"page_number": pd.array(range(1, 101), dtype="int32"),       # 1-based
                       "n_strokes": pd.array(range(100), dtype="int32"),
                       "delta": np.linspace(-5.0, 5.0, 100)})
    cols = profiling.infer_schema("t", profiling.profile_frame(df))["columns"]
    assert 0 < cols["page_number"]["min"] < 1      # page 0 is invalid, so it must fail
    assert cols["n_strokes"]["min"] == 0.0         # 0 was seen: still allowed, never negative
    assert cols["delta"]["min"] == pytest.approx(-6.0)


def test_infer_schema_never_lets_a_column_go_all_null(context):
    n = 1000
    df = pd.DataFrame({"mostly_null": pd.array([1.0] * 20 + [None] * (n - 20), dtype="Float64"),
                       "always_null": pd.array([None] * n, dtype="string")})
    schema = profiling.infer_schema("t", profiling.profile_frame(df))
    cols = schema["columns"]
    assert cols["mostly_null"]["not_all_null"] is True and "nulls_unconstrained" not in cols["mostly_null"]
    assert cols["always_null"]["nulls_unconstrained"] is True   # an explicit decision, not a side effect

    emptied = df.assign(mostly_null=pd.array([None] * n, dtype="Float64"))
    grown = pd.concat([df] + [emptied] * 5, ignore_index=True)        # 98% → 99.7% null
    assert profiling.validate(grown, schema, context) == []           # a growing source stays green
    assert ("expect_column_values_to_not_be_null", "mostly_null") in checks_failed(emptied, schema, context)


def test_tiny_values_are_rounded_to_significant_figures_not_zero():
    one_null = pd.Series([None] + [1.0] * 2_999_999, dtype="Float64")
    assert profiling.profile_column(one_null)["null_fraction"] > 0
    tiny = pd.DataFrame({"p": np.linspace(1e-9, 5e-9, 50)})
    spec = profiling.infer_schema("t", profiling.profile_frame(tiny))["columns"]["p"]
    assert 0 < spec["min"] <= 1e-9 and spec["max"] >= 5e-9   # still contains the data it came from


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

    below_floor = df.copy()
    below_floor.loc[0, "word_count"] = 0
    assert ("expect_column_values_to_be_between", "word_count") in checks_failed(below_floor, schema, context)

    emptied = df.copy()
    emptied["text"] = ""
    assert ("expect_column_value_lengths_to_be_between", "text") in checks_failed(emptied, schema, context)

    all_null = df.copy()
    all_null["pub_year"] = pd.NA
    assert ("expect_column_values_to_not_be_null", "pub_year") in checks_failed(all_null, schema, context)

    new_category = df.copy()
    new_category.loc[0, "source"] = "unknown_source"
    assert ("expect_column_values_to_be_in_set", "source") in checks_failed(new_category, schema, context)

    assert ("expect_table_row_count_to_be_between", None) in checks_failed(df.head(10), schema, context)


def test_going_above_the_inferred_max_is_a_warning_not_an_error(schema, context):
    df = frame()
    df.loc[0, "word_count"] = 100_000       # e.g. a new Quick, Draw! category with more strokes
    assert profiling.validate(df, schema, context) == []
    [warning] = profiling.range_warnings(schema, profiling.profile_frame(df))
    assert (warning.severity, warning.check, warning.column) == ("warning", "above_range", "word_count")


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
    assert ("length_shift", "text") not in checks
    assert all(a.severity == "warning" for a in found)


def test_drift_sees_what_shares_and_standard_deviations_miss():
    def checks(before, after):
        found = profiling.drift("t", profiling.profile_frame(before), profiling.profile_frame(after))
        return {(a.check, a.column) for a in found}

    n = 200
    constant = pd.DataFrame({"x": [0.0] * n})
    assert checks(constant, constant) == set()
    assert ("mean_shift", "x") in checks(constant, pd.DataFrame({"x": [1000.0] * n}))   # sd was 0

    one_value = pd.DataFrame({"c": pd.array(["a"] * n, dtype="string")})
    many_values = pd.DataFrame({"c": pd.array([f"v{i % 60}" for i in range(n)], dtype="string")})
    assert ("cardinality_shift", "c") in checks(one_value, many_values)   # shares → top: shares can't compare
    assert ("cardinality_shift", "c") in checks(many_values, one_value)

    texts = pd.DataFrame({"text": pd.array(["word " * 60] * n, dtype="string")})
    assert ("length_shift", "text") in checks(texts, pd.DataFrame({"text": pd.array(["w"] * n, dtype="string")}))


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


def test_missing_schema_is_an_error_and_nothing_is_saved_without_save(store, tmp_path):
    _write(store, frame(), "20261001T000000Z")
    report = profiling.check_datasets(store, run_id="r", save=False, schemas_dir=tmp_path / "none",
                                      datasets=("readability",))
    assert [a["check"] for a in report["anomalies"]] == ["no_schema"]
    assert report["errors"] == 1          # nothing was validated, so the run must not look green
    assert not store.glob("metadata/profiles/*/*.json") and not store.glob("metadata/anomalies/*.json")


def test_a_dataset_that_loses_all_its_rows_is_an_error(store, tmp_path):
    schemas = tmp_path / "schemas"
    _write(store, frame(), "20261001T000000Z")
    profiling.write_schemas(store, schemas_dir=schemas, datasets=("readability",))
    profiling.check_datasets(store, run_id="run1", schemas_dir=schemas, datasets=("readability",))

    _write(store, frame().head(0), "20261002T000000Z")      # the upstream feed returned nothing
    report = profiling.check_datasets(store, run_id="run2", schemas_dir=schemas, datasets=("readability",))
    assert ("error", "empty_dataset") in {(a["severity"], a["check"]) for a in report["anomalies"]}
    assert report["datasets"]["readability"]["rows"] == 0

    again = profiling.check_datasets(store, run_id="run3", schemas_dir=schemas, datasets=("readability",))
    assert [a["check"] for a in again["anomalies"]] == ["empty_dataset"]   # still red, not silently skipped
    assert again["datasets"]["readability"]["compared_with"] == "run1"     # the last good profile


def test_a_dataset_that_never_had_data_is_not_reported(store, tmp_path):
    report = profiling.check_datasets(store, run_id="r", schemas_dir=tmp_path, datasets=("readability",))
    assert (report["datasets"], report["anomalies"]) == ({}, [])


def test_one_broken_dataset_does_not_stop_the_others_or_the_report(store, tmp_path, monkeypatch):
    schemas = tmp_path / "schemas"
    _write(store, frame(), "20261001T000000Z")
    profiling.write_schemas(store, schemas_dir=schemas, datasets=("readability",))
    real_read = profiling.read_dataset

    def read(dataset, store):
        if dataset == "drawings":
            raise OSError("corrupt parquet")
        return real_read(dataset, store)

    monkeypatch.setattr(profiling, "read_dataset", read)
    report = profiling.check_datasets(store, run_id="r", schemas_dir=schemas, datasets=("drawings", "readability"))
    assert [(a["dataset"], a["check"]) for a in report["anomalies"]] == [("drawings", "check_failed")]
    assert report["datasets"]["readability"]["errors"] == 0
    assert store.exists("metadata/anomalies/r.json")


def test_an_unreadable_previous_profile_is_skipped(store):
    store.write_json("metadata/profiles/readability/old.json", {"run_id": "old", "created_at": "1", "rows": 5})
    store.write_bytes("metadata/profiles/readability/torn.json", b'{"run_id": "torn", "rows"')
    assert profiling.previous_profile(store, "readability")["run_id"] == "old"


# ------------------------------------------------------------------ committed schemas

@pytest.mark.parametrize("dataset", profiling.DATASETS)
def test_committed_schema_matches_the_parquet_schema(dataset):
    from primer.data import schemas

    committed = profiling.load_schema(dataset)
    assert committed is not None, f"configs/schemas/{dataset}.json is missing: run primer-data profile --write-schemas"
    assert committed["dataset"] == dataset and committed["min_rows"] > 0
    assert set(committed["columns"]) == set(getattr(schemas, dataset.upper()))
    assert "allowed" not in committed["columns"]["batch_id"]
    for name, spec in committed["columns"].items():
        # "may be all null" must be an explicit decision, not a side effect of the headroom
        explicit = spec.get("nulls_unconstrained") or spec.get("not_all_null")
        assert (spec["max_null_fraction"] >= 1.0) == bool(explicit), (dataset, name)
    if dataset == "storybook_pages":
        assert committed["columns"]["page_number"]["min"] > 0      # 1-based (schemas.py), so 0 must fail
