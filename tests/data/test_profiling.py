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


def test_credit_columns_may_be_empty():
    # The storybook source keeps an untitled book as title "" and an empty credit as "".
    df = frame()
    df = df.assign(title=[f"Book {i}" for i in range(len(df))], illustrator=[f"Artist {i}" for i in range(len(df))])
    schema = profiling.infer_schema("t", profiling.profile_frame(df))
    assert "min_length" not in schema["columns"]["title"]
    assert "min_length" not in schema["columns"]["illustrator"]
    df.loc[0, ["title", "illustrator"]] = ""
    assert profiling.validate(df, schema) == []
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


def test_above_range_warns_once_per_new_high_not_every_run(schema):
    high = frame()
    high.loc[0, "word_count"] = 100_000
    first = profiling.profile_frame(high)
    assert len(profiling.range_warnings(schema, first)) == 1                 # first time above max
    assert profiling.range_warnings(schema, first, previous=first) == []      # same data next run
    higher = high.copy()
    higher.loc[1, "word_count"] = 200_000
    [again] = profiling.range_warnings(schema, profiling.profile_frame(higher), previous=first)
    assert again.column == "word_count"                                       # a new high warns again


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


def test_write_schemas_keeps_the_date_of_an_unchanged_contract(store, tmp_path):
    schemas = tmp_path / "schemas"
    _write(store, frame(), "20261001T000000Z")
    path = profiling.write_schemas(store, schemas_dir=schemas, datasets=("readability",))[0]
    first = path.read_bytes()
    assert b"\r\n" not in first                                   # LF, as git stores it

    profiling.write_schemas(store, schemas_dir=schemas, datasets=("readability",))
    assert path.read_bytes() == first                             # nothing changed: no diff at all

    _write(store, frame(seed=3), "20261002T000000Z")              # new data, new contract and date
    profiling.write_schemas(store, schemas_dir=schemas, datasets=("readability",))
    assert json.loads(path.read_text())["created_at"] != json.loads(first)["created_at"]


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
    real_load = profiling.load_dataset

    def load(store, dataset, min_lengths=None):
        if dataset == "drawings":
            raise OSError("corrupt parquet")
        return real_load(store, dataset, min_lengths)

    monkeypatch.setattr(profiling, "load_dataset", load)
    report = profiling.check_datasets(store, run_id="r", schemas_dir=schemas, datasets=("drawings", "readability"))
    assert [(a["dataset"], a["check"]) for a in report["anomalies"]] == [("drawings", "check_failed")]
    assert report["datasets"]["readability"]["errors"] == 0
    assert store.exists("metadata/anomalies/r.json")


# ------------------------------------------------------------------ long text (KAN-91)
#
# 1.8M stories made Great Expectations need over 7 GB, so long text columns are checked from exact
# statistics gathered a chunk at a time instead. These tests pin that the results do not change.

def _write_stories(store, df, parts=2):
    """Write ``df`` as several TinyStories batches, an incremental source, so every file is read."""
    for i, chunk in enumerate(np.array_split(np.arange(len(df)), parts)):
        store.write_parquet(f"processed/stories_text/source=tinystories/batch=2026100{i + 1}T000000Z.parquet",
                            df.iloc[chunk].reset_index(drop=True))


def stories(n=120):
    df = frame(n)
    df.loc[3, "text"] = pd.NA                     # a missing story, allowed by the schema below
    df.loc[5, "text"] = "short " + "x" * 400       # lengths differ, so min/max/mean are tested
    return df


def test_long_text_never_reaches_great_expectations(store, tmp_path, monkeypatch):
    schemas = tmp_path / "schemas"
    _write_stories(store, stories())
    profiling.write_schemas(store, schemas_dir=schemas, datasets=("stories_text",))
    seen = []
    real = profiling._gx_ready
    monkeypatch.setattr(profiling, "_gx_ready", lambda df: seen.append(list(df.columns)) or real(df))

    report = profiling.check_datasets(store, run_id="r", save=False, schemas_dir=schemas, datasets=("stories_text",))

    assert report["errors"] == 0
    assert seen and all("text" not in columns for columns in seen)
    assert all("word_count" in columns for columns in seen)   # the other columns still go through GX


def test_streamed_profile_matches_the_in_memory_profile(store, monkeypatch):
    monkeypatch.setattr(profiling, "TEXT_BATCH_ROWS", 7)   # many chunks per file, so merging is exercised
    df = stories()
    _write_stories(store, df, parts=3)

    loaded = profiling.load_dataset(store, "stories_text")

    assert set(loaded.text) == {"text"}                     # only the long column is streamed
    assert "text" not in loaded.frame.columns
    assert loaded.profile() == profiling.profile_frame(df)
    assert list(loaded.profile()["columns"]) == list(df.columns)


def test_a_text_column_missing_from_one_file_counts_as_null_there(store):
    df = stories()
    first, second = df.iloc[:60].reset_index(drop=True), df.iloc[60:].reset_index(drop=True)
    store.write_parquet("processed/stories_text/source=tinystories/batch=20261001T000000Z.parquet", first)
    store.write_parquet("processed/stories_text/source=tinystories/batch=20261002T000000Z.parquet",
                        second.drop(columns="text"))   # an older batch written before the column existed

    loaded = profiling.load_dataset(store, "stories_text")

    assert loaded.text["text"].nulls == 60 + int(first["text"].isna().sum())
    assert loaded.profile() == profiling.profile_frame(pd.concat([first, second.drop(columns="text")],
                                                                 ignore_index=True))


@pytest.mark.parametrize("change, expected", [
    (lambda df: df, set()),
    (lambda df: df.assign(text=df["text"].where(df.index > 20)),
     {("expect_column_values_to_not_be_null", "text")}),
    (lambda df: df.assign(text=df["text"].mask(df.index == 0, "")),
     {("expect_column_value_lengths_to_be_between", "text")}),
    (lambda df: df.drop(columns="text"), {("expect_table_columns_to_match_set", None),   # and the column's
                                           ("expect_column_values_to_not_be_null", "text"),  # own checks,
                                           ("expect_column_value_lengths_to_be_between", "text")}),  # as GX does
    (lambda df: df.assign(notes="note " * 60), {("expect_table_columns_to_match_set", None)}),
    (lambda df: df.assign(notes="note " * 60).drop(columns="score"),  # one column-set break, not two,
     {("expect_table_columns_to_match_set", None),                    # plus the missing column's checks
      ("expect_column_values_to_be_between", "score"), ("expect_column_values_to_not_be_null", "score")}),
], ids=["clean", "too_many_nulls", "one_empty_story", "text_missing", "extra_long_column",
        "extra_long_and_missing_short"])
def test_long_text_contract_breaks_match_the_full_frame_check(tmp_path, context, change, expected):
    from primer.data.storage import Store

    schema = profiling.infer_schema("stories_text", profiling.profile_frame(stories()))
    df = change(stories())
    store = Store((tmp_path / "data").as_posix())
    _write_stories(store, df)
    loaded = profiling.load_dataset(store, "stories_text", min_lengths=profiling.min_lengths(schema))

    # Lists, not sets: a contract break must be reported as many times as the full frame reports it.
    streamed = sorted((a.check, a.column or "") for a in profiling.validate(loaded.frame, schema, context, loaded.text))
    full = sorted((a.check, a.column or "") for a in profiling.validate(df, schema, context))

    assert streamed == full == sorted((check, column or "") for check, column in expected)


def test_check_datasets_catches_one_empty_story(store, tmp_path):
    """End to end: ``check_datasets`` must count short values while streaming, or this passes silently."""
    schemas = tmp_path / "schemas"
    _write_stories(store, stories())
    profiling.write_schemas(store, schemas_dir=schemas, datasets=("stories_text",))
    emptied = stories()
    emptied.loc[7, "text"] = ""
    for path in store.glob("processed/stories_text/source=tinystories/*.parquet"):
        store.fs.rm(store.path(path))
    _write_stories(store, emptied)

    report = profiling.check_datasets(store, run_id="r", save=False, schemas_dir=schemas, datasets=("stories_text",))

    assert {(a["check"], a["column"]) for a in report["anomalies"]} == {
        ("expect_column_value_lengths_to_be_between", "text")}


def test_a_min_length_the_stream_did_not_count_is_refused(tmp_path, context):
    from primer.data.storage import Store

    schema = profiling.infer_schema("stories_text", profiling.profile_frame(stories()))
    store = Store((tmp_path / "data").as_posix())
    _write_stories(store, stories())
    loaded = profiling.load_dataset(store, "stories_text")    # no min_lengths: short values not counted

    with pytest.raises(ValueError, match="min_lengths"):
        profiling.validate(loaded.frame, schema, context, loaded.text)


def test_a_column_that_is_not_text_in_every_file_is_loaded_not_streamed(store):
    df = stories()
    first = df.iloc[:60].reset_index(drop=True).assign(notes="note " * 60)
    second = df.iloc[60:].reset_index(drop=True).assign(notes=7)    # the same column, as a number
    store.write_parquet("processed/stories_text/source=tinystories/batch=20261001T000000Z.parquet", first)
    store.write_parquet("processed/stories_text/source=tinystories/batch=20261002T000000Z.parquet", second)

    loaded = profiling.load_dataset(store, "stories_text")

    assert "notes" not in loaded.text and "notes" in loaded.frame.columns
    assert loaded.profile() == profiling.profile_frame(pd.concat([first, second], ignore_index=True))


def test_write_schemas_never_loads_long_text_whole(store, tmp_path, monkeypatch):
    _write_stories(store, stories())
    profiled = []
    real = profiling.profile_column
    monkeypatch.setattr(profiling, "profile_column", lambda s: profiled.append(s.name) or real(s))

    profiling.write_schemas(store, schemas_dir=tmp_path / "schemas", datasets=("stories_text",))

    assert profiled and "text" not in profiled and "word_count" in profiled


def test_a_categorical_column_that_turns_into_long_text_is_an_error(tmp_path, context):
    from primer.data.storage import Store

    schema = profiling.infer_schema("stories_text", profiling.profile_frame(stories()))
    assert "allowed" in schema["columns"]["source"]
    store = Store((tmp_path / "data").as_posix())
    _write_stories(store, stories().assign(source="tinystories " * 30))
    loaded = profiling.load_dataset(store, "stories_text", min_lengths=profiling.min_lengths(schema))

    found = {(a.check, a.column) for a in profiling.validate(loaded.frame, schema, context, loaded.text)}
    assert ("expect_column_values_to_be_in_set", "source") in found


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
    for name in profiling.CREDIT_COLUMNS & set(committed["columns"]):
        assert "min_length" not in committed["columns"][name], (dataset, name)
    if dataset == "storybook_pages":
        assert committed["columns"]["page_number"]["min"] > 0      # 1-based (schemas.py), so 0 must fail
