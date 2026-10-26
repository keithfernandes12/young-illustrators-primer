"""Quick, Draw!, CLEAR, and word-list sources, run against fakes and a temp store."""

import io
import re
import time
import zipfile

import pandas as pd
import pytest

from primer.data import pipeline
from primer.data.catalog import read_dataset
from primer.data.config import SourceConfig
from primer.data.manifest import ValidationFailed
from primer.data.sources import word_lists
from primer.data.sources.clear_corpus import COLUMNS as CLEAR_COLUMNS
from primer.data.sources.clear_corpus import ClearCorpus
from primer.data.sources.quickdraw import QuickDraw, valid_strokes
from primer.data.sources.word_lists import KupermanAoA, SightWords


def config(key, **ingest):
    return SourceConfig(key=key, name=key, license="TEST-LICENSE",
                        attribution=f"{key} attribution", ingest=ingest)


FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
WORKBOOK_TIMESTAMPS = re.compile(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:)")


def xlsx_bytes(df, sheet="Data"):
    """The same frame always gives the same bytes.

    An .xlsx is a zip whose entries carry their write time, and whose docProps/core.xml records
    when the workbook was created and modified. Left as is, two builds of one frame a second apart
    hash differently, so snapshot sources see an "unchanged" file as new and tests flake.
    """
    buf = io.BytesIO()
    df.to_excel(buf, sheet_name=sheet, index=False)
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename == "docProps/core.xml":
                data = WORKBOOK_TIMESTAMPS.sub(rb"\g<1>2026-01-01T00:00:00Z\g<2>", data)
            dst.writestr(zipfile.ZipInfo(info.filename, date_time=FIXED_ZIP_TIME), data,
                         compress_type=zipfile.ZIP_DEFLATED)
    return out.getvalue()


@pytest.fixture
def batch_ids(monkeypatch):
    ids = iter(f"2026101{i}T000000Z" for i in range(10))
    monkeypatch.setattr(pipeline, "new_batch_id", lambda: next(ids))


# ----------------------------------------------------------------- Quick, Draw!

STROKE = [[0, 10, 20], [5, 15, 25]]


def drawing_rows(category, n, *, bad_every=0, unrecognized_every=0):
    rows = []
    for i in range(n):
        rows.append({
            "word": category.title(), "countrycode": "US", "timestamp": "2017-01-01",
            "recognized": not (unrecognized_every and i % unrecognized_every == 0),
            "key_id": f"{category}-{i}",
            "drawing": [[[1, 2], [3]]] if bad_every and i % bad_every == 0 else [STROKE, STROKE],
        })
    return rows


def make_quickdraw(store, categories=("cat", "dog", "sun"), **rows_kw):
    cfg = config("quickdraw", categories_per_batch=2, max_per_category=10, min_per_category=5)
    return QuickDraw(cfg, store, list_categories=lambda: list(categories),
                     read_category=lambda cat, limit: drawing_rows(cat, limit, **rows_kw))


def test_valid_strokes():
    assert valid_strokes([STROKE])
    assert not valid_strokes([])
    assert not valid_strokes([[[1, 2], [3]]])
    assert not valid_strokes("nope")


def test_quickdraw_batches_categories_and_preprocesses(store, batch_ids):
    source = make_quickdraw(store, unrecognized_every=5)
    first = pipeline.run_all(source)
    assert store.read_json(source.state_path)["next_category"] == 2

    df = store.read_parquet(source.processed_path(first))
    assert set(df["label"]) == {"cat", "dog"}
    assert len(df) == 16                      # 2 categories x 10, minus 2 unrecognized each
    assert df["n_strokes"].eq(2).all() and df["n_points"].eq(6).all()
    assert df["license"].eq("TEST-LICENSE").all()

    second = pipeline.run_all(source)
    assert set(store.read_parquet(source.processed_path(second))["label"]) == {"sun"}
    assert pipeline.ingest(source) is None    # all categories done
    assert len(read_dataset("drawings", store)) == 24


def test_quickdraw_validation_catches_malformed_strokes(store, batch_ids):
    source = make_quickdraw(store, bad_every=2)
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed, match="malformed_strokes"):
        pipeline.validate(source, batch)


# ------------------------------------------------------------------------ CLEAR

def clear_frame(n=5, **overrides):
    df = pd.DataFrame({
        "ID": range(100, 100 + n),
        "Title": [f"Story {i}" for i in range(n)],
        "Author": "Someone",
        "Excerpt": [f"This is excerpt number {i}. It is short." for i in range(n)],
        "BT_easiness": [-1.0 + i * 0.1 for i in range(n)],
        "s.e.": 0.4,
        "Flesch-Kincaid-Grade-Level": 5.5,
        "Lexile Band": ["900", 1000, None, "1100", "800"][:n],
        "Categ": "Lit",
        "Pub Year": [1900.0, None, 1950.0, 2000.0, 1914.0][:n],
        "License": None,
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def make_clear(store, frames):
    payloads = iter([xlsx_bytes(f) for f in frames])
    return ClearCorpus(config("clear_corpus", url="https://example.test/clear.xlsx", min_records=3),
                       store, download=lambda url: next(payloads))


def test_clear_snapshot_validate_preprocess(store, batch_ids):
    assert set(clear_frame().columns) == set(CLEAR_COLUMNS)
    source = make_clear(store, [clear_frame(), clear_frame(), clear_frame(n=4)])

    first = pipeline.run_all(source)
    df = store.read_parquet(source.processed_path(first))
    assert len(df) == 5
    assert df["score_name"].eq("BT_easiness").all()
    assert df["lexile_band"].iloc[:2].tolist() == ["900", "1000"]  # mixed int/str column, as in the real file
    assert df["lexile_band"].isna().sum() == 1
    assert df["pub_year"].isna().sum() == 1

    assert pipeline.ingest(source) is None               # same file: no new batch
    second = pipeline.run_all(source)                    # changed file: new snapshot
    assert second is not None
    assert len(read_dataset("readability", store)) == 4  # only the latest snapshot


def test_clear_validation_fails_on_bad_scores(store, batch_ids):
    source = make_clear(store, [clear_frame(BT_easiness=[0.0, 9.0, 0.1, None, 0.2])])
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed) as err:
        pipeline.validate(source, batch)
    assert "missing_score" in str(err.value) and "score_in_range" in str(err.value)


# ------------------------------------------------------------------- word lists

def test_sight_words(store, batch_ids, monkeypatch):
    dolch = [f"d{i}" for i in range(220)]
    fry = [f"f{i}" for i in range(999)] + ["f5"]   # one known duplicate, like "hot" in the real list
    files = {"dolch": "\n".join(dolch).encode(), "fry": "\n".join(fry).encode()}
    monkeypatch.setattr(word_lists, "read_rda_words", lambda data, name: data.decode().split("\n"))
    cfg = config("sight_words", dolch_url="dolch", fry_url="fry", allowed_duplicates=["f5"])
    source = SightWords(cfg, store, download=lambda url: files[url])

    batch = pipeline.run_all(source)
    df = store.read_parquet(source.processed_path(batch))
    counts = df[df["list_name"] == "dolch"]["level"].value_counts().to_dict()
    assert counts == {"primer": 52, "grade2": 46, "grade1": 41, "grade3": 41, "pre-primer": 40}
    fry_df = df[df["list_name"] == "fry"]
    assert fry_df.iloc[0][["word", "level", "rank"]].tolist() == ["f0", "fry_100", 1]
    assert len(fry_df) == 999 and fry_df.loc[fry_df["word"] == "f5", "rank"].item() == 6
    assert fry_df.iloc[-1]["level"] == "fry_1000"


def test_sight_words_unexpected_duplicate_fails(store, batch_ids, monkeypatch):
    files = {"dolch": "\n".join(f"d{i}" for i in range(220)).encode(),
             "fry": "\n".join([f"f{i}" for i in range(999)] + ["f5"]).encode()}
    monkeypatch.setattr(word_lists, "read_rda_words", lambda data, name: data.decode().split("\n"))
    source = SightWords(config("sight_words", dolch_url="dolch", fry_url="fry"), store,
                        download=lambda url: files[url])
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed, match="unique_words"):
        pipeline.validate(source, batch)


def test_kuperman(store, batch_ids):
    frame = pd.DataFrame({
        "Word": ["a", "aardvark", "abacus", "zebra", "unrated"],
        "AoA_Kup": [2.9, 9.9, 8.7, 4.0, None],
        "Perc_known": [1.0, 1.0, 0.65, 1.0, None],
        "Freq_pm": [20000.0, 0.4, 0.2, 3.0, 1.0],
        "Dom_PoS_SUBTLEX": ["Article", "Noun", "Noun", "Noun", None],
        "Nsyll": [1, 2, 3, 2, 2],
    })
    source = KupermanAoA(config("kuperman_aoa", url="u", min_rated=3), store,
                         download=lambda url: xlsx_bytes(frame, sheet="Sheet1"))
    batch = pipeline.run_all(source)
    df = store.read_parquet(source.processed_path(batch))
    assert df["word"].tolist() == ["a", "aardvark", "abacus", "zebra"]
    assert df["list_name"].eq("kuperman_aoa").all()
    assert df.loc[df["word"] == "zebra", "age_of_acquisition"].item() == 4.0



def test_xlsx_bytes_is_deterministic():
    """Guards the fix for a 1-in-40 flake: identical frames built in different seconds must match."""
    first = xlsx_bytes(clear_frame())
    second_started = int(time.time())
    while int(time.time()) == second_started:  # cross a wall-clock second boundary (at most 1 s)
        time.sleep(0.05)
    assert xlsx_bytes(clear_frame()) == first
    back = pd.read_excel(io.BytesIO(first))
    assert list(back.columns) == list(clear_frame().columns) and len(back) == len(clear_frame())
