"""Bias slicing and mitigations, on small synthetic data and a temporary local store."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from primer.data import bias

# ------------------------------------------------------------------ gender

@pytest.mark.parametrize("text, swapped", [
    ("Tom saw her ball. He gave it to her.", "Lily saw his ball. She gave it to him."),
    ("She hugged her mom and Lily smiled.", "He hugged his dad and Tom smiled."),
    ("HE ran to the King.", "SHE ran to the Queen."),
    ("The cat sat on the mat.", "The cat sat on the mat."),
])
def test_gender_swap(text, swapped):
    assert bias.gender_swap(text) == swapped


@pytest.mark.parametrize("text, lead", [
    ("He and his dad went out. He was happy.", "male"),
    ("Lily and her mom baked. She smiled.", "female"),
    ("The cat sat on the mat.", "neutral"),
    ("He met her.", "neutral"),
])
def test_lead_gender(text, lead):
    assert bias.lead_gender(text) == lead


def test_balance_by_swap_reaches_parity_and_marks_copies():
    stories = pd.DataFrame({"text": ["He ran fast. He won."] * 6 + ["She sang. She smiled."] * 2 + ["A cat sat."]})
    out = bias.balance_by_swap(stories)
    assert (out["lead"] == "male").sum() == (out["lead"] == "female").sum() == 6
    assert out["augmented"].sum() == 4
    assert (out.loc[out["augmented"], "lead"] == "female").all()
    assert (out.loc[~out["augmented"], "text"] == stories["text"]).all()   # originals untouched


# ------------------------------------------------------------------ slices

def test_representation_pools_the_tail():
    table = bias.representation(pd.Series(["US"] * 6 + ["GB"] * 2 + ["DE", "FR"]), top=2)
    assert table["count"].to_dict() == {"US": 6, "GB": 2, "Other": 2}
    assert table["share"].sum() == pytest.approx(1.0)


def test_recognition_by_country_reports_the_gap():
    df = pd.DataFrame({"country_code": ["US"] * 10 + ["IN"] * 10 + ["FR", "DE"],
                       "recognized": [True] * 9 + [False] + [True] * 6 + [False] * 4 + [True, False]})
    out = bias.recognition_by_country(df, top=2)
    rates = out["by_country"]["recognition_rate"].to_dict()
    assert rates == {"US": pytest.approx(0.9), "IN": pytest.approx(0.6), "Other": pytest.approx(0.5)}
    assert out["gap"] == pytest.approx(0.4)
    assert out["worst"] == ("Other", pytest.approx(0.5))


def test_readability_baseline_detects_a_slice_it_gets_wrong():
    rng = np.random.default_rng(0)
    fk = rng.uniform(2, 12, 400)
    old = np.arange(400) < 200
    score = -0.3 * fk + 1.0 - np.where(old, 0.8, 0.0)       # old texts are harder than FK suggests
    clear = pd.DataFrame({"score": score, "flesch_kincaid_grade": fk,
                          "category": np.where(rng.random(400) < 0.5, "Lit", "Info"),
                          "pub_year": np.where(old, 1880, 2010)})
    out = bias.readability_baseline_by_slice(clear)
    era = out["era"]
    assert era.loc["1850–1899", "mean_error"] > 0.3        # baseline thinks old texts are easier
    assert era.loc["2000+", "mean_error"] < -0.3
    assert out["era_mae_gap"] < out["category_mae_gap"] + 1   # computed, finite
    assert set(out["category"].index) == {"Lit", "Info"}


# ------------------------------------------------------------------ mitigations

def test_capped_weights_cap_the_big_group_and_keep_the_rest():
    groups = pd.Series(["US"] * 70 + ["GB"] * 20 + ["DE"] * 10)
    weights = bias.capped_weights(groups, max_share=0.4)
    shares = bias.effective_shares(groups, weights)
    assert shares["US"] == pytest.approx(0.4, abs=1e-4)
    assert (weights[groups != "US"] == 1.0).all()


def test_capped_weights_iterates_when_capping_pushes_others_over():
    groups = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 10 + ["D"] * 10)
    shares = bias.effective_shares(groups, bias.capped_weights(groups, max_share=0.3))
    assert shares.max() <= 0.3 + 1e-9
    assert shares["A"] == pytest.approx(shares["B"])


def test_stratify_key():
    df = pd.DataFrame({"category": ["Lit", "Info"], "era": ["2000+", "1850–1899"]})
    assert bias.stratify_key(df, ["category", "era"]).tolist() == ["Lit / 2000+", "Info / 1850–1899"]


def test_balance_by_swap_balances_each_source_separately():
    """A big balanced source must not hide a small skewed one."""
    stories = pd.DataFrame({
        "source": ["big"] * 10 + ["small"] * 4,
        "text": ["He ran."] * 5 + ["She ran."] * 5 + ["He ran."] * 4,
    })
    out = bias.balance_by_swap(stories, by="source")
    counts = pd.crosstab(out["source"], out["lead"])
    assert counts.loc["big", "male"] == counts.loc["big", "female"] == 5
    assert counts.loc["small", "male"] == counts.loc["small", "female"] == 4
    assert out.loc[out["augmented"], "source"].eq("small").all()


# ------------------------------------------------------------------ end to end

def _write_datasets(store):
    store.write_parquet("processed/drawings/source=quickdraw/batch=1.parquet",
                        pd.DataFrame({"country_code": ["US"] * 8 + ["GB", "IN"], "label": "cat"}))
    raw = [{"countrycode": c, "recognized": r} for c, r in
           [("US", True)] * 8 + [("US", False), ("GB", True), ("IN", False), ("IN", True)]]
    store.write_bytes("raw/quickdraw/1/part-00000.jsonl.gz",
                      gzip.compress("\n".join(json.dumps(r) for r in raw).encode()))
    store.write_parquet("processed/stories_text/source=tinystories/batch=1.parquet",
                        pd.DataFrame({"source": "tinystories",
                                      "text": ["He ran. He won."] * 3 + ["She sang."]}))
    store.write_parquet("processed/readability/source=clear_corpus/batch=1.parquet",
                        pd.DataFrame({"score": [-1.0, -0.5, 0.0, 0.5], "flesch_kincaid_grade": [10, 8, 6, 4],
                                      "category": ["Lit", "Info", "Lit", "Info"],
                                      "pub_year": [1880, 1920, 1990, 2010]}))
    store.write_parquet("processed/storybook_pages/source=storyweaver/batch=1.parquet",
                        pd.DataFrame({"source": "storyweaver", "book_id": ["b1", "b1", "b2"],
                                      "page_number": [1, 2, 1], "image_uri": ["x", None, "y"],
                                      "word_count": [10, 20, 30]}))


def test_build_and_render_report(store):
    _write_datasets(store)
    report = bias.build_report(store, max_country_share=0.5)
    assert report["drawings"]["shares_before"]["US"] == pytest.approx(0.8)
    assert report["drawings"]["shares_after"]["US"] == pytest.approx(0.5, abs=1e-4)
    assert report["drawings"]["raw_rows"] == 12
    assert report["stories"]["after_counts"]["male"] == report["stories"]["after_counts"]["female"]
    md = bias.render_markdown(report)
    for heading in ("Drawings by country", "lead-character gender", "Readability (CLEAR)", "Picture books"):
        assert heading in md
    json.dumps(bias.report_to_json(report), default=str)    # serialisable


def test_excerpts_without_a_year_go_in_an_unknown_era():
    """Regression: missing pub_year became a missing era, which Fairlearn's MetricFrame rejects."""
    assert bias.era(pd.Series([1880.0, None, 2010.0])).tolist() == ["1850–1899", "unknown", "2000+"]
    clear = pd.DataFrame({"score": [-1.0, -0.5, 0.0, 0.5], "flesch_kincaid_grade": [10, 8, 6, 4],
                          "category": ["Lit", "Info", "Lit", "Info"], "pub_year": [1880, None, 1990, 2010]})
    assert "unknown" in bias.readability_baseline_by_slice(clear)["era"].index
