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
    # possessives and contractions: the name or pronoun before the apostrophe still swaps
    ("Tom was sad. Tom's mom hugged him. He's happy now.", "Lily was sad. Lily's dad hugged her. She's happy now."),
    ("Lily's dog barked. She'd run home.", "Tom's dog barked. He'd run home."),
    # "his"/"hers" without a following noun
    ("The ball is his.", "The ball is hers."),
    ("She said it was hers.", "He said it was his."),
    ("Ms Brown and Mrs Lee", "Mr Brown and Mr Lee"),
])
def test_gender_swap(text, swapped):
    assert bias.gender_swap(text) == swapped


@pytest.mark.parametrize("text, lead", [
    ("He and his dad went out. He was happy.", "male"),
    ("Lily and her mom baked. She smiled.", "female"),
    ("The cat sat on the mat.", "neutral"),
    ("He met her.", "neutral"),
    ("Tom's dog ran. Tom's cat sat.", "male"),              # possessive names count
    ("She said it was hers.", "female"),
])
def test_lead_gender(text, lead):
    assert bias.lead_gender(text) == lead


def test_names_count_only_when_capitalised():
    assert bias.gender_counts("Jack picked a lily and the max") == (1, 0)   # the flower isn't a girl


def test_balance_by_swap_reaches_parity_and_marks_copies():
    stories = pd.DataFrame({"text": ["He ran fast. He won."] * 6 + ["She sang. She smiled."] * 2 + ["A cat sat."]})
    out = bias.balance_by_swap(stories, by=None)
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
    # the pooled "Other" group is shown but doesn't decide best, worst or gap
    assert out["gap"] == pytest.approx(0.3)
    assert out["worst"] == ("IN", pytest.approx(0.6)) and out["best"] == ("US", pytest.approx(0.9))
    assert out["top"] == 2


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
    assert out["era_mae_gap"] == pytest.approx(abs(era.loc["1850–1899", "mae"] - era.loc["2000+", "mae"]))
    assert set(out["category"].index) == {"Lit", "Info"}


def test_mae_gap_ignores_slices_too_small_to_judge():
    """Regression: a 9-excerpt slice inflated the era gap (0.322 vs 0.072 on real data)."""
    rng = np.random.default_rng(1)
    fk = rng.uniform(2, 12, 409)
    score = -0.3 * fk + 1.0 + rng.normal(0, 0.1, 409)
    score[400:] += 3.0                                        # 9 excerpts, wildly off
    years = np.r_[np.full(200, 1920.0), np.full(200, 2010.0), np.full(9, np.nan)]
    clear = pd.DataFrame({"score": score, "flesch_kincaid_grade": fk, "category": "Lit", "pub_year": years})
    out = bias.readability_baseline_by_slice(clear)
    big = out["era"].loc[["1900–1949", "2000+"], "mae"]
    assert out["era_mae_gap"] == pytest.approx(big.max() - big.min())
    assert out["era_mae_gap_all_slices"] > out["era_mae_gap"] + 1      # the unknown slice dominates the raw gap
    assert out["category_mae_gap"] is None                              # one genre: nothing to compare


# ------------------------------------------------------------------ mitigations

def test_capped_weights_cap_the_big_group_and_keep_the_rest():
    groups = pd.Series(["US"] * 70 + ["GB"] * 20 + ["DE"] * 10)
    weights = bias.capped_weights(groups, max_share=0.4)
    shares = bias.effective_shares(groups, weights)
    assert shares["US"] == pytest.approx(0.4, abs=1e-4)
    assert (weights[groups != "US"] == 1.0).all()


@pytest.mark.parametrize("groups, cap", [
    (["US"] * 10, 0.2),                       # one group can't be held to 20%
    (["A", "B", "C", "D"] * 5, 0.2),          # four groups need at least 25% each
    (["A", "B"], 0.0),
    (["A", "B"], 1.5),
])
def test_capped_weights_rejects_a_cap_that_cant_be_met(groups, cap):
    """Regression: an infeasible cap silently produced all-zero (or NaN) weights."""
    with pytest.raises(ValueError):
        bias.capped_weights(pd.Series(groups), max_share=cap)


def test_capped_weights_at_exactly_the_feasible_cap_gives_equal_shares():
    groups = pd.Series(["A"] * 70 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
    shares = bias.effective_shares(groups, bias.capped_weights(groups, max_share=0.25))
    assert shares.tolist() == pytest.approx([0.25] * 4)


def test_capped_weights_counts_missing_groups_as_unknown():
    groups = pd.Series(["US"] * 8 + [None] * 2)
    weights = bias.capped_weights(groups, max_share=0.5)
    assert weights.notna().all() and (weights > 0).all()


def test_capped_weights_iterates_when_capping_pushes_others_over():
    groups = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 10 + ["D"] * 10)
    shares = bias.effective_shares(groups, bias.capped_weights(groups, max_share=0.3))
    assert shares.max() <= 0.3 + 1e-9
    assert shares["A"] == pytest.approx(shares["B"])


def test_stratify_key():
    df = pd.DataFrame({"category": ["Lit", "Info"], "era": ["2000+", "1850–1899"]})
    assert bias.stratify_key(df, ["category", "era"]).tolist() == ["Lit / 2000+", "Info / 1850–1899"]


def test_stratify_key_handles_the_schemas_nullable_dtypes():
    """Regression: a missing genre in the schema's "string" dtype crashed the join."""
    df = pd.DataFrame({"category": pd.array(["Lit", pd.NA], dtype="string"),
                       "era": bias.era(pd.Series(pd.array([2010, None], dtype="Int32")))})
    assert bias.stratify_key(df, ["category", "era"]).tolist() == ["Lit / 2000+", "unknown / unknown"]


def test_split_strata_pools_small_strata_into_the_nearest_era():
    category = pd.Series(["Lit"] * 285 + ["Info"] * 20)
    eras = pd.Series(["2000+"] * 150 + ["1900–1949"] * 120 + ["1850–1899"] * 10 + ["unknown"] * 5
                     + ["2000+"] * 20)
    counts = bias.split_strata(category, eras, min_size=100).value_counts().to_dict()
    assert counts == {"Lit / 2000+": 155,           # 5 without a year join the genre's largest era
                      "Lit / 1900–1949": 130,       # 10 from 1850–1899 join the nearest big era
                      "Info / all eras": 20}        # no era big enough: one stratum


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


def test_balance_by_swap_reaches_parity_from_a_female_majority():
    """Regression: "hers" had no swap target, so swapped copies came out neutral."""
    stories = pd.DataFrame({"text": ["She said it was hers."] * 3 + ["He ran."]})
    out = bias.balance_by_swap(stories, by=None)
    assert (out["lead"] == "male").sum() == (out["lead"] == "female").sum() == 3


def test_balance_by_swap_rejects_a_missing_column_and_keeps_rows_without_a_source():
    stories = pd.DataFrame({"text": ["He ran."] * 3 + ["She ran."], "source": [None] * 4})
    with pytest.raises(KeyError):
        bias.balance_by_swap(stories.drop(columns="source"), by="source")
    out = bias.balance_by_swap(stories, by="source")
    assert out["augmented"].sum() == 2


# ------------------------------------------------------------------ findings follow the numbers

def test_origin_finding_follows_the_shares():
    def finding(shares):
        return bias._origin_finding(pd.DataFrame({"share_of_illustrations": shares}))
    assert "similar proportions" in finding({"african_storybook": 0.55, "storyweaver": 0.45})
    skewed = finding({"african_storybook": 0.99, "storyweaver": 0.01})
    assert "skew toward African" in skewed and "No re-weighting" not in skewed
    assert "one tradition only" in finding({"storyweaver": 1.0})


def test_drawing_findings_name_the_largest_country_and_skip_a_gap_that_isnt_there():
    d = {"shares_before": pd.Series({"GB": 0.6, "IN": 0.4}),
         "recognition": {"gap": 0.0, "best": ("GB", 0.9), "worst": ("IN", 0.9)}}
    text = "\n".join(bias._drawing_findings({"max_country_share": 0.2}, d))
    assert "GB alone is 60.0%" in text and "US" not in text
    assert "Recognition is even" in text and "Performance gap" not in text
    d["recognition"] = {"gap": 0.06, "best": ("BR", 0.955), "worst": ("TH", 0.895)}
    d["shares_before"] = pd.Series({"US": 0.15, "GB": 0.1})
    text = "\n".join(bias._drawing_findings({"max_country_share": 0.2}, d))
    assert "no country is above" in text and "Performance gap" in text and "TH (89.5%)" in text


def test_readability_findings_only_claim_what_the_signs_show():
    table = pd.DataFrame({"excerpts": [500], "mae": [0.7], "mean_error": [0.0]}, index=["2000+"])
    b = {"era": table, "category": table.rename(index={"2000+": "Lit"}), "era_mae_gap": None,
         "category_mae_gap": None}
    text = "\n".join(bias._readability_findings(b))
    assert "easier" not in text and "harder" not in text and "random split" not in text
    era = pd.DataFrame({"excerpts": [300, 300, 9], "mae": [0.6, 0.8, 1.5], "mean_error": [0.2, -0.1, 2.0]},
                       index=["1900–1949", "2000+", "unknown"])
    b = {"era": era, "category": table.rename(index={"2000+": "Lit"}), "era_mae_gap": 0.2,
         "category_mae_gap": None}
    text = "\n".join(bias._readability_findings(b))
    assert "**1900–1949** texts easier" in text and "**2000+** texts harder" in text
    assert "unknown** texts" not in text and "Too small to judge" in text


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
    assert "US alone is 80.0%" in md and "US share of the training signal: 80.0% → 50.0%" in md
    assert "equal in every source" in md
    assert "golden evaluation set" not in md and "neighbouring era" not in md   # nothing unimplemented
    json.dumps(bias.report_to_json(report), default=str)    # serialisable


def test_bias_report_command_reads_only_and_writes_to_out(tmp_path):
    from primer.data import cli
    from primer.data.storage import Store

    data, out = tmp_path / "data", tmp_path / "out"
    _write_datasets(Store(data.as_posix()))

    def snapshot():
        return {(p.relative_to(data).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                for p in data.rglob("*") if p.is_file()}

    before = snapshot()
    assert cli.main(["--data-uri", data.as_posix(), "bias-report", "--out", out.as_posix(),
                     "--max-country-share", "0.5"]) == 0
    assert snapshot() == before                                   # the data root is untouched
    assert sorted(p.name for p in out.iterdir()) == ["bias_report.json", "bias_report.md"]
    assert json.loads((out / "bias_report.json").read_text(encoding="utf-8"))["max_country_share"] == 0.5


def test_bias_report_command_refuses_a_cap_that_cant_be_met(tmp_path):
    from primer.data import cli
    from primer.data.storage import Store

    data, out = tmp_path / "data", tmp_path / "out"
    _write_datasets(Store(data.as_posix()))
    assert cli.main(["--data-uri", data.as_posix(), "bias-report", "--out", out.as_posix(),
                     "--max-country-share", "0.05"]) == 2
    assert not out.exists()                                       # no half-written "nan%" report


def test_excerpts_without_a_year_go_in_an_unknown_era():
    """Regression: missing pub_year became a missing era, which Fairlearn's MetricFrame rejects."""
    assert bias.era(pd.Series([1880.0, None, 2010.0])).tolist() == ["1850–1899", "unknown", "2000+"]
    clear = pd.DataFrame({"score": [-1.0, -0.5, 0.0, 0.5], "flesch_kincaid_grade": [10, 8, 6, 4],
                          "category": ["Lit", "Info", "Lit", "Info"], "pub_year": [1880, None, 1990, 2010]})
    assert "unknown" in bias.readability_baseline_by_slice(clear)["era"].index
