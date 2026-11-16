"""S6g's bias report in `bias_report.py`, on hand-built rows whose answers are known.

No weights and no bucket. Rows are built so that each statistical claim is checkable by hand: a
country whose raw gap is all label mix must not be named, and one with the same mix but worse
accuracy must be.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from primer.drawing import bias_report
from primer.drawing.bias_report import build_report, check_rows, holm, wilson


def make_rows(groups):
    """``groups``: (country, label_index, label, n, hits). Hits come first, then misses."""
    records = []
    for country, label_index, label, n, hits in groups:
        for i in range(n):
            rank = 0 if i < hits else 9
            records.append(
                {
                    "record_id": f"{country}-{label}-{i}",
                    "label_index": label_index,
                    "label": label,
                    "country_code": country,
                    "rank": rank,
                    "hit": rank < 5,
                }
            )
    return pd.DataFrame(records)


def metrics_for(rows, **overrides):
    """A metrics file that the rows reconcile with, as `evaluate` would have written it."""
    slices = []
    for country, group in rows.groupby("country_code"):
        hits = group["hit"].tolist()
        slices.append(
            {
                "key": "country_code",
                "value": country,
                "n": len(hits),
                "judged": len(hits) >= 100,
                "metrics": {"drawing_top5_acc": sum(hits) / len(hits)},
            }
        )
    hits = rows["hit"].tolist()
    metrics = {
        "stage": "drawing",
        "limit": None,
        "split": "test",
        "data_version": "v1.1",
        "encoder": "enc",
        "encoder_revision": "abcdef123",
        "device": "cuda",
        "dtype": "bfloat16",
        "git_sha": "abc",
        "scoring": {"prompt_template": "a child's drawing of a {label}."},
        "overall": {"drawing_top5_acc": sum(hits) / len(hits), "drawings": len(hits)},
        "slices": slices,
    }
    metrics.update(overrides)
    return metrics


@pytest.fixture
def world():
    """X and Y are the reference, scoring 0.9 on the easy label and 0.1 on the hard one. A draws only
    the easy label and scores 0.9 on it: a raw gap that is all label mix. B draws their mix but
    misses every hard drawing: a real gap. Two references, so B's deficit cannot by itself make the
    others look better than their own peers."""
    rows = make_rows(
        [
            ("X", 0, "easy", 2000, 1800),
            ("X", 1, "hard", 2000, 200),
            ("Y", 0, "easy", 2000, 1800),
            ("Y", 1, "hard", 2000, 200),
            ("A", 0, "easy", 400, 360),
            ("B", 0, "easy", 300, 270),
            ("B", 1, "hard", 300, 0),
            ("S", 0, "easy", 20, 20),  # under the 100-drawing floor: reported, never judged
        ]
    )
    return rows, metrics_for(rows)


# ------------------------------------------------------------------------------------- statistics


def test_wilson_matches_the_known_interval_and_stays_in_range():
    low, high = wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson(5, 10)
    assert low == pytest.approx(1 - high)
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_holm_scales_by_rank_and_keeps_order():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


# ----------------------------------------------------------------------------------- reconciling


def test_rows_that_reconcile_are_accepted(world):
    rows, metrics = world
    check_rows(rows, metrics)


@pytest.mark.parametrize(
    "spoil, message",
    [
        (lambda r, m: m.update(limit=300), "probe"),
        (lambda r, m: m["overall"].update(drawings=1), "drawings"),
        (lambda r, m: m["overall"].update(drawing_top5_acc=0.5), "not the same run"),
        (lambda r, m: m["slices"][0]["metrics"].update(drawing_top5_acc=0.0), "does not match"),
        (lambda r, m: m.update(stage="speech"), "not a drawing"),
    ],
    ids=["probe", "count", "accuracy", "slice", "stage"],
)
def test_rows_from_another_run_are_refused(world, spoil, message):
    rows, metrics = world
    spoil(rows, metrics)
    with pytest.raises(ValueError, match=message):
        check_rows(rows, metrics)


def test_a_row_whose_hit_disagrees_with_its_rank_is_refused(world):
    rows, metrics = world
    rows.loc[0, "rank"] = 7  # still marked a hit
    with pytest.raises(ValueError, match="disagrees"):
        check_rows(rows, metrics)


# ---------------------------------------------------------------------------------------- report


def test_a_gap_that_is_only_label_mix_is_not_named_and_a_real_one_is(world):
    rows, metrics = world

    report = build_report(rows, metrics)
    tests = report["country"]["tests"]

    # A: 0.90 against everyone else's ~0.5, but ~0.90 is what its easy-only mix predicts.
    assert "A" in report["country"]["raw_named"], "the raw test sees A's gap"
    assert "A" not in report["country"]["named"], "label mix explains it"
    assert tests.loc["A", "expected_from_label_mix"] == pytest.approx(3890 / 4320)
    # B: the references' mix, but 0.0 on the hard label where they score 0.1.
    assert report["country"]["named"] == ["B"]
    assert tests.loc["B", "difference"] < 0


def test_a_country_under_the_floor_is_reported_but_not_tested(world):
    rows, metrics = world

    report = build_report(rows, metrics)

    assert not bool(report["country"]["by_group"].loc["S", "judged"])
    assert "S" not in report["country"]["tests"].index
    assert report["country"]["judged"] == 4


def test_a_country_is_not_compared_with_itself(world):
    # X is 45% of the rows, as the US is of the test split. Its expectation must come from the
    # other countries only, or a large country would be measured mostly against itself.
    rows, metrics = world

    tests = build_report(rows, metrics)["country"]["tests"]

    easy_elsewhere = (1800 + 360 + 270 + 20) / (2000 + 400 + 300 + 20)
    hard_elsewhere = (200 + 0) / (2000 + 300)
    assert tests.loc["X", "expected_from_label_mix"] == pytest.approx((easy_elsewhere + hard_elsewhere) / 2)


def test_findings_say_which_way_the_verdict_went(world):
    rows, metrics = world
    named = bias_report.findings(build_report(rows, metrics))
    assert "Country disparity named" in named[0] and "B (" in named[0]

    even = make_rows([("X", 0, "easy", 1000, 900), ("Y", 0, "easy", 1000, 900)])
    verdict = bias_report.findings(build_report(even, metrics_for(even)))
    assert "No country disparity survives" in verdict[0] and "ruled out" in verdict[0]


def test_categories_are_ranked_hardest_first(world):
    rows, metrics = world

    category = build_report(rows, metrics)["category"]

    assert list(category["hardest"].index[:1]) == ["hard"]
    assert category["spread"][0] < category["spread"][1]


# ------------------------------------------------------------------------------------------- CLI


def test_the_cli_keeps_a_country_called_na(tmp_path):
    # Namibia's code is "NA", which pandas reads as missing by default and would drop.
    rows = make_rows([("NA", 0, "easy", 150, 120), ("US", 0, "easy", 150, 90)])
    rows_path = tmp_path / "rows.csv.gz"
    rows.to_csv(rows_path, index=False, compression="gzip")
    metrics_path = tmp_path / "m.json"
    metrics_path.write_text(json.dumps(metrics_for(rows)), encoding="utf-8")
    out = tmp_path / "bias.json"
    markdown = tmp_path / "bias.md"

    code = bias_report.main(
        [
            "--metrics",
            str(metrics_path),
            "--rows",
            str(rows_path),
            "--out",
            str(out),
            "--markdown",
            str(markdown),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert {g["country_code"] for g in report["country"]["by_group"]} == {"NA", "US"}
    assert "| NA |" in markdown.read_text(encoding="utf-8")
    assert report["provenance"]["prompt_template"] == "a child's drawing of a {label}."


def test_the_expectation_s_own_uncertainty_is_in_the_test():
    # X scores 0.6 on a label that the only other country, Y, scored 0.5 on from just 20 drawings.
    # By hand: own part 200 x 0.25 = 50; estimate part 200^2 x 0.25 / 20 = 500; sd = sqrt(550)/200,
    # z = 0.1 / 0.1173 = 0.853, p = 0.394. Without the estimate's part, z would be 2.83, p 0.0047:
    # a country named on a comparison with twenty drawings.
    rows = make_rows([("X", 0, "easy", 200, 120), ("Y", 0, "easy", 20, 10)])

    tests = build_report(rows, metrics_for(rows))["country"]["tests"]

    assert tests.loc["X", "p_label_mix"] == pytest.approx(0.3938, abs=1e-3)
    assert not tests.loc["X", "named"]
