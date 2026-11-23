"""The drawing recogniser's bias report: per-slice accuracy, disparities named, a mitigation (S6g).

    python -m primer.drawing.evaluate --model M --version v1.1 --split test --out m.json \\
        --per-drawing rows.csv.gz
    python -m primer.drawing.bias_report --metrics m.json --rows rows.csv.gz \\
        --out bias.json --markdown bias.md

Step 3 of `docs/engineering/model-development.md` for the drawing stage: per-slice metrics over its
two slice keys, ``country_code`` and category (the label), with every disparity either named or
ruled out by the numbers, and a mitigation proposed or ruled out for `CARD.md`.

**What makes a disparity "named" here, and not noise.** With 54 judged countries, many near the
100-drawing floor, best minus worst mostly measures sample size. So only countries with at least
`SMALL_SLICE` drawings are judged, each is tested, and a judged country is named when:

- **Its label mix is controlled for.** Countries draw different things, and labels differ in
  difficulty by a factor of ten. A country's *expected* accuracy is the mean, over its own
  drawings, of each label's accuracy **in the other countries** (leave-country-out, so the US,
  with 45% of the split, is not compared with itself). Observed minus expected is the disparity
  that label mix does not explain.
- **A two-sided z-test on that difference, Holm-corrected** across the judged countries, gives
  p < ``ALPHA``.

Reported beside the verdict, and not part of it:

- A Wilson 95% interval on each slice's own top-5 accuracy.
- A 95% interval on the difference itself (``difference_low`` / ``difference_high``), from the
  same variance as the test and **not** adjusted for the number of countries. It is the yardstick
  for "moved against it beyond its interval" when a candidate is compared with this baseline.
- The unadjusted test (against the other countries' overall rate), so a reader can see how much of
  a raw gap the label mix accounts for.

Limits: drawings are treated as independent. Quick, Draw! rows carry no drawer id, so a player who
drew many items, or a country-by-label style effect, would make the p-values too small and this
cannot be adjusted for. "Named" is statistical, not practical: at the US's size a gap of a couple of
points is named, so read ``difference`` and its interval, not only the verdict.

The rows are checked against the metrics file before anything is computed. They must reproduce
its drawing count, its ``drawing_top5_acc`` and every slice exactly; otherwise they are not the run
the file describes. A ``--limit`` probe is refused, because it is a timing check and never a
result.

Fairlearn's ``MetricFrame`` computes the per-slice rates, as `primer.data.bias` does for the data's
own recogniser.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate

from primer.drawing.evaluate import TOP_K
from primer.drawing.slices import SMALL_SLICE

#: The two-sided z for a 95% interval.
Z95 = 1.959963984540054
#: Family-wise error rate for naming a country, after Holm.
ALPHA = 0.05
#: How many of the hardest and easiest categories the report lists.
CATEGORY_LIST = 10
ROW_COLUMNS = ("record_id", "label_index", "label", "country_code", "rank", "hit")


# ------------------------------------------------------------------------------------- statistics


def wilson(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for ``hits`` of ``n``. Stays inside [0, 1] at small ``n``."""
    if n < 1:
        raise ValueError("an interval needs at least one drawing")
    if not 0 <= hits <= n:
        raise ValueError(f"{hits} hits of {n} drawings is impossible")
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def two_sided_p(z: float) -> float:
    """The two-sided normal tail probability of ``z``."""
    return math.erfc(abs(z) / math.sqrt(2))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values, which control the family-wise error rate without assuming independence."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


# ------------------------------------------------------------------------------------ reconciling


def check_rows(rows: pd.DataFrame, metrics: Mapping[str, Any]) -> None:
    """Refuse rows that are not the run ``metrics`` describes, or that came from a probe."""
    missing = [c for c in ROW_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"the rows file lacks {missing}; write it with evaluate --per-drawing")
    if metrics.get("stage") != "drawing":
        raise ValueError(f"the metrics file is a {metrics.get('stage')!r} file, not a drawing one")
    if metrics.get("limit") is not None:
        raise ValueError("the metrics file is a --limit probe: a timing check, never a result")
    if rows["record_id"].duplicated().any():
        raise ValueError("a drawing appears twice in the rows file")
    if (rows["hit"] != (rows["rank"] < TOP_K)).any():
        raise ValueError("a row's hit disagrees with its rank")
    # The categories key on `label` and the country tests on `label_index`: if the two disagree,
    # the report would describe one set of categories and test another.
    if (
        rows.groupby("label_index")["label"].nunique().gt(1).any()
        or rows.groupby("label")["label_index"].nunique().gt(1).any()
    ):
        raise ValueError("label and label_index are not one-to-one in the rows")
    n = int(metrics["overall"]["drawings"])
    if len(rows) != n:
        raise ValueError(f"the rows file has {len(rows)} drawings; the metrics file scored {n}")
    accuracy = int(rows["hit"].sum()) / len(rows)
    if accuracy != metrics["overall"]["drawing_top5_acc"]:
        raise ValueError(
            f"the rows give drawing_top5_acc {accuracy!r}; the metrics file says "
            f"{metrics['overall']['drawing_top5_acc']!r}: they are not the same run"
        )
    by_country = rows.groupby("country_code")["hit"].agg(["sum", "count"])
    entries = [entry for entry in metrics["slices"] if entry["key"] == "country_code"]
    # Both directions: every slice the file lists must be in the rows (below), and every country the
    # rows carry must be a slice the file lists, or a country could hide behind the totals.
    if {str(entry["value"]) for entry in entries} != set(by_country.index.astype(str)):
        raise ValueError("the rows' countries are not the metrics file's country slices")
    for entry in entries:
        got = by_country.loc[entry["value"]] if entry["value"] in by_country.index else None
        if (
            got is None
            or int(got["count"]) != entry["n"]
            or (int(got["sum"]) / int(got["count"]) != entry["metrics"]["drawing_top5_acc"])
        ):
            raise ValueError(f"country {entry['value']!r} does not match the metrics file's slice")


# ----------------------------------------------------------------------------------------- slices


def rates_by(rows: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-group drawings, top-5 and top-1 accuracy (Fairlearn), with a Wilson interval on top-5."""
    hits = rows["hit"].astype(int).to_numpy()
    top1 = (rows["rank"] == 0).astype(int).to_numpy()
    frame = MetricFrame(
        metrics={"drawings": count, "top5": selection_rate},
        y_true=hits,
        y_pred=hits,
        sensitive_features=rows[key].astype(str).to_numpy(),
    ).by_group
    frame["top1"] = MetricFrame(
        metrics=selection_rate, y_true=top1, y_pred=top1, sensitive_features=rows[key].astype(str).to_numpy()
    ).by_group
    frame["drawings"] = frame["drawings"].astype(int)
    frame["hits"] = (frame["top5"] * frame["drawings"]).round().astype(int)
    bounds = [wilson(int(h), int(n)) for h, n in zip(frame["hits"], frame["drawings"], strict=True)]
    frame["top5_low"] = [low for low, _ in bounds]
    frame["top5_high"] = [high for _, high in bounds]
    frame["judged"] = frame["drawings"] >= SMALL_SLICE
    return frame.sort_values("drawings", ascending=False)


def _mix_variance(drawn: pd.Series, elsewhere: pd.Series, rate: pd.Series, n: int) -> float:
    """Variance of observed minus the label-mix expectation, for a country of ``n`` compared drawings.

    Per label: ``drawn`` times here, ``elsewhere`` drawings in the other countries, ``rate`` their
    accuracy. The country's own hits contribute k p(1-p); estimating p from ``elsewhere`` drawings
    contributes k² p(1-p)/m.
    """
    spread = rate * (1 - rate)
    return float((drawn * spread).sum() + (drawn * drawn * spread / elsewhere).sum()) / (n * n)


def country_tests(rows: pd.DataFrame, judged: Sequence[str]) -> pd.DataFrame:
    """For each judged country: observed vs the label-mix expectation, and vs the rest, Holm-corrected."""
    if not judged:
        raise ValueError(f"no country has {SMALL_SLICE} drawings, so there is nothing to judge")
    results = []
    for country in judged:
        own = rows[rows["country_code"] == country]
        rest = rows[rows["country_code"] != country]
        per_label = rest.groupby("label_index")["hit"].agg(["sum", "count"])
        rate = (per_label["sum"] / per_label["count"]).rename("p")
        matched = own.join(rate, on="label_index")
        # A label no other country drew has no outside accuracy to compare with; those drawings
        # are counted and left out rather than given a made-up expectation.
        usable = matched.dropna(subset=["p"])
        n = len(usable)
        if n == 0:
            raise ValueError(
                f"country {country!r} cannot be tested: no label it drew was drawn in any other country"
            )
        observed = float(usable["hit"].mean())
        expected = float(usable["p"].mean())
        # Two sources of variance. The country's own hits vary around their expectations. And the
        # expectation is itself estimated, from the other countries' drawings of each label: label
        # l, drawn k times here and estimated from m drawings elsewhere, adds k² p(1-p)/m. Leaving
        # that term out understates the error most for the largest country, whose "elsewhere" is
        # comparatively small (about √2 for the US on v1.1's test split).
        k = usable.groupby("label_index").size()
        outside = per_label.loc[k.index]

        variance = _mix_variance(k, outside["count"], outside["sum"] / outside["count"], n)
        if variance == 0:
            # Every outside rate is exactly 0 or 1, so the plug-in variance is zero and any gap
            # would read as "no evidence" (p=1). Shrink the rates by half a drawing: identical to
            # the plug-in wherever it is non-zero, and an honest tiny variance where it is not.
            shrunk_rate = (outside["sum"] + 0.5) / (outside["count"] + 1)
            variance = _mix_variance(k, outside["count"], shrunk_rate, n)
        z_mix = (observed - expected) / math.sqrt(variance)
        half_width = Z95 * math.sqrt(variance)
        rest_rate = float(rest["hit"].mean())
        raw_variance = rest_rate * (1 - rest_rate) * (1 / n + 1 / len(rest))
        if raw_variance == 0:
            shrunk = (float(rest["hit"].sum()) + 0.5) / (len(rest) + 1)
            raw_variance = shrunk * (1 - shrunk) * (1 / n + 1 / len(rest))
        z_raw = (observed - rest_rate) / math.sqrt(raw_variance)
        results.append(
            {
                "country_code": country,
                "drawings": len(own),
                "compared": n,
                "no_outside_label": len(own) - n,
                "observed": observed,
                "expected_from_label_mix": expected,
                "difference": observed - expected,
                "difference_low": observed - expected - half_width,
                "difference_high": observed - expected + half_width,
                "p_label_mix": two_sided_p(z_mix),
                "rest_rate": rest_rate,
                "raw_difference": observed - rest_rate,
                "p_raw": two_sided_p(z_raw),
            }
        )
    frame = pd.DataFrame(results).set_index("country_code")
    frame["p_label_mix_holm"] = pd.Series(holm(frame["p_label_mix"].to_dict()))
    frame["p_raw_holm"] = pd.Series(holm(frame["p_raw"].to_dict()))
    frame["named"] = frame["p_label_mix_holm"] < ALPHA
    return frame.sort_values("difference")


# ----------------------------------------------------------------------------------------- report


def build_report(rows: pd.DataFrame, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the report says, derived from the rows after they reconcile with ``metrics``."""
    check_rows(rows, metrics)
    countries = rates_by(rows, "country_code")
    judged = [str(c) for c in countries.index[countries["judged"]]]
    tests = country_tests(rows, judged)
    categories = rates_by(rows, "label")
    judged_categories = categories[categories["judged"]].sort_values("top5")
    named = tests[tests["named"]]
    raw_named = tests[tests["p_raw_holm"] < ALPHA]
    return {
        "provenance": {
            key: metrics.get(key)
            for key in ("data_version", "split", "encoder", "encoder_revision", "device", "dtype", "git_sha")
        }
        | {"prompt_template": metrics.get("scoring", {}).get("prompt_template")},
        "overall": {
            "drawings": len(rows),
            "top5": float(rows["hit"].mean()),
            "top1": float((rows["rank"] == 0).mean()),
        },
        "country": {
            "slices": len(countries),
            "judged": len(judged),
            "judged_drawings": int(countries.loc[judged, "drawings"].sum()),
            "by_group": countries,
            "tests": tests,
            "named": [str(c) for c in named.index],
            "raw_named": [str(c) for c in raw_named.index],
        },
        "category": {
            "slices": len(categories),
            "judged": int(categories["judged"].sum()),
            "by_group": categories,
            "hardest": judged_categories.head(CATEGORY_LIST),
            "easiest": judged_categories.tail(CATEGORY_LIST).iloc[::-1],
            "spread": (float(judged_categories["top5"].min()), float(judged_categories["top5"].max())),
        },
    }


def findings(report: Mapping[str, Any]) -> list[str]:
    """The verdicts, each derived from the numbers, plus the mitigation proposed or ruled out."""
    country = report["country"]
    tests = country["tests"]
    out = []
    if country["named"]:

        def listed(codes: Sequence[str]) -> str:
            return (
                ", ".join(
                    f"{c} ({tests.loc[c, 'difference']:+.3f}, Holm p={tests.loc[c, 'p_label_mix_holm']:.3g})"
                    for c in codes
                )
                or "none"
            )

        below = [c for c in country["named"] if tests.loc[c, "difference"] < 0]
        above = [c for c in country["named"] if tests.loc[c, "difference"] > 0]
        sizes = country["by_group"]["drawings"]
        largest = str(sizes.idxmax())
        share = sizes.max() / sizes.sum()
        out.append(
            f"**Country disparity named** after controlling for label mix, at a family-wise {ALPHA:.0%} "
            f"(Holm, {country['judged']} judged countries). Below what their label mix predicts: {listed(below)}. "
            f"Above it: {listed(above)}. "
            "*Mitigation, proposed for S6d, not applied here* (a zero-shot model has nothing to reweight): "
            "cap any one country's share of the training sample (`capped_weights` in `primer.data.bias`), so a "
            f"fine-tune on a split where {largest} is {share:.0%} of the drawings cannot buy {largest}'s "
            "accuracy with the other countries'; then re-run this report on the candidate and reject it if a "
            "named country's difference falls below its baseline value by more than the half-width of the "
            "baseline's 95% interval on that difference (`difference_low` to `difference_high`, not adjusted "
            "for the number of countries)."
        )
    else:
        out.append(
            f"**No country disparity survives** controlling for label mix at a family-wise {ALPHA:.0%} "
            f"(Holm, {country['judged']} judged countries). "
            f"Without that control, {len(country['raw_named'])} would have been named "
            f"({', '.join(country['raw_named']) or 'none'}). "
            "*Mitigation ruled out* for country at this sample size, which is not the same as no gap: "
            "a gap smaller than these intervals could remain. S6d re-runs this report on the "
            "fine-tuned model, since fine-tuning on a US-heavy split could open a gap the baseline lacks."
        )
    left_out = int(tests["no_outside_label"].sum())
    if left_out:
        out.append(
            f"**{left_out} drawings are left out of the label-mix comparison**, in "
            f"{int((tests['no_outside_label'] > 0).sum())} countries: they carry a label no other country "
            "drew, so there is no outside accuracy to compare with. Each country's top-5 above counts them; "
            "its difference does not."
        )
    low, high = report["category"]["spread"]
    hardest = report["category"]["hardest"]
    out.append(
        f"**Category accuracy spans {low:.2f} to {high:.2f}** across {report['category']['judged']} judged "
        f"categories. That is per-class difficulty, not a fairness gap, but it is where the model fails: the "
        f"hardest are {', '.join(f'{label} ({rate:.2f})' for label, rate in hardest['top5'].head(5).items())}. "
        "*Mitigation, proposed for S6d:* report per-category accuracy for every candidate, and do not accept a "
        "fine-tune whose gain comes only from categories already above the median."
    )
    return out


def render_markdown(report: Mapping[str, Any]) -> str:
    """A readable report: provenance, findings, the country tests and the category extremes."""
    prov = report["provenance"]
    overall = report["overall"]
    country = report["country"]
    summary = (
        f"{overall['drawings']:,} `{prov['split']}` drawings of `{prov['data_version']}`, "
        f"`{prov['encoder']}` @ `{str(prov['encoder_revision'])[:8]}` on {prov['device']} ({prov['dtype']}), "
        f"prompt `{prov['prompt_template']}`, commit `{prov['git_sha']}`. "
        f"Top-5 {overall['top5']:.4f}, top-1 {overall['top1']:.4f}."
    )
    heading = (
        f"## Country ({country['judged']} of {country['slices']} judged, n ≥ {SMALL_SLICE}; "
        f"{country['judged_drawings']:,} drawings)"
    )
    legend = (
        "Sorted by the label-mix-adjusted difference. `p (Holm)` is for that difference; `raw p (Holm)` is "
        "against the other countries' overall rate."
    )
    lines = [
        "# Drawing recogniser: bias report (S6g)",
        "",
        summary,
        "",
        "## Findings",
        "",
        *[f"- {line}" for line in findings(report)],
        "",
        heading,
        "",
        legend,
        "",
        "| Country | n | top-5 [95% CI] | expected from label mix | difference [95% CI] | p (Holm) | raw p (Holm) |",
        "|---|---|---|---|---|---|---|",
    ]
    by_group = country["by_group"]
    for code, row in country["tests"].iterrows():
        g = by_group.loc[code]
        lines.append(
            f"| {code} | {int(row['drawings'])} | {g['top5']:.3f} [{g['top5_low']:.3f}, {g['top5_high']:.3f}] | "
            f"{row['expected_from_label_mix']:.3f} | {row['difference']:+.3f} "
            f"[{row['difference_low']:+.3f}, {row['difference_high']:+.3f}] | {row['p_label_mix_holm']:.3g} | "
            f"{row['p_raw_holm']:.3g} |"
        )
    category = report["category"]
    lines += [
        "",
        f"## Category ({category['judged']} of {category['slices']} judged)",
        "",
        "| | Category | n | top-5 [95% CI] | top-1 |",
        "|---|---|---|---|---|",
    ]
    for name, frame in (("hardest", category["hardest"]), ("easiest", category["easiest"])):
        for label, row in frame.iterrows():
            lines.append(
                f"| {name} | {label} | {int(row['drawings'])} | {row['top5']:.3f} "
                f"[{row['top5_low']:.3f}, {row['top5_high']:.3f}] | {row['top1']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def report_to_json(report: Mapping[str, Any]) -> dict[str, Any]:
    """The report with its tables as records, for the JSON file and later for MLflow."""

    def table(frame: pd.DataFrame, index: str) -> list[dict[str, Any]]:
        return json.loads(frame.rename_axis(index).reset_index().to_json(orient="records"))

    country, category = report["country"], report["category"]
    return {
        "provenance": report["provenance"],
        "overall": report["overall"],
        "alpha": ALPHA,
        "small_slice": SMALL_SLICE,
        "country": {
            "slices": country["slices"],
            "judged": country["judged"],
            "named": country["named"],
            "raw_named": country["raw_named"],
            "by_group": table(country["by_group"], "country_code"),
            "tests": table(country["tests"], "country_code"),
        },
        "category": {
            "slices": category["slices"],
            "judged": category["judged"],
            "spread": list(category["spread"]),
            "by_group": table(category["by_group"], "label"),
        },
        "findings": findings(report),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bias_report", description="the drawing recogniser's bias report (S6g)")
    p.add_argument("--metrics", required=True, type=Path, help="the metrics file evaluate wrote")
    p.add_argument("--rows", required=True, type=Path, help="the rows evaluate --per-drawing wrote")
    p.add_argument("--out", required=True, type=Path, help="the report as JSON")
    p.add_argument("--markdown", type=Path, default=None, help="also write the report as Markdown")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    try:
        # Explicitly gzip, whatever the name: `evaluate --per-drawing` always writes gzip, and
        # pandas would otherwise go by the extension and fail on gzip bytes in a `.csv`.
        rows = pd.read_csv(
            args.rows, compression="gzip", dtype={"country_code": str, "record_id": str}, keep_default_na=False
        )
    except (gzip.BadGzipFile, UnicodeDecodeError) as error:
        raise SystemExit(f"{args.rows} is not the gzipped CSV that evaluate --per-drawing writes: {error}") from error
    report = build_report(rows, metrics)
    args.out.write_text(json.dumps(report_to_json(report), indent=2) + "\n", encoding="utf-8")
    text = render_markdown(report)
    if args.markdown is not None:
        args.markdown.write_text(text, encoding="utf-8")
    # The report holds characters (≥) that a Windows console's cp1252 cannot print; both files are
    # already written, so a console that cannot show it must not turn a good run into exit 1.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
