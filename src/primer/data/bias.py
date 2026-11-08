"""Data bias analysis by slicing, and the mitigations applied before training.

The training data has no information about people, so bias is measured on the
slices it does have:

- **Drawings (Quick, Draw!)** by country: who drew them, and how often the game's
  own recognizer understood each country's drawings. That second number is a real
  model's performance per slice (Fairlearn ``MetricFrame``), and the drawing
  recognizer will learn from the same data.
- **Stories** by source: the gender of each story's lead characters, from gendered
  words and common names.
- **Readability (CLEAR)** by genre and publication era: teacher-rated difficulty,
  and how wrong the Flesch-Kincaid baseline is for each slice (``MetricFrame``).
- **Picture books** by origin (African Storybook vs StoryWeaver, India).

Mitigations: capped sample weights for countries, gender-swap augmentation for
stories, and pooled genre × era split keys for readability. ``build_report`` computes
all of it and ``render_markdown`` writes the report (see ``primer-data bias-report``);
every verdict in the report is derived from the numbers. Applying the mitigations to
the training data is the training loaders' job (KAN-80).
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate
from sklearn.metrics import mean_absolute_error

from primer.data.catalog import read_dataset
from primer.data.storage import Store

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ gender terms

MALE_TO_FEMALE = {
    "he": "she", "him": "her", "his": "her", "himself": "herself", "boy": "girl", "boys": "girls",
    "man": "woman", "men": "women", "king": "queen", "kings": "queens", "prince": "princess",
    "father": "mother", "dad": "mom", "daddy": "mommy", "brother": "sister", "brothers": "sisters",
    "uncle": "aunt", "grandpa": "grandma", "grandfather": "grandmother", "son": "daughter",
    "husband": "wife", "mr": "mrs", "sir": "madam", "gentleman": "lady",
}
# "her" and "his" are handled separately: each maps to two words depending on what follows.
FEMALE_TO_MALE = {v: k for k, v in MALE_TO_FEMALE.items() if k != "his"} | {"hers": "his", "ms": "mr", "miss": "mr"}
MALE_NAMES = {"Tom": "Lily", "Ben": "Anna", "Timmy": "Mia", "Tim": "Sue", "Max": "Lucy", "Jack": "Sara",
              "John": "Amy", "Bob": "Jill", "Joe": "Emma", "Billy": "Molly", "Jake": "Zoe", "Leo": "Ella"}
FEMALE_NAMES = {v: k for k, v in MALE_NAMES.items()}
MALE_WORDS = set(MALE_TO_FEMALE)
FEMALE_WORDS = set(FEMALE_TO_MALE) | {"her"}
# Letters only, so "Tom's" is "Tom" + "'s" and "He's" is "He" + "'s": the name or pronoun still
# swaps and counts. Names are matched case-sensitively ("Lily", not the flower "lily").
_WORD = re.compile(r"[A-Za-z]+")
# After "her", these words mean it is an object ("saw her and ...") → "him"; otherwise a possessive → "his".
_HER_OBJECT_FOLLOWERS = {"and", "or", "but", "to", "with", "for", "a", "an", "the", "that", "up", "down",
                         "back", "again", "too", "so", "very", "in", "on", "at", "of", "from", "as", "if"}


def gender_counts(text: str) -> tuple[int, int]:
    words = _WORD.findall(text)
    male = sum(w.lower() in MALE_WORDS or w in MALE_NAMES for w in words)
    female = sum(w.lower() in FEMALE_WORDS or w in FEMALE_NAMES for w in words)
    return male, female


def lead_gender(text: str) -> str:
    """"male", "female" or "neutral" (no gendered words, or exactly as many of each)."""
    male, female = gender_counts(text)
    return "male" if male > female else "female" if female > male else "neutral"


def _match_case(word: str, template: str) -> str:
    if template.isupper() and len(template) > 1:
        return word.upper()
    return word.capitalize() if template[:1].isupper() else word


def gender_swap(text: str) -> str:
    """Swap gendered words and common names (he↔she, Tom↔Lily, ...), keeping capitalisation.

    "her" is ambiguous (him/his) and so is "his" (her/hers): before a word that looks like a noun
    phrase they are possessive determiners ("his"/"her"), otherwise "him"/"hers". A heuristic, good
    enough for augmentation but not for publishing ("made her happy" still becomes "made his happy").
    """
    tokens = re.split(r"([A-Za-z]+)", text)
    out = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        gap = tokens[i + 1] if i + 1 < len(tokens) else ""
        nxt = tokens[i + 2] if i + 2 < len(tokens) else ""
        before_noun = gap == " " and bool(_WORD.fullmatch(nxt)) and nxt.lower() not in _HER_OBJECT_FOLLOWERS
        if tok in MALE_NAMES:
            out.append(MALE_NAMES[tok])
        elif tok in FEMALE_NAMES:
            out.append(FEMALE_NAMES[tok])
        elif low == "her":
            out.append(_match_case("his" if before_noun else "him", tok))
        elif low == "his":
            out.append(_match_case("her" if before_noun else "hers", tok))
        elif low in MALE_TO_FEMALE:
            out.append(_match_case(MALE_TO_FEMALE[low], tok))
        elif low in FEMALE_TO_MALE:
            out.append(_match_case(FEMALE_TO_MALE[low], tok))
        else:
            out.append(tok)
    return "".join(out)


# ------------------------------------------------------------------ slicing helpers

def representation(values: pd.Series, top: int = 10) -> pd.DataFrame:
    """Count and share per group, the ``top`` largest groups plus "Other"."""
    vc = values.fillna("unknown").value_counts()
    shown = vc.head(top)
    if len(vc) > top:
        shown = pd.concat([shown, pd.Series({"Other": vc.iloc[top:].sum()})])
    return pd.DataFrame({"count": shown.astype(int), "share": (shown / vc.sum()).round(4)})


def group_top(values: pd.Series, top: int) -> pd.Series:
    keep = set(values.value_counts().head(top).index)
    return values.where(values.isin(keep), "Other").fillna("Other")


def recognition_by_country(drawings: pd.DataFrame, top: int = 15) -> dict[str, Any]:
    """The game recognizer's hit rate per country (Fairlearn MetricFrame).

    ``drawings`` needs ``country_code`` and ``recognized``. Countries outside the ``top`` largest
    are pooled as "Other" so every slice is large enough to compare; the pooled group is shown but
    doesn't decide the best, the worst or the gap.
    """
    recognized = drawings["recognized"].astype(int).to_numpy()
    countries = group_top(drawings["country_code"], top)
    mf = MetricFrame(metrics={"drawings": count, "recognition_rate": selection_rate},
                     y_true=recognized, y_pred=recognized, sensitive_features=countries.to_numpy())
    by_group = mf.by_group.sort_values("drawings", ascending=False)
    rates = by_group["recognition_rate"]
    named = rates.drop("Other", errors="ignore")
    named = named if not named.empty else rates
    return {
        "by_country": by_group,
        "top": top,
        "overall_rate": float(mf.overall["recognition_rate"]),
        "gap": float(named.max() - named.min()),
        "worst": (str(named.idxmin()), float(named.min())),
        "best": (str(named.idxmax()), float(named.max())),
    }


ERA_ORDER = ["before 1850", "1850–1899", "1900–1949", "1950–1999", "2000+", "unknown"]
SMALL_SLICE = 100  # slices with fewer rows are reported but flagged as too small to judge
# Thresholds that turn numbers into verdicts in the report, so a finding only appears when it holds.
RECOGNITION_GAP = 0.02      # best vs worst country recognition rate
MEAN_ERROR_FLAG = 0.05      # baseline mean error (teacher-score units) worth calling out
ORIGIN_TOLERANCE = 0.10     # largest picture-book origin share above an even split
ORIGIN = {"african_storybook": "African (African Storybook)", "storyweaver": "Indian (StoryWeaver)"}


def era(years: pd.Series) -> pd.Series:
    """Publication era; excerpts without a year go in "unknown" (Fairlearn rejects missing values)."""
    eras = pd.cut(years, bins=[-np.inf, 1849, 1899, 1949, 1999, np.inf],
                  labels=["before 1850", "1850–1899", "1900–1949", "1950–1999", "2000+"])
    return pd.Series(eras.astype(object), index=years.index).fillna("unknown").astype(str)


def _mean_error(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def readability_baseline_by_slice(clear: pd.DataFrame) -> dict[str, Any]:
    """How well the Flesch-Kincaid grade predicts teacher-rated ease, per genre and era.

    FK grade is mapped onto the teacher score with a straight-line fit over all excerpts; per slice
    we report the mean absolute error and the mean error (positive = the baseline says the slice is
    easier than teachers do). ``<name>_mae_gap`` compares only slices with at least ``SMALL_SLICE``
    excerpts (None if fewer than two), so a handful of excerpts can't inflate it;
    ``<name>_mae_gap_all_slices`` is the raw difference.
    """
    data = clear.dropna(subset=["score", "flesch_kincaid_grade"]).copy()
    slope, intercept = np.polyfit(data["flesch_kincaid_grade"], data["score"], 1)
    data["predicted"] = slope * data["flesch_kincaid_grade"] + intercept
    data["era"] = era(data["pub_year"])
    metrics = {"excerpts": count, "mae": mean_absolute_error, "mean_error": _mean_error}
    out: dict[str, Any] = {"fit": {"slope": float(slope), "intercept": float(intercept)},
                           "overall_mae": float(mean_absolute_error(data["score"], data["predicted"]))}
    for name, feature in (("category", data["category"].fillna("unknown")), ("era", data["era"])):
        mf = MetricFrame(metrics=metrics, y_true=data["score"].to_numpy(), y_pred=data["predicted"].to_numpy(),
                         sensitive_features=feature.to_numpy())
        table = mf.by_group.copy()
        table["teacher_score_mean"] = data.groupby(feature.to_numpy())["score"].mean()
        if name == "era":
            table = table.reindex([e for e in ERA_ORDER if e in table.index])
        out[name] = table
        judged = table[table["excerpts"] >= SMALL_SLICE]
        out[f"{name}_mae_gap"] = float(judged["mae"].max() - judged["mae"].min()) if len(judged) >= 2 else None
        out[f"{name}_mae_gap_all_slices"] = float(mf.difference()["mae"])
    return out


# ------------------------------------------------------------------ mitigations

def capped_weights(groups: pd.Series, max_share: float) -> pd.Series:
    """Per-row sample weights so no group holds more than ``max_share`` of the total weight.

    Groups under the cap keep weight 1, so no data is thrown away and small groups aren't inflated;
    only over-represented groups are down-weighted. Missing groups count as "unknown". Raises
    ``ValueError`` when the cap can't be met (shares must add up to 100%, so ``max_share`` times the
    number of groups must be at least 1).
    """
    if not 0 < max_share <= 1:
        raise ValueError(f"max_share must be in (0, 1], got {max_share}")
    groups = groups.fillna("unknown")
    counts = groups.value_counts().astype(float)
    if max_share * len(counts) < 1 - 1e-12:
        raise ValueError(f"can't cap {len(counts)} groups at {max_share:.0%} each: the shares must add up to "
                         f"100%, so the cap must be at least {1 / len(counts):.1%}")
    capped: set = set()
    while True:
        if len(capped) == len(counts) or 1 - max_share * len(capped) <= 1e-12:
            # Every group at the cap: equal shares.
            return groups.map({g: counts.sum() / (len(counts) * c) for g, c in counts.items()}).astype(float)
        free = counts.drop(list(capped)).sum()
        total = free / (1 - max_share * len(capped)) if capped else counts.sum()
        over = {g for g, c in counts.items() if g not in capped and c / total > max_share + 1e-12}
        if not over:
            break
        capped |= over
    weight = {g: (max_share * total / c if g in capped else 1.0) for g, c in counts.items()}
    return groups.map(weight).astype(float)


def effective_shares(groups: pd.Series, weights: pd.Series | None = None, top: int = 10) -> pd.Series:
    w = weights if weights is not None else pd.Series(1.0, index=groups.index)
    shares = w.groupby(groups).sum() / w.sum()
    return shares.sort_values(ascending=False).head(top).round(4)


def balance_by_swap(stories: pd.DataFrame, by: str | None = "source", seed: int = 0) -> pd.DataFrame:
    """Add gender-swapped copies of majority-lead stories until male and female leads are equal.

    Balanced within each ``by`` group (default: per source), so a large balanced source can't hide a
    skewed small one; ``by=None`` balances overall. Copies are marked ``augmented=True``; originals are
    never changed. A swap the heuristic gets wrong can leave a group short of parity; that is logged.
    """
    if by is not None and by not in stories:
        raise KeyError(f"balance_by_swap: no column {by!r} to balance by (pass by=None to balance overall)")
    groups = [stories] if by is None else [g for _, g in stories.groupby(by, sort=False, dropna=False)]
    extras = []
    for group in groups:
        leads = group["text"].map(lead_gender)
        n_male, n_female = int((leads == "male").sum()), int((leads == "female").sum())
        if n_male == n_female:
            continue
        majority = "male" if n_male > n_female else "female"
        extra = group[leads == majority].sample(n=abs(n_male - n_female), random_state=seed).copy()
        extra["text"] = extra["text"].map(gender_swap)
        extras.append(extra)
    original = stories.assign(augmented=False)
    out = pd.concat([original] + [e.assign(augmented=True) for e in extras], ignore_index=True)
    out["lead"] = out["text"].map(lead_gender)
    for key, group in ([(None, out)] if by is None else out.groupby(by, sort=False, dropna=False)):
        n_male, n_female = int((group["lead"] == "male").sum()), int((group["lead"] == "female").sum())
        if n_male != n_female:
            log.warning("Gender balance short of parity%s: %d male vs %d female leads after swapping",
                        "" if key is None else f" for {key}", n_male, n_female)
    return out


def stratify_key(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """One string per row combining ``columns``, for stratified train/val/test splits. Missing values
    become "unknown" (the same label the readability baseline uses)."""
    cols = df[list(columns)]
    return cols.astype(object).where(cols.notna(), "unknown").astype(str).agg(" / ".join, axis=1)


def split_strata(category: pd.Series, eras: pd.Series, min_size: int = SMALL_SLICE) -> pd.Series:
    """Genre × era split keys, with strata under ``min_size`` pooled so every stratum can be split.

    A small era joins the nearest era of the same genre that is big enough (ties go to the earlier
    era); a small "unknown" era joins the genre's largest era; a genre with no era big enough becomes
    one stratum ("all eras").
    """
    category = category.astype(object).where(category.notna(), "unknown").astype(str)
    eras = eras.astype(object).where(eras.notna(), "unknown").astype(str)
    dated = [e for e in ERA_ORDER if e != "unknown"]
    pooled = eras.copy()
    for cat in category.unique():
        in_cat = category == cat
        sizes = eras[in_cat].value_counts()
        big = [e for e in dated if sizes.get(e, 0) >= min_size]
        if not big:
            pooled[in_cat] = "all eras"
            continue
        for e, n in sizes.items():
            if n >= min_size:
                continue
            if e in dated:
                at = dated.index(str(e))
                target = min(big, key=lambda b: abs(dated.index(b) - at))
            else:
                target = max(big, key=lambda b: sizes[b])
            pooled[in_cat & (eras == e)] = target
    return stratify_key(pd.DataFrame({"category": category, "era": pooled}), ["category", "era"])


# ------------------------------------------------------------------ report

def read_quickdraw_recognition(store: Store, chunk: int = 50) -> pd.DataFrame:
    """Country and recognized flag for every raw Quick, Draw! drawing (processed data keeps only
    recognized drawings, so the hit rate has to come from raw/)."""
    parts = store.glob("raw/quickdraw/*/part-*.jsonl.gz")
    rows: list[tuple[str, bool]] = []
    for i in range(0, len(parts), chunk):
        for data in store.read_many(parts[i:i + chunk]).values():
            for line in gzip.decompress(data).splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rows.append((rec.get("countrycode") or "unknown", bool(rec.get("recognized"))))
    return pd.DataFrame(rows, columns=["country_code", "recognized"])


def build_report(store: Store | None = None, max_country_share: float = 0.20, seed: int = 0,
                 story_sample: int | None = 20000) -> dict[str, Any]:
    """Compute every slice, finding and mitigation. Reads only; writes nothing."""
    store = store or Store()
    report: dict[str, Any] = {"created_at": datetime.now(UTC).isoformat(), "data_uri": store.uri,
                              "max_country_share": max_country_share}

    drawings = read_dataset("drawings", store)
    if not drawings.empty:
        raw = read_quickdraw_recognition(store)
        weights = capped_weights(drawings["country_code"].fillna("unknown"), max_country_share)
        report["drawings"] = {
            "rows": len(drawings), "countries": int(drawings["country_code"].nunique()),
            "representation": representation(drawings["country_code"]),
            "recognition": recognition_by_country(raw) if not raw.empty else None,
            "raw_rows": len(raw),
            "shares_before": effective_shares(drawings["country_code"].fillna("unknown")),
            "shares_after": effective_shares(drawings["country_code"].fillna("unknown"), weights),
        }

    stories = read_dataset("stories_text", store)
    if not stories.empty:
        sample = stories
        if story_sample and len(stories) > story_sample:
            # Every picture book plus a fixed random sample of TinyStories, to keep the run quick.
            books = stories[stories["source"] != "tinystories"]
            tiny = stories[stories["source"] == "tinystories"]
            sample = pd.concat([books, tiny.sample(n=min(len(tiny), story_sample), random_state=seed)])
        leads = sample["text"].map(lead_gender)
        before = pd.crosstab(sample["source"], leads, normalize="index").round(4)
        balanced = balance_by_swap(sample, by="source", seed=seed)
        report["stories"] = {
            "rows": len(stories), "analysed": len(sample),
            "lead_by_source": before,
            "lead_counts": leads.value_counts().to_dict(),
            "after_by_source": pd.crosstab(balanced["source"], balanced["lead"]),
            "augmented_by_source": balanced[balanced["augmented"]]["source"].value_counts().to_dict(),
            "after_counts": balanced["lead"].value_counts().to_dict(),
            "augmented_rows": int(balanced["augmented"].sum()),
        }

    clear = read_dataset("readability", store)
    if not clear.empty:
        baseline = readability_baseline_by_slice(clear)
        eras = era(clear["pub_year"])
        unpooled = stratify_key(pd.DataFrame({"category": clear["category"], "era": eras}), ["category", "era"])
        keys = split_strata(clear["category"], eras)
        report["readability"] = {"rows": len(clear), "baseline": baseline,
                                 "strata": keys.value_counts().sort_index(),
                                 "pooled_rows": int((keys != unpooled).sum())}

    pages = read_dataset("storybook_pages", store)
    if not pages.empty:
        per_source = pages.groupby("source").agg(
            books=("book_id", "nunique"), pages=("page_number", "size"),
            illustrated_pages=("image_uri", lambda s: int(s.notna().sum())),
            words_per_page=("word_count", "median"))
        per_source["share_of_illustrations"] = (per_source["illustrated_pages"]
                                                / per_source["illustrated_pages"].sum()).round(4)
        report["picture_books"] = {"by_source": per_source}
    return report


def _md_table(df: pd.DataFrame, index_name: str, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = [f"| {index_name} | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [format(row[c], fmt[c]) if c in fmt and pd.notna(row[c]) else str(row[c]) for c in cols]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _slice_verdict(table: pd.DataFrame, noun: str) -> str | None:
    """"rates X easier ... and Y harder", from the slices big enough to judge; None if none is off by
    at least ``MEAN_ERROR_FLAG``. Mean error > 0 means the formula rates the slice too easy."""
    judged = table[table["excerpts"] >= SMALL_SLICE]["mean_error"]
    if len(judged) < 2:
        return None
    parts = []
    if judged.max() >= MEAN_ERROR_FLAG:
        parts.append(f"**{judged.idxmax()}** {noun} easier than teachers do ({judged.max():+.3f})")
    if judged.min() <= -MEAN_ERROR_FLAG:
        parts.append(f"**{judged.idxmin()}** {noun} harder ({judged.min():+.3f})")
    return " and ".join(parts) or None


def _readability_findings(b: dict[str, Any]) -> list[str]:
    """Plain-language findings computed from the baseline tables."""
    lines = []
    for key, label, noun in (("era", "By era", "texts"), ("category", "By genre", "texts")):
        verdict = _slice_verdict(b[key], noun)
        if verdict:
            lines.append(f"- {label}, the formula rates {verdict}; MAE differs by {b[f'{key}_mae_gap']:.3f} "
                         f"across slices with at least {SMALL_SLICE} excerpts.")
    flagged = bool(lines)
    small = [str(e) for e in b["era"].index if b["era"].loc[e, "excerpts"] < SMALL_SLICE]
    if small:
        lines.append(f"- Too small to judge (under {SMALL_SLICE} excerpts): {', '.join(small)}.")
    if flagged:
        lines.append("- So a scorer trained on a random split could learn era and genre quirks instead of "
                     "difficulty, and a single overall score would hide it.")
    else:
        lines.append(f"- No genre or era large enough to judge is off by {MEAN_ERROR_FLAG} or more: the baseline "
                     "treats them alike.")
    return lines


def _drawing_findings(report: dict[str, Any], d: dict[str, Any]) -> list[str]:
    cap = report["max_country_share"]
    top, top_share = str(d["shares_before"].index[0]), float(d["shares_before"].iloc[0])
    lines = []
    if top_share > cap:
        lines.append(f"- **Representation bias:** {top} alone is {_pct(top_share)} of drawings, above the "
                     f"{_pct(cap)} cap. A recognizer trained on this learns mostly how people in {top} draw a "
                     "cat or a house.")
    else:
        lines.append(f"- **Representation:** no country is above the {_pct(cap)} cap; the largest, {top}, is "
                     f"{_pct(top_share)}.")
    rec = d["recognition"]
    if rec and rec["gap"] >= RECOGNITION_GAP:
        lines.append(f"- **Performance gap:** the game's own model recognizes drawings from {rec['worst'][0]} "
                     f"({_pct(rec['worst'][1])}) less often than from {rec['best'][0]} ({_pct(rec['best'][1])}). "
                     "Part of that is drawing style, which is exactly what our recognizer must handle for "
                     "children everywhere.")
    elif rec:
        lines.append(f"- **Recognition is even across countries:** a {rec['gap'] * 100:.1f}-point gap between the "
                     "best and the worst.")
    return lines


def _origin_finding(by_source: pd.DataFrame) -> str:
    shares = by_source["share_of_illustrations"]
    names = {src: ORIGIN.get(str(src), str(src)) for src in shares.index}
    if len(shares) < 2:
        return (f"Only {', '.join(names.values()) or 'no source'} contributes illustrations, so the illustration "
                "style model would see one tradition only.")
    listed = ", ".join(f"{names[s]} {_pct(v)}" for s, v in shares.items())
    if shares.max() - 1 / len(shares) <= ORIGIN_TOLERANCE:
        return (f"Illustrations come in similar proportions ({listed}), so the illustration style model sees "
                "every tradition. No re-weighting is needed; the balance is re-checked whenever this report "
                "is regenerated.")
    return (f"Illustrations skew toward {names[shares.idxmax()]} ({listed}): re-weight illustrations by origin "
            "when training the illustration style model.")


def render_markdown(report: dict[str, Any]) -> str:
    out = ["# Data bias report", "",
           (f"Generated by `primer-data bias-report` on {report['created_at'][:10]} from `{report['data_uri']}`. "
           "Regenerate it after new data lands; the numbers below come from the code in "
           "`src/primer/data/bias.py`, not from hand edits."), ""]

    d = report.get("drawings")
    if d:
        rec = d["recognition"]
        top = str(d["shares_before"].index[0])
        top_before = float(d["shares_before"].iloc[0])
        top_after = float(d["shares_after"].get(top, 0.0))
        out += ["## 1. Drawings by country (Quick, Draw!)", "",
                f"{d['rows']:,} drawings from {d['countries']} countries.", "",
                "**Who drew them**", "",
                _md_table(d["representation"], "country", {"count": ",.0f", "share": ".1%"}), ""]
        if rec:
            out += [("**How often the game's recognizer understood them** (Fairlearn `MetricFrame`, "
                    f"{d['raw_rows']:,} raw drawings, top {rec['top']} countries, rest pooled as Other). "
                    "These come from `raw/`, because processed data keeps only recognized drawings, so the "
                    "count differs from the table above."), "",
                    _md_table(rec["by_country"], "country", {"drawings": ",.0f", "recognition_rate": ".1%"}), "",
                    (f"Overall {_pct(rec['overall_rate'])}; best {rec['best'][0]} {_pct(rec['best'][1])}, "
                    f"worst {rec['worst'][0]} {_pct(rec['worst'][1])} (Other not counted): a "
                    f"**{rec['gap'] * 100:.1f}-point gap**."), ""]
        out += ["**Findings**", "", *_drawing_findings(report, d), "",
                "**Mitigation: capped sample weights** (`capped_weights`)", "",
                (f"No country may hold more than {_pct(report['max_country_share'])} of the training weight. "
                "Over-represented countries are down-weighted; every other drawing keeps weight 1, so no data is "
                "dropped and small countries are not inflated by huge weights."), "",
                _md_table(pd.DataFrame({"before": d["shares_before"], "after": d["shares_after"]}).fillna(0)
                          .sort_values("after", ascending=False).head(10), "country",
                          {"before": ".1%", "after": ".1%"}), "",
                (f"{top} share of the training signal: {_pct(top_before)} → {_pct(top_after)}. The recognizer "
                "team passes these weights to the loss (or sampler) and re-measures the per-country recognition "
                "gap on the trained model with the same `MetricFrame` slices (KAN-80)."), "",
                ("**Trade-off:** the effective training set shrinks (the down-weighted rows count for less), so "
                "overall accuracy may dip slightly in exchange for more even per-country accuracy."), ""]

    s = report.get("stories")
    if s:
        lb = s["lead_by_source"]
        out += ["## 2. Stories by lead-character gender", "",
                (f"{s['analysed']:,} of {s['rows']:,} stories analysed (every picture book plus a fixed random "
                "sample of TinyStories). A story's lead is male or female when its gendered words "
                "and common names (he/she, boy/girl, Tom/Lily, ...) lean that way; otherwise neutral."), "",
                _md_table(lb, "source", {c: ".1%" for c in lb.columns}), "",
                "**Findings**", ""]
        for source, row in lb.iterrows():
            male, female = float(row.get("male", 0.0)), float(row.get("female", 0.0))
            gap = male - female
            verdict = ("roughly balanced" if abs(gap) < 0.05
                       else f"skews **{'male' if gap > 0 else 'female'}** by {abs(gap) * 100:.0f} points")
            out.append(f"- `{source}`: {_pct(male)} male vs {_pct(female)} female leads, {verdict}.")
        after_table = s["after_by_source"]
        after = after_table.assign(added=pd.Series(s["augmented_by_source"])).fillna(0)
        short = [str(src) for src, row in after_table.iterrows()
                 if row.get("male", 0) != row.get("female", 0)]
        parity = ("Male and female leads are equal in every source after swapping." if not short else
                  f"Parity is not reached for {', '.join(short)}: some swapped copies come out neutral or "
                  "unchanged (the swap is a heuristic).")
        out += ["", "**Mitigation: gender-swap augmentation, per source** (`gender_swap`, `balance_by_swap`)", "",
                (f"{s['augmented_rows']:,} gender-swapped copies of majority-lead stories are computed, balanced "
                 "within each source so a large balanced source can't hide a skewed small one:"), "",
                _md_table(after, "source", dict.fromkeys(after.columns, ",.0f")), "",
                parity, ("Once the copies are added to the training data (KAN-80), the story LLM sees male and "
                         "female leads equally often within each source."), "",
                ("**Trade-off / limits:** the swap is word-based (\"her\" → him/his and \"his\" → her/hers are "
                 "heuristics) and only knows common names, so a few swapped stories read awkwardly. Copies are "
                 "marked `augmented=True` so they can be kept out of evaluation and in the same split as their "
                 "original (KAN-80)."), ""]

    r = report.get("readability")
    if r:
        b = r["baseline"]
        fmt = {"excerpts": ",.0f", "mae": ".3f", "mean_error": "+.3f", "teacher_score_mean": "+.3f"}
        out += ["## 3. Readability (CLEAR) by genre and era", "",
                (f"{r['rows']:,} excerpts. Teacher score `BT_easiness`: higher = easier. The Flesch-Kincaid grade "
                "(the baseline our readability scorer must beat) is mapped onto that score with a straight-line "
                f"fit; overall MAE {b['overall_mae']:.3f}."), "",
                "**By genre**", "", _md_table(b["category"], "genre", fmt), "",
                "**By publication era**", "", _md_table(b["era"], "era", fmt), "",
                "**Findings** (mean error > 0 means the formula rates the slice easier than teachers do)", "",
                *_readability_findings(b), "",
                ("**Mitigation: stratified splits** (`split_strata`): train/validation/test are split by "
                "genre × era so every slice appears in each split in the same proportion, and the scorer is "
                "evaluated per slice with the same `MetricFrame`, not just overall."), "",
                (f"Strata with fewer than {SMALL_SLICE} excerpts are pooled into the nearest era of the same genre "
                 f"(excerpts without a year into the genre's largest era); {r['pooled_rows']:,} excerpts were "
                 "moved. The split keys after pooling:"), "",
                _md_table(r["strata"].to_frame("excerpts"), "genre / era", {"excerpts": ",.0f"}), ""]

    p = report.get("picture_books")
    if p:
        out += ["## 4. Picture books by origin", "",
                _md_table(p["by_source"], "source", {"books": ",.0f", "pages": ",.0f", "illustrated_pages": ",.0f",
                                                     "share_of_illustrations": ".1%", "words_per_page": ".0f"}), "",
                _origin_finding(p["by_source"]), ""]

    out += ["## What this does not cover yet", "",
            ("- The mitigations are computed and checked here, but not yet applied to the training data: "
             "the weights column, the augmented stories and the split assignment come with the training "
             "loaders (KAN-80)."),
            ("- This report reads the current processed data; tying it to a frozen dataset version is "
             "KAN-81."),
            ("- Our own models are not trained yet. Once they are, the same slices are reused to measure them "
            "(recognizer per country, readability scorer per genre/era, story LLM output per lead gender)."),
            ("- Country is the only demographic-like field; there is no age, gender or location of the people "
            "who drew. Children's drawings in the pilot are never sliced by child."), ""]
    return "\n".join(out)


def report_to_json(report: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy of ``build_report``'s output (DataFrames become records)."""
    def convert(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return value.reset_index().to_dict(orient="records")
        if isinstance(value, pd.Series):
            return {str(k): (None if pd.isna(v) else v) for k, v in value.to_dict().items()}
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value
    return convert(report)
