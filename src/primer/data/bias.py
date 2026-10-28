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
stories, and stratified split keys for readability. ``build_report`` computes all of
it; ``render_markdown`` writes the report (see ``primer-data bias-report``).
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
FEMALE_TO_MALE = {v: k for k, v in MALE_TO_FEMALE.items() if k != "his"}  # "her" is handled separately
MALE_NAMES = {"Tom": "Lily", "Ben": "Anna", "Timmy": "Mia", "Tim": "Sue", "Max": "Lucy", "Jack": "Sara",
              "John": "Amy", "Bob": "Jill", "Joe": "Emma", "Billy": "Molly", "Jake": "Zoe", "Leo": "Ella"}
FEMALE_NAMES = {v: k for k, v in MALE_NAMES.items()}
MALE_WORDS = set(MALE_TO_FEMALE) | {n.lower() for n in MALE_NAMES}
FEMALE_WORDS = set(FEMALE_TO_MALE) | {"her", "hers"} | {n.lower() for n in FEMALE_NAMES}
_WORD = re.compile(r"[A-Za-z']+")
# After "her", these words mean it is an object ("saw her and ...") → "him"; otherwise a possessive → "his".
_HER_OBJECT_FOLLOWERS = {"and", "or", "but", "to", "with", "for", "a", "an", "the", "that", "up", "down",
                         "back", "again", "too", "so", "very", "in", "on", "at", "of", "from", "as", "if"}


def gender_counts(text: str) -> tuple[int, int]:
    words = [w.lower() for w in _WORD.findall(text)]
    return sum(w in MALE_WORDS for w in words), sum(w in FEMALE_WORDS for w in words)


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

    "her" is ambiguous (him/his); it becomes "his" before a word that looks like a noun phrase and
    "him" otherwise. A heuristic: good enough for augmentation, not for publishing.
    """
    tokens = re.split(r"([A-Za-z']+)", text)
    out = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if tok in MALE_NAMES:
            out.append(MALE_NAMES[tok])
        elif tok in FEMALE_NAMES:
            out.append(FEMALE_NAMES[tok])
        elif low == "her":
            nxt = next((t for t in tokens[i + 1:] if t.strip()), "")
            gap = tokens[i + 1] if i + 1 < len(tokens) else ""
            possessive = bool(_WORD.fullmatch(nxt)) and gap == " " and nxt.lower() not in _HER_OBJECT_FOLLOWERS
            out.append(_match_case("his" if possessive else "him", tok))
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
    are pooled as "Other" so every slice is large enough to compare.
    """
    recognized = drawings["recognized"].astype(int).to_numpy()
    countries = group_top(drawings["country_code"], top)
    mf = MetricFrame(metrics={"drawings": count, "recognition_rate": selection_rate},
                     y_true=recognized, y_pred=recognized, sensitive_features=countries.to_numpy())
    by_group = mf.by_group.sort_values("drawings", ascending=False)
    rates = by_group["recognition_rate"]
    return {
        "by_country": by_group,
        "overall_rate": float(mf.overall["recognition_rate"]),
        "gap": float(mf.difference()["recognition_rate"]),
        "worst": (str(rates.idxmin()), float(rates.min())),
        "best": (str(rates.idxmax()), float(rates.max())),
    }


ERA_ORDER = ["before 1850", "1850–1899", "1900–1949", "1950–1999", "2000+", "unknown"]
SMALL_SLICE = 100  # slices with fewer rows are reported but flagged as too small to judge


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
    easier than teachers do).
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
        out[f"{name}_mae_gap"] = float(mf.difference()["mae"])
    return out


# ------------------------------------------------------------------ mitigations

def capped_weights(groups: pd.Series, max_share: float) -> pd.Series:
    """Per-row sample weights so no group holds more than ``max_share`` of the total weight.

    Groups under the cap keep weight 1, so no data is thrown away and small groups aren't inflated;
    only over-represented groups are down-weighted.
    """
    counts = groups.value_counts().astype(float)
    capped: set = set()
    while True:
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
    skewed small one. Copies are marked ``augmented=True``; originals are never changed.
    """
    groups = [stories] if by is None or by not in stories else [g for _, g in stories.groupby(by, sort=False)]
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
    return out


def stratify_key(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """One string per row combining ``columns``, for stratified train/val/test splits."""
    return df[list(columns)].astype(str).agg(" / ".join, axis=1)


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
        keys = stratify_key(clear.assign(era=era(clear["pub_year"])), ["category", "era"])
        report["readability"] = {"rows": len(clear), "baseline": baseline,
                                 "strata": keys.value_counts().sort_index()}

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


def _readability_findings(b: dict[str, Any]) -> list[str]:
    """Plain-language findings computed from the baseline tables (mean error > 0 = rated too easy)."""
    era_t = b["era"][b["era"]["excerpts"] >= SMALL_SLICE]
    lines = []
    if not era_t.empty:
        easy, hard = era_t["mean_error"].idxmax(), era_t["mean_error"].idxmin()
        mae_gap = era_t["mae"].max() - era_t["mae"].min()  # only slices big enough to judge
        lines.append(f"- By era, the formula rates **{easy}** texts easier than teachers do "
                     f"({era_t.loc[easy, 'mean_error']:+.3f}) and **{hard}** texts harder "
                     f"({era_t.loc[hard, 'mean_error']:+.3f}); MAE differs by {mae_gap:.3f} across those eras.")
    cat = b["category"]
    if len(cat) > 1:
        easy = cat["mean_error"].idxmax()
        lines.append(f"- By genre, it rates **{easy}** easier than teachers do "
                     f"({cat.loc[easy, 'mean_error']:+.3f}); MAE differs by {b['category_mae_gap']:.3f}.")
    small = [str(e) for e in b["era"].index if b["era"].loc[e, "excerpts"] < SMALL_SLICE]
    if small:
        lines.append(f"- Too small to judge (under {SMALL_SLICE} excerpts): {', '.join(small)}.")
    lines.append("- So a scorer trained on a random split could learn era and genre quirks instead of "
                 "difficulty, and a single overall score would hide it.")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    out = ["# Data bias report", "",
           (f"Generated by `primer-data bias-report` on {report['created_at'][:10]} from `{report['data_uri']}`. "
           "Regenerate it after new data lands; the numbers below come from the code in "
           "`src/primer/data/bias.py`, not from hand edits."), ""]

    d = report.get("drawings")
    if d:
        rec = d["recognition"]
        us_before = float(d["shares_before"].get("US", 0.0))
        us_after = float(d["shares_after"].get("US", 0.0))
        out += ["## 1. Drawings by country (Quick, Draw!)", "",
                f"{d['rows']:,} drawings from {d['countries']} countries.", "",
                "**Who drew them**", "",
                _md_table(d["representation"], "country", {"count": ",.0f", "share": ".1%"}), ""]
        if rec:
            out += [("**How often the game's recognizer understood them** (Fairlearn `MetricFrame`, "
                    f"{d['raw_rows']:,} raw drawings, top 15 countries, rest pooled as Other)"), "",
                    _md_table(rec["by_country"], "country", {"drawings": ",.0f", "recognition_rate": ".1%"}), "",
                    (f"Overall {_pct(rec['overall_rate'])}; best {rec['best'][0]} {_pct(rec['best'][1])}, "
                    f"worst {rec['worst'][0]} {_pct(rec['worst'][1])}: a **{rec['gap'] * 100:.1f}-point gap**."), ""]
        out += ["**Findings**", "",
                (f"- **Representation bias:** the US alone is {_pct(us_before)} of drawings. A recognizer trained "
                "on this learns mostly how Americans draw a cat or a house."),
                ("- **Performance gap:** drawings from some countries are recognized less often by the game's "
                "own model. Part of that is drawing style, which is exactly what our recognizer must handle for "
                "children everywhere."), "",
                "**Mitigation: capped sample weights** (`capped_weights`)", "",
                (f"No country may hold more than {_pct(report['max_country_share'])} of the training weight. "
                "Over-represented countries are down-weighted; every other drawing keeps weight 1, so no data is "
                "dropped and small countries are not inflated by huge weights."), "",
                _md_table(pd.DataFrame({"before": d["shares_before"], "after": d["shares_after"]}).fillna(0)
                          .sort_values("after", ascending=False).head(10), "country",
                          {"before": ".1%", "after": ".1%"}), "",
                (f"US share of the training signal: {_pct(us_before)} → {_pct(us_after)}. The recognizer team "
                "passes these weights to the loss (or sampler) and re-measures the per-country recognition gap "
                "on the trained model with the same `MetricFrame` slices."), "",
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
        after = s["after_by_source"].assign(added=pd.Series(s["augmented_by_source"])).fillna(0)
        out += ["", "**Mitigation: gender-swap augmentation, per source** (`gender_swap`, `balance_by_swap`)", "",
                (f"{s['augmented_rows']:,} gender-swapped copies of majority-lead stories are added, balanced "
                 "within each source so a large balanced source can't hide a skewed small one:"), "",
                _md_table(after, "source", dict.fromkeys(after.columns, ",.0f")), "",
                "The story LLM then sees each kind of plot with a male and a female lead equally often.", "",
                ("**Trade-off / limits:** the swap is word-based (\"her\" → him/his is a heuristic) and only knows "
                 "common names, so a few swapped stories read awkwardly. They are marked `augmented=True`, and the "
                 "golden evaluation set uses only original stories."), ""]

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
                ("**Mitigation: stratified splits** (`stratify_key`): train/validation/test are split by "
                "genre × era so every slice appears in each split in the same proportion, and the scorer is "
                "evaluated per slice with the same `MetricFrame`, not just overall."), "",
                _md_table(r["strata"].to_frame("excerpts"), "genre / era", {"excerpts": ",.0f"}), "",
                (f"Strata with fewer than {SMALL_SLICE} excerpts are too small to judge on their own; for "
                 "splitting they are pooled with the neighbouring era."), ""]

    p = report.get("picture_books")
    if p:
        out += ["## 4. Picture books by origin", "",
                _md_table(p["by_source"], "source", {"books": ",.0f", "pages": ",.0f", "illustrated_pages": ",.0f",
                                                     "share_of_illustrations": ".1%", "words_per_page": ".0f"}), "",
                ("African and Indian picture books contribute illustrations in similar proportions, so the "
                "illustration style model sees both traditions. No re-weighting is needed now; the balance is "
                "re-checked here whenever the books are re-ingested."), ""]

    out += ["## What this does not cover yet", "",
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
