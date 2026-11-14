"""Slice keys for the speech stage, reusing ``primer.data.bias``.

Every metric is also reported per slice, so a recogniser that is good on average and bad for young
children is visible rather than averaged away. A slice under ``SMALL_SLICE`` clips is reported but
not judged, the same floor the data pipeline's bias report uses.

Accent is the slice scoping §9 worries about most and the one Speechocean762 cannot measure (every
speaker's first language is Mandarin); it needs MyST or Common Voice metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from primer.data.bias import SMALL_SLICE

#: Slice keys this stage reports. `accent` is reported whenever a dataset supplies it.
SLICE_KEYS: tuple[str, ...] = ("age_band", "gender", "accent")

#: Age bands, aligned with the product's two modes (3-5 Learn to Read, 5-8 Grow Vocabulary).
AGE_BANDS: tuple[tuple[int, str], ...] = ((5, "3-5"), (8, "6-8"), (12, "9-12"), (17, "13-17"))


def age_band(age: int) -> str:
    """The band for `age` in years. Exact ages are never reported, only the band."""
    for upper, band in AGE_BANDS:
        if age <= upper:
            return band
    return "adult"


def slice_of(record: Any, key: str) -> str | None:
    """The value of `key` for `record` (a ``ReadAloudClip``), or None when it does not carry it."""
    slices: Mapping[str, str] = getattr(record, "slices", {})
    return slices.get(key)


def judged(slices: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The slices with enough clips to draw a conclusion from."""
    return [s for s in slices if s["n"] >= SMALL_SLICE]


def disparities(slices: Sequence[Mapping[str, Any]], metric: str = "asr_wer") -> dict[str, dict[str, Any]]:
    """Per slice key, the best and worst judged value of `metric` and the gap between them."""
    out: dict[str, dict[str, Any]] = {}
    for key in sorted({s["key"] for s in slices}):
        values = [
            (s["metrics"][metric], s["value"])
            for s in judged([s for s in slices if s["key"] == key])
            if s["metrics"].get(metric) is not None
        ]
        if len(values) < 2:
            continue
        (low, low_at), (high, high_at) = min(values), max(values)
        out[key] = {"metric": metric, "best": low_at, "worst": high_at, "gap": round(high - low, 4)}
    return out


def bias_report(slices: Sequence[Mapping[str, Any]], metric: str = "asr_wer") -> str:
    """A short Markdown table of every slice, with the small ones marked, plus the gaps."""
    lines = [f"| slice | value | n | {metric} | judged |", "|---|---|---|---|---|"]
    for s in slices:
        value = s["metrics"].get(metric)
        shown = "n/a" if value is None else f"{value:.3f}"
        lines.append(
            f"| {s['key']} | {s['value']} | {s['n']} | {shown} | {'yes' if s['n'] >= SMALL_SLICE else 'too small'} |"
        )
    gaps = disparities(slices, metric)
    lines.append("")
    if not gaps:
        lines.append(
            f"No slice key has two judged values (floor: {SMALL_SLICE} clips), so no gap is claimed."
        )
    for key, gap in gaps.items():
        lines.append(
            f"- **{key}**: {metric} is worst for `{gap['worst']}` and best for `{gap['best']}`, a gap of {gap['gap']:.3f}."
        )
    return "\n".join(lines) + "\n"
