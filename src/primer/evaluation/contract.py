"""The metrics contract shared by every model area.

`golden.py` merges each area's numbers into one flat file and `.claude/skills/golden-gate` reads it.
The gate looks keys up at the top level of that file, so a key the writer forgets is reported as
``metric missing`` and fails the gate -- it is never silently skipped. ``GOLDEN_METRIC_KEYS`` is the
list the writer must produce, and a test asserts it matches `thresholds.json` exactly, so adding a
threshold without teaching the writer to emit it breaks the build rather than passing quietly.
"""

from __future__ import annotations

from typing import Final

#: Keys the gate compares against a threshold. Must stay equal to the `key` fields in
#: .claude/skills/golden-gate/thresholds.json.
GATED_METRIC_KEYS: Final[tuple[str, ...]] = (
    "drawing_top5_acc",
    "target_word_inclusion",
    "story_quality_mean",
    "reading_level_within1",
    "readability_rmse",
    "safety_violations_golden",
    "clip_consistency",
    "p95_first_page_s",
    "p95_full_book_s",
    "cost_per_book_usd",
)

#: Keys a threshold refers to with `ref` rather than a fixed value. The gate reads these from the
#: metrics file too, so the writer must emit them or the comparison has nothing to compare against.
REFERENCE_METRIC_KEYS: Final[tuple[str, ...]] = ("fk_baseline_rmse",)

#: Everything `golden.py` must write.
GOLDEN_METRIC_KEYS: Final[tuple[str, ...]] = GATED_METRIC_KEYS + REFERENCE_METRIC_KEYS

#: Which stage owns which gated metric. Used to report per-stage detail and to tell a member which
#: numbers their area is responsible for.
METRIC_OWNERS: Final[dict[str, str]] = {
    "drawing_top5_acc": "drawing",
    "target_word_inclusion": "story",
    "story_quality_mean": "story",
    "reading_level_within1": "readability",
    "readability_rmse": "readability",
    "fk_baseline_rmse": "readability",
    "safety_violations_golden": "safety",
    "clip_consistency": "illustration",
    "p95_first_page_s": "system",
    "p95_full_book_s": "system",
    "cost_per_book_usd": "system",
}


def missing_keys(metrics: dict[str, object]) -> list[str]:
    """Which contract keys `metrics` does not carry. Empty means the file is complete."""
    return [key for key in GOLDEN_METRIC_KEYS if key not in metrics]
