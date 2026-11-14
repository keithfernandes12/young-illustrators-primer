"""Run the golden set and write the merged metrics file the release gate reads.

    python -m primer.evaluation.golden --split golden --fake-models --out .local-data/eval/golden_metrics.json

Every model area writes its own `metrics.json`; this merges them into one flat file, because
`.claude/skills/golden-gate/scripts/check_gate.py` looks each threshold key up at the top level. The
writer emits exactly `contract.GOLDEN_METRIC_KEYS`, so a key that is never written is reported by the
gate as ``metric missing`` and fails it.

No area under `src/primer/<area>/` is implemented yet, so `--fake-models` is currently the only mode
that runs; it is also what CI would use, since the real stages need GPUs.

**These fakes do not share the area Protocols.** They predate them and their signatures differ for
five of six stages -- `drawing.top5(label)` against `DrawingRecognizer.analyse(image)`, and so on --
so an area implementation cannot be passed here as it stands. `safety.flagged(prompt, expect_flagged)`
takes the ground truth as an argument, which no real classifier can. Reconciling the two, by
adapter or by rewriting these to the Protocols, is open work and belongs with the first real area;
it is recorded in docs/plans/autonomous-loops.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from primer.evaluation import contract
from primer.evaluation.fakes import FakeComponents

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIOS = ROOT / "configs" / "golden" / "scenarios.jsonl"

Scenario = dict[str, Any]


def load_scenarios(path: Path) -> list[Scenario]:
    """Read the golden set. One JSON object per line, so a diff shows one scenario per line."""
    scenarios: list[Scenario] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                scenarios.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not valid JSON: {exc}") from exc
    if not scenarios:
        raise ValueError(f"{path} has no scenarios")
    return scenarios


def _rmse(errors: Sequence[float]) -> float:
    return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0


def contains_target_word(text: str, target: str) -> bool:
    """Whether `text` uses `target` as a word.

    Substring membership overstates the metric: asking for "cat" and generating "concatenate" would
    count as inclusion, which hides a real target-word regression. A multiword target must appear as
    a contiguous run of words, in order.
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    wanted = re.findall(r"[a-z0-9']+", target.lower())
    if not wanted:
        return False
    return any(words[i : i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1))


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def run(scenarios: list[Scenario], components: FakeComponents) -> dict[str, Any]:
    """Score every scenario and return the merged metrics, flat keys plus per-stage detail."""
    books = [s for s in scenarios if s.get("kind") == "book"]
    if not books:
        raise ValueError("the golden set has no book scenarios")

    drawing_hits = 0
    words_wanted = 0
    words_found = 0
    quality: list[float] = []
    within_one = 0
    model_errors: list[float] = []
    baseline_errors: list[float] = []
    consistency: list[float] = []

    for scenario in books:
        label = str(scenario["drawing_label"])
        target_words = [str(word) for word in scenario.get("target_words", [])]
        age = int(scenario["age"])
        target_level = float(scenario["target_reading_level"])

        if label in components.drawing.top5(label):
            drawing_hits += 1

        story = components.story.write(str(scenario["prompt"]), target_words, age)
        words_wanted += len(target_words)
        words_found += sum(1 for word in target_words if contains_target_word(story, word))
        quality.append(components.story.quality_score(story))

        predicted = components.readability.grade(story, target_level)
        if abs(predicted - target_level) <= 1.0:
            within_one += 1
        model_errors.append(predicted - target_level)
        baseline_errors.append(components.readability.baseline_grade(story) - target_level)

        consistency.append(components.illustration.consistency_score(story, label))

    # Safety runs over every scenario: the 20 unsafe ones must be caught, the 80 safe ones must not
    # be flagged. The gated number is the unsafe cases that got through.
    missed = 0
    false_positives = 0
    for scenario in scenarios:
        expect = bool(scenario.get("expect_flagged"))
        flagged = components.safety.flagged(str(scenario["prompt"]), expect)
        if expect and not flagged:
            missed += 1
        if flagged and not expect:
            false_positives += 1

    metrics: dict[str, Any] = {
        "drawing_top5_acc": round(drawing_hits / len(books), 4),
        "target_word_inclusion": round(words_found / words_wanted, 4) if words_wanted else 0.0,
        "story_quality_mean": round(sum(quality) / len(quality), 4),
        "reading_level_within1": round(within_one / len(books), 4),
        "readability_rmse": round(_rmse(model_errors), 4),
        "fk_baseline_rmse": round(_rmse(baseline_errors), 4),
        "safety_violations_golden": missed,
        "clip_consistency": round(sum(consistency) / len(consistency), 4),
        "p95_first_page_s": components.system.first_page_s,
        "p95_full_book_s": components.system.full_book_s,
        "cost_per_book_usd": components.system.cost_per_book_usd,
    }

    absent = contract.missing_keys(metrics)
    if absent:  # a programming error here would reach the gate as a silent gap
        raise AssertionError(f"golden runner did not write: {', '.join(absent)}")

    metrics["safety_false_positives"] = false_positives  # reported, not gated
    metrics["stages"] = {
        stage: {key: metrics[key] for key, owner in contract.METRIC_OWNERS.items() if owner == stage}
        for stage in sorted(set(contract.METRIC_OWNERS.values()))
    }
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="golden", choices=["golden"], help="which split to run")
    parser.add_argument(
        "--fake-models",
        action="store_true",
        help="score with the built-in fakes; required until the model areas are implemented",
    )
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-version", default="unpinned", help="the frozen primer-data version")
    args = parser.parse_args(argv)

    if not args.fake_models:
        parser.error(
            "no model area is implemented yet, so --fake-models is required. See "
            "docs/plans/autonomous-loops.md for what has to land first."
        )

    scenarios = load_scenarios(args.scenarios)
    metrics = run(scenarios, FakeComponents())
    metrics.update(
        {
            "split": args.split,
            "fake_models": True,
            "scenarios": len(scenarios),
            "data_version": args.data_version,
            "git_sha": _git_sha(),
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(scenarios)} scenarios of metrics to {args.out}")
    for key in contract.GOLDEN_METRIC_KEYS:
        print(f"  {key} = {metrics[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
