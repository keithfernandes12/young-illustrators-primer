"""Comparison charts for the grid. Needs matplotlib (the `story` extra); `grid.py` skips them without it.

Two charts, the ones the course guidelines ask for when comparing candidates:

- `comparison.png` -- target-word inclusion, first-attempt valid JSON and format compliance for every
  configuration, side by side;
- `temperature_sweep.png` -- target-word inclusion against temperature, one line per model and
  max_new_tokens, which is the sensitivity result in picture form.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

BARS = ("target_word_inclusion", "first_attempt_valid_json_rate", "format_compliance_rate")


def write_charts(runs: Sequence[dict[str, Any]], out: Path) -> list[Path]:
    matplotlib.use("Agg")  # files only, never a window; must be chosen before pyplot is imported
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    names = [run["name"] for run in runs]
    width = 0.8 / len(BARS)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(names)), 4.5))
    for i, metric in enumerate(BARS):
        values = [run["metrics"].get(metric) or 0.0 for run in runs]
        ax.bar([x + i * width for x in range(len(names))], values, width, label=metric)
    ax.set_xticks([x + width for x in range(len(names))], names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Story grid: constraint metrics per configuration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    comparison = out / "comparison.png"
    fig.savefig(comparison, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    lines: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        label = f"{run['params']['model']} n={run['params']['max_new_tokens']}"
        lines.setdefault(label, []).append((run["params"]["temperature"],
                                            run["metrics"].get("target_word_inclusion") or 0.0))
    for label, points in sorted(lines.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
    ax.set_xlabel("temperature")
    ax.set_ylabel("target_word_inclusion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sensitivity: target-word inclusion vs temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    sweep = out / "temperature_sweep.png"
    fig.savefig(sweep, dpi=120)
    plt.close(fig)
    return [comparison, sweep]
