"""Sensitivity analysis for the story stage: which settings actually move the metrics.

SHAP and LIME do not apply to a generative stage: there is no feature vector to attribute
(`model-development` section 3). The honest substitute is ablation and sweeps. In this submission that
is the generation grid: for each hyperparameter, how far each metric moves across that parameter's
values, averaged over the other settings. A wide spread means the metric is sensitive to that knob.

Prompt-constraint ablation (drop the word list, the level, the avoid-list topics in turn) is S8a in
the plan and is deferred (`CARD.md`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

PARAMETERS = ("model", "temperature", "max_new_tokens")
METRICS = ("target_word_inclusion", "first_attempt_valid_json_rate", "format_compliance_rate",
           "success_rate", "p95_latency_s")


def explain(runs: Sequence[dict[str, Any]], parameters: Sequence[str] = PARAMETERS,
            metrics: Sequence[str] = METRICS) -> dict[str, dict[str, float | None]]:
    """`{parameter: {metric: spread}}`, where spread = max - min of the metric's mean at each value.

    Each run is `{"params": {...}, "metrics": {...}}`, as `grid.py` produces.
    """
    out: dict[str, dict[str, float | None]] = {}
    for parameter in parameters:
        out[parameter] = {}
        for metric in metrics:
            by_value: dict[str, list[float]] = {}
            for run in runs:
                value = run["metrics"].get(metric)
                if value is not None:
                    by_value.setdefault(str(run["params"][parameter]), []).append(float(value))
            means = [sum(v) / len(v) for v in by_value.values() if v]
            out[parameter][metric] = round(max(means) - min(means), 4) if len(means) >= 2 else None
    return out
