"""Sensitivity analysis for the illustration stage.

SHAP for classifiers, ablation for generative stages. Logged as a run artifact, because the course
guidelines ask for an explanation of what the model keyed on.
"""

from __future__ import annotations

from typing import Any


def explain(model: Any, sample: Any) -> Any:
    """Return the per-feature or per-component attribution for one sample."""
    raise NotImplementedError("illustration/sensitivity.py is a stub")
