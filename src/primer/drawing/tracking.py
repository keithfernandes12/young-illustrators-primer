"""Log the drawing stage's evaluation files to the shared MLflow server (S6c3, S6f2).

`evaluate.py --mlflow` and `sensitivity.py --mlflow` call this after their file is written, so a
run is a copy of a file that already exists on disk, never the only record of a number. The names follow
`docs/engineering/model-development.md` (Tracking conventions): experiment ``yip-drawing``, the
seven comparison tags, params from the settings the file records, ``overall`` and ``baseline``
metrics, and slices as ``slice/<key>/<value>/<metric>``. MLflow rejects ``=`` in metric names
everywhere and ``:`` on Windows, which is why a slice is a path rather than ``key=value``.

The S6c run (0.4956) and its re-measure (0.5436) were logged by a scratch script with these names;
this module replaces it, so a later run logs the same names without anyone keeping that script.

Nothing here imports ``mlflow`` at module level: the metric and tag builders are plain functions
the tests check without the package, and only `log_run` needs a server.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

EXPERIMENT = "yip-drawing"
STAGE = "drawing"
CARD_PATH = Path(__file__).resolve().parent / "CARD.md"
#: MLflow's own limit on a metric or param name's characters, applied to the parts this module
#: does not choose (a slice value, a variant name). Anything else becomes ``_``.
_ALLOWED = set("-_. /")


def metric_part(text: str) -> str:
    """One segment of a metric name, safe for MLflow on every client OS."""
    cleaned = "".join(ch if ch.isalnum() or ch in _ALLOWED - {"/"} else "_" for ch in str(text))
    if not cleaned:
        raise ValueError(f"{text!r} leaves nothing to name a metric with")
    return cleaned


def slice_metric_name(key: str, value: str, metric: str) -> str:
    """``slice/<key>/<value>/<metric>``, the name `model-development.md` fixes."""
    return f"slice/{metric_part(key)}/{metric_part(value)}/{metric_part(metric)}"


def _numbers(prefix: str, values: Mapping[str, Any]) -> dict[str, float]:
    # `bool` is an `int`; a flag is not a measurement and would log as 0/1.
    out: dict[str, float] = {}
    for k, v in values.items():
        if isinstance(v, int | float) and not isinstance(v, bool):
            _put(out, f"{prefix}{metric_part(k)}", float(v))
    return out


def _put(metrics: dict[str, float], name: str, value: float) -> None:
    """Add one metric, refusing a name already taken.

    Cleaning a name can make two different ones equal (``a:b`` and ``a_b``), and MLflow would keep
    whichever was logged last without a word -- a number silently replaced by its neighbour's.
    """
    if name in metrics:
        raise ValueError(f"two metrics would both be logged as {name!r}; rename one of the sources")
    metrics[name] = value


def _merge(*parts: Mapping[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for part in parts:
        for name, value in part.items():
            _put(merged, name, value)
    return merged


def run_tags(result: Mapping[str, Any], *, owner: str, kind: str) -> dict[str, str]:
    """The seven comparison tags, plus what kind of file the run copies.

    ``where`` is ``local`` unless ``YIP_WHERE`` says otherwise: only a Vertex job sets it, and a
    wrong ``local`` is the safer error, because a registered model has to be a Vertex run.
    """
    if not owner.strip():
        raise ValueError("--owner is required with --mlflow: runs from six machines are told apart by it")
    return {
        "stage": STAGE,
        "owner": owner,
        "data_version": str(result["data_version"]),
        # The stability summary does not record its commit; "unknown" says so rather than guessing.
        "git_sha": str(result.get("git_sha") or "unknown"),
        "base_model": str(result["encoder"]),
        "gate_group": STAGE,
        "where": os.environ.get("YIP_WHERE", "local"),
        "kind": kind,
        # Whether the code differed from `git_sha` when the run started: a sha alone over-claims.
        # `None` is what the CLIs write when git failed or timed out; it must read "unknown", not "none".
        "git_dirty": "unknown" if result.get("git_dirty") is None else str(result["git_dirty"]).lower(),
        # Whether `check_gate` may read the file at all. Only S6c's whole-test-split file is; a
        # `val` run is diagnostic and stability never is.
        "gate_eligible": str(bool(result.get("gate_eligible", False))).lower(),
    }


def run_params(result: Mapping[str, Any]) -> dict[str, str]:
    """Every setting the file records that changes a number. Nested settings are flattened one level.

    MLflow keeps params as strings, so a nested value is JSON rather than Python's ``repr``, which a
    reader could not parse back.
    """
    params: dict[str, str] = {}
    for key in ("encoder", "encoder_revision", "split", "device", "dtype", "prompt_count", "prompts_sha256"):
        if key in result:
            params[key] = str(result[key])
    if "license" in result:
        params["license"] = ",".join(result["license"])
    for group in ("scoring", "preprocessing"):
        for key, value in result.get(group, {}).items():
            text = json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else str(value)
            params[f"{group}.{metric_part(key)}"] = text
    return params


def accuracy_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    """S6c's file: ``overall``, ``baseline/…``, the run's size and time, and every slice.

    ``drawing_top5_acc`` is logged from ``overall``, never the top-level copy: that copy exists only
    when the file is gate-eligible, and a probe run must still be comparable with a full one.
    """
    metrics = _merge(_numbers("", result["overall"]), _numbers("baseline/", result["baseline"]))
    if "seconds" in result:
        _put(metrics, "seconds", float(result["seconds"]))
    for entry in result["slices"]:
        for metric, value in _numbers("", entry["metrics"]).items():
            _put(metrics, slice_metric_name(entry["key"], entry["value"], metric), value)
        _put(metrics, slice_metric_name(entry["key"], entry["value"], "n"), float(entry["n"]))
    return metrics


def stability_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    """S6c2's summary: ``stability/<variant>/<figure>`` and ``stability/<a>_to_<b>/<figure>``.

    A pair is named ``as_published->greyscale`` in the file; ``>`` is not a legal metric character,
    so the arrow becomes ``_to_``.
    """
    parts = [_numbers(f"stability/{metric_part(v)}/", stats) for v, stats in summary["variants"].items()]
    parts += [
        _numbers(f"stability/{metric_part(pair.replace('->', '_to_'))}/", stats)
        for pair, stats in summary["pairs"].items()
    ]
    metrics = _merge(*parts)
    _put(metrics, "drawings", float(summary["drawings"]))
    if "seconds" in summary:
        _put(metrics, "seconds", float(summary["seconds"]))
    return metrics


#: The occlusion summary's figures that are logged. The per-drawing maps are in the file, not here.
OCCLUSION_FIGURES = ("top3_share_median", "top3_share_undefined", "inked_patches_median", "max_drop_median")


def _prompt_names(result: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """``prompt:<template>`` perturbation → (``prompt_<n>``, template), numbered in file order.

    One function for both the metrics and the params, so the number on a metric always names the
    template in the param of the same number.
    """
    keys = [name for name in result["perturbations"] if name.startswith("prompt:")]
    return {key: (f"prompt_{n}", key.removeprefix("prompt:")) for n, key in enumerate(keys)}


def sensitivity_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    """S6f's summary: ``sensitivity/<perturbation>/<figure>``, ``sensitivity/occlusion/<figure>``
    and ``baseline_top5_acc``.

    The names are the ones S6f's own logger used for runs ``d99d75b5`` and ``dfaee7e2``, so those
    runs stay comparable with every later one. A prompt is ``prompt_<n>``, because a template has
    ``{``, ``}`` and spaces in it; the template itself is the param of the same number. An
    occlusion figure that is ``None`` (no map defined it) is left out, not logged as 0.
    """
    prompts = _prompt_names(result)
    parts = [{"baseline_top5_acc": float(result["baseline_top5_acc"])}]
    for name, figures in result["perturbations"].items():
        segment = prompts[name][0] if name in prompts else metric_part(name)
        parts.append(_numbers(f"sensitivity/{segment}/", figures))
    occluded = result["occlusion"]
    parts.append(
        _numbers(
            "sensitivity/occlusion/",
            {key: occluded[key] for key in OCCLUSION_FIGURES if occluded.get(key) is not None},
        )
    )
    return _merge(*parts)


def sensitivity_params(result: Mapping[str, Any]) -> dict[str, str]:
    """What S6f's file records beyond `run_params`: the templates, the sample and its seed, the render."""
    params = {number: template for number, template in _prompt_names(result).values()}
    params |= {"sample": str(result["drawings"]), "seed": str(result["seed"])}
    if "shipped_prompt_template" in result:
        params["shipped_prompt_template"] = str(result["shipped_prompt_template"])
    if "render" in result:
        params["render"] = json.dumps(result["render"], sort_keys=True)
    return params


def log_run(
    *,
    kind: str,
    result: Mapping[str, Any],
    metrics_path: Path,
    owner: str,
    run_name: str,
    card_path: Path = CARD_PATH,
) -> str:
    """Log one run copying ``metrics_path`` and return its id.

    ``kind`` is ``accuracy``, ``stability`` or ``sensitivity``. The experiment is
    ``MLFLOW_EXPERIMENT_NAME`` if set, else ``yip-drawing``; the server is ``MLFLOW_TRACKING_URI``,
    which for the shared server is the IAP tunnel (`docs/runbooks/mlflow-server.md`).
    """
    if kind not in BUILDERS:
        raise ValueError(f"kind must be one of {sorted(BUILDERS)}, not {kind!r}")
    build_metrics, build_params = BUILDERS[kind]
    # Built before the import and before any run starts, so a malformed file fails without leaving
    # an empty run on the server.
    tags = run_tags(result, owner=owner, kind=kind)
    params = run_params(result)
    for key, value in build_params(result).items():
        # The same key from both builders must agree: two values for one param would be a silent
        # choice of one of them.
        if key in params and params[key] != value:
            raise ValueError(f"param {key!r} is {params[key]!r} in the file's settings and {value!r} here")
        params[key] = value
    metrics = build_metrics(result)

    import mlflow

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", EXPERIMENT))
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="eval")
        mlflow.log_artifact(str(card_path))
        return str(run.info.run_id)


def _no_params(result: Mapping[str, Any]) -> dict[str, str]:
    return {}


#: Each kind of file: how its metrics are named, and the params it records beyond `run_params`.
BUILDERS: dict[
    str,
    tuple[Callable[[Mapping[str, Any]], dict[str, float]], Callable[[Mapping[str, Any]], dict[str, str]]],
] = {
    "accuracy": (accuracy_metrics, _no_params),
    "stability": (stability_metrics, _no_params),
    "sensitivity": (sensitivity_metrics, sensitivity_params),
}
