"""Compare the small pretrained models over a temperature x max_new_tokens grid, and pick one.

    python -m primer.story.grid --config src/primer/story/configs/story_grid.yaml --owner <github-user>

For every model and every grid point it runs the validation requests, logs **one MLflow run** to the
local git-ignored store (tags as `model-development` section 2 lists), then applies the selection rule
fixed in the config **before** any run, and writes to `out`:

- `metrics.json` -- the contract shape: the selected configuration's gated keys in `overall`, the
  named baseline configuration's in `baseline`, per-slice rows, and every metric in `details`;
- `grid_results.json` -- every run's params and metrics, plus the sensitivity spreads;
- `charts/*.png` -- the comparison bar chart and the temperature sweep (if matplotlib is installed).

Run on the member's Mac (MPS), never in CI and never as a cloud job. The model loader and the logger
are injectable, which is how `tests/story/test_grid.py` runs the whole grid with a stub model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from primer.story.data import ValidationRequest, read_requests
from primer.story.evaluate import metrics_file, run_requests, summarise
from primer.story.model import HFChatBackend, ModelSpec
from primer.story.sensitivity import explain
from primer.story.slices import slice_metrics
from primer.story.writer import ChatBackend, GenerationSettings, LLMStoryWriter

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class GridConfig:
    raw: dict[str, Any]

    @property
    def models(self) -> list[ModelSpec]:
        return [ModelSpec(**m) for m in self.raw["models"]]

    def points(self) -> list[tuple[ModelSpec, float, int]]:
        grid = self.raw["grid"]
        return [(m, float(t), int(n)) for m in self.models
                for t, n in itertools.product(grid["temperature"], grid["max_new_tokens"])]


def load_config(path: Path) -> GridConfig:
    return GridConfig(yaml.safe_load(path.read_text(encoding="utf-8")))


class RunLogger(Protocol):
    def log_run(self, name: str, tags: dict[str, str], params: dict[str, Any], metrics: dict[str, float],
                artifact: dict[str, Any]) -> str:
        """Record one run; return its id."""
        ...


class MlflowLogger:
    """Logs to MLflow. Imported lazily: `mlflow` comes from the `story` extra."""

    def __init__(self, tracking_uri: str, experiment: str) -> None:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._mlflow = mlflow

    def log_run(self, name: str, tags: dict[str, str], params: dict[str, Any], metrics: dict[str, float],
                artifact: dict[str, Any]) -> str:
        with self._mlflow.start_run(run_name=name) as active:
            self._mlflow.set_tags(tags)
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
            self._mlflow.log_metrics(metrics)
            self._mlflow.log_dict(artifact, "results.json")
            return str(active.info.run_id)


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _flat_metrics(summary: dict[str, float | None], slices: list[dict[str, Any]]) -> dict[str, float]:
    """MLflow metrics: every summary value, plus `slice/<key>=<value>/<metric>` (with `=` made safe)."""
    flat = {k: float(v) for k, v in summary.items() if v is not None}
    for row in slices:
        for metric, value in row["metrics"].items():
            if value is not None:
                flat[f"slice/{row['key']}_{row['value']}/{metric}"] = float(value)
    return flat


def select(runs: Sequence[dict[str, Any]], rule: dict[str, Any], size_rank: dict[str, int]) -> dict[str, Any] | None:
    """Apply the config's selection rule. Returns the chosen run, or None if no run is eligible.

    1. Eligible: `success_rate >= min_success_rate` and `returned_safety_violation_rate <= max_...`.
    2. Best `primary` metric.
    3. Among eligible runs within `tie_margin` of the best, order by each tie-breaker in turn
       (higher is better, except `p95_latency_s` and `model_size_rank`, where lower is better).
    """
    eligible = [r for r in runs
                if (r["metrics"].get("success_rate") or 0.0) >= rule["min_success_rate"]
                and (r["metrics"].get("returned_safety_violation_rate") or 0.0)
                <= rule["max_returned_safety_violation_rate"]]
    if not eligible:
        return None
    primary = rule["primary"]
    best = max(r["metrics"][primary] for r in eligible)
    contenders = [r for r in eligible if r["metrics"][primary] >= best - rule["tie_margin"]]
    lower_is_better = {"p95_latency_s", "model_size_rank"}

    def key(run: dict[str, Any]) -> tuple[float, ...]:
        parts = []
        for breaker in rule["tie_breakers"]:
            if breaker == "model_size_rank":
                value = float(size_rank[run["params"]["model"]])
            else:
                value = float(run["metrics"].get(breaker) or 0.0)
            parts.append(value if breaker in lower_is_better else -value)
        return tuple(parts)

    return min(contenders, key=key)


def run_grid(
    config: GridConfig,
    requests: Sequence[ValidationRequest],
    backend_for: Callable[[ModelSpec], ChatBackend],
    logger: RunLogger,
    owner: str,
    sha: str,
) -> dict[str, Any]:
    """Run every grid point, log each, select, and return the full report (nothing written to disk)."""
    raw = config.raw
    backends: dict[str, ChatBackend] = {}
    runs: list[dict[str, Any]] = []
    for spec, temperature, max_new_tokens in config.points():
        backend = backends.setdefault(spec.name, backend_for(spec))
        settings = GenerationSettings(temperature=temperature, max_new_tokens=max_new_tokens,
                                      top_p=float(raw["top_p"]), seed=int(raw["seed"]))
        writer = LLMStoryWriter(backend=backend, settings=settings, max_attempts=int(raw["max_attempts"]))
        results = run_requests(writer, requests)
        summary = summarise(results)
        slices = slice_metrics(results, summarise)
        params = {"model": spec.name, "temperature": temperature, "max_new_tokens": max_new_tokens,
                  "revision": spec.revision, "top_p": raw["top_p"], "seed": raw["seed"],
                  "max_attempts": raw["max_attempts"], "n_requests": len(requests)}
        tags = {"stage": "story", "owner": owner, "data_version": str(raw["data_version"]), "git_sha": sha,
                "base_model": f"{spec.repo}@{spec.revision}", "gate_group": "story", "where": "local",
                "level_source": str(raw["level_source"])}
        name = f"{spec.name}-t{temperature}-n{max_new_tokens}"
        run_id = logger.log_run(name, tags, params, _flat_metrics(summary, slices),
                                {"results": [asdict(r) for r in results], "slices": slices})
        runs.append({"name": name, "run_id": run_id, "params": params, "metrics": summary, "slices": slices})

    size_rank = {m.name: i for i, m in enumerate(config.models)}
    chosen = select(runs, raw["selection"], size_rank)
    base = raw["baseline"]
    baseline = next((r for r in runs if r["params"]["model"] == base["model"]
                     and r["params"]["temperature"] == float(base["temperature"])
                     and r["params"]["max_new_tokens"] == int(base["max_new_tokens"])), None)
    if baseline is None:
        raise ValueError("the baseline configuration is not one of the grid points")
    reference = chosen or baseline
    model = next(m for m in config.models if m.name == reference["params"]["model"])
    metrics = metrics_file(
        model_uri=f"hf://{model.repo}@{model.revision}", data_version=str(raw["data_version"]), git_sha=sha,
        overall=reference["metrics"], baseline=baseline["metrics"], slices=reference["slices"],
        details={"selected": chosen["name"] if chosen else None, "selected_run_id": chosen and chosen["run_id"],
                 "baseline_config": baseline["name"], "metrics": reference["metrics"],
                 "selection_rule": raw["selection"]},
    )
    return {"runs": runs, "selected": chosen["name"] if chosen else None, "baseline": baseline["name"],
            "metrics": metrics, "sensitivity": explain(runs)}


def write_outputs(report: dict[str, Any], out: Path) -> list[Path]:
    """metrics.json, grid_results.json and (if matplotlib is present) the charts."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report["metrics"], indent=2) + "\n", encoding="utf-8")
    (out / "grid_results.json").write_text(json.dumps(
        {k: report[k] for k in ("runs", "selected", "baseline", "sensitivity")}, indent=2) + "\n",
        encoding="utf-8")
    written = [out / "metrics.json", out / "grid_results.json"]
    try:
        from primer.story.charts import write_charts
    except ImportError:  # matplotlib is in the `story` extra only
        return written
    return written + write_charts(report["runs"], out / "charts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "story_grid.yaml")
    parser.add_argument("--owner", required=True, help="GitHub username, for the MLflow `owner` tag")
    parser.add_argument("--requests", type=Path, help="overrides the config's requests file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    requests = read_requests(args.requests or Path(config.raw["requests"]))
    logger = MlflowLogger(str(config.raw["mlflow_tracking_uri"]), str(config.raw["experiment"]))
    report = run_grid(config, requests, HFChatBackend, logger, owner=args.owner, sha=git_sha())
    for path in write_outputs(report, Path(config.raw["out"])):
        print(path)
    print(f"selected: {report['selected']} (baseline: {report['baseline']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
