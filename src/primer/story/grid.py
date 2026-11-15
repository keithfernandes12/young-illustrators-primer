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
import hashlib
import itertools
import json
import subprocess
import sys
import time
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

    def log_files(self, name: str, tags: dict[str, str], paths: Sequence[Path]) -> str:
        """Record a run holding `paths` as artifacts (the grid's metrics.json and charts); return its id."""
        ...


class MlflowLogger:
    """Logs to MLflow. Imported lazily: `mlflow` comes from the `story` extra.

    MLflow 3 refuses the plain `file:` store (maintenance mode), so the local store is SQLite. With a
    database backend the artifact root must be given when the experiment is created, or MLflow puts
    artifacts in `./mlruns` under whatever directory the run started from.
    """

    def __init__(self, tracking_uri: str, experiment: str, artifact_location: str | None = None) -> None:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            location = Path(artifact_location).resolve().as_uri() if artifact_location else None
            mlflow.create_experiment(experiment, artifact_location=location)
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

    def log_files(self, name: str, tags: dict[str, str], paths: Sequence[Path]) -> str:
        with self._mlflow.start_run(run_name=name) as active:
            self._mlflow.set_tags(tags)
            for path in paths:
                self._mlflow.log_artifact(str(path), artifact_path=path.parent.name if path.parent.name == "charts"
                                          else None)
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

    1. Eligible: `success_rate >= min_success_rate`, `returned_safety_violation_rate <= max_...` and
       `raw_safety_flag_rate <= max_raw_safety_flag_rate`. The returned-violation filter is zero by
       construction while the judge is the writer's own guardrail, so on its own it could never
       exclude a run (review Q2); the raw flag rate -- how often the model *tried* to write something
       unsafe -- is the filter that bites until an independent classifier exists.
    2. Best `primary` metric.
    3. Among eligible runs within `tie_margin` of the best, order by each tie-breaker in turn
       (higher is better, except `p95_latency_s` and `model_size_rank`, where lower is better).
    """
    eligible = [r for r in runs
                if (r["metrics"].get("success_rate") or 0.0) >= rule["min_success_rate"]
                and (r["metrics"].get("returned_safety_violation_rate") or 0.0)
                <= rule["max_returned_safety_violation_rate"]
                and (r["metrics"].get("raw_safety_flag_rate") or 0.0) <= rule.get("max_raw_safety_flag_rate", 1.0)]
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


def limit_requests(requests: Sequence[ValidationRequest], n: int | None) -> list[ValidationRequest]:
    """The first `n` requests (all of them when `n` is None). The file is written round-robin by source,
    with ages cycling within each source, so a prefix keeps both balanced."""
    if n is not None and n < 1:
        raise ValueError("max_requests must be at least 1")
    return list(requests if n is None else requests[:n])


def config_params(raw: dict[str, Any]) -> dict[str, str]:
    """The whole config as MLflow params (review Q3; model-development section 2): scalars as text,
    lists and mappings (models, grid, baseline, selection rule) as JSON."""
    return {key: value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            for key, value in raw.items()}


def dvc_tags(data_version: str) -> dict[str, str]:
    """`dvc/<dataset>` tags: a short SHA-1 of each data pointer file as committed at the `data-<version>`
    git tag, which pins the exact object versions the data came from. "unavailable" if git cannot
    show it (no such tag in this clone)."""
    tags = {}
    for dataset in ("stories_text", "storybook_pages", "word_lists"):
        try:
            out = subprocess.run(["git", "show", f"data-{data_version}:data/{dataset}.dvc"], cwd=ROOT,
                                 capture_output=True, check=False, timeout=10)
        except (OSError, subprocess.SubprocessError):
            tags[f"dvc/{dataset}"] = "unavailable"
            continue
        tags[f"dvc/{dataset}"] = hashlib.sha1(out.stdout).hexdigest()[:12] if out.returncode == 0 else "unavailable"
    return tags


def run_grid(
    config: GridConfig,
    requests: Sequence[ValidationRequest],
    backend_for: Callable[[ModelSpec], ChatBackend],
    logger: RunLogger,
    owner: str,
    sha: str,
    extra_tags: dict[str, str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run every grid point, log each, select, and return the full report (nothing written to disk).

    `clock` times each request; tests pass a fake one so latency, and anything chosen by latency, is
    deterministic (review F12).
    """
    raw = config.raw
    backends: dict[str, ChatBackend] = {}
    runs: list[dict[str, Any]] = []
    base_params = config_params(raw)
    for spec, temperature, max_new_tokens in config.points():
        if spec.name not in backends:  # one backend per model, created once (review F11)
            backends[spec.name] = backend_for(spec)
        backend = backends[spec.name]
        settings = GenerationSettings(temperature=temperature, max_new_tokens=max_new_tokens,
                                      top_p=float(raw["top_p"]), seed=int(raw["seed"]))
        writer = LLMStoryWriter(backend=backend, settings=settings, max_attempts=int(raw["max_attempts"]))
        results = run_requests(writer, requests, clock=clock)
        summary = summarise(results)
        slices = slice_metrics(results, summarise)
        params = {"model": spec.name, "temperature": temperature, "max_new_tokens": max_new_tokens,
                  "revision": spec.revision, "top_p": raw["top_p"], "seed": raw["seed"],
                  "max_attempts": raw["max_attempts"], "n_requests": len(requests)}
        tags = {"stage": "story", "owner": owner, "data_version": str(raw["data_version"]), "git_sha": sha,
                "base_model": f"{spec.repo}@{spec.revision}", "gate_group": "story", "where": "local",
                "level_source": str(raw["level_source"]), **(extra_tags or {})}
        name = f"{spec.name}-t{temperature}-n{max_new_tokens}"
        run_id = logger.log_run(name, tags, {**{f"config/{k}": v for k, v in base_params.items()}, **params},
                                _flat_metrics(summary, slices),
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
    # With no eligible run, `overall` stays empty, so the gate sees the story keys as missing rather
    # than reading the baseline's numbers as if a model had been selected (review F6).
    if chosen is not None:
        model = next(m for m in config.models if m.name == chosen["params"]["model"])
        model_uri, overall, slices = f"hf://{model.repo}@{model.revision}", chosen["metrics"], chosen["slices"]
    else:
        model_uri, overall, slices = "none: no configuration passed the selection rule", {}, []
    metrics = metrics_file(
        model_uri=model_uri, data_version=str(raw["data_version"]), git_sha=sha,
        overall=overall, baseline=baseline["metrics"], slices=slices,
        details={"selected": chosen["name"] if chosen else None,
                 "selected_run_id": chosen["run_id"] if chosen else None,
                 "baseline_config": baseline["name"], "baseline_run_id": baseline["run_id"],
                 "metrics": chosen["metrics"] if chosen else None, "baseline_metrics": baseline["metrics"],
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
    requests = limit_requests(read_requests(args.requests or Path(config.raw["requests"])),
                              config.raw.get("max_requests"))
    print(f"{len(requests)} requests x {len(config.points())} configurations")
    logger = MlflowLogger(str(config.raw["mlflow_tracking_uri"]), str(config.raw["experiment"]),
                          config.raw.get("mlflow_artifact_location"))
    sha = git_sha()
    tags = dvc_tags(str(config.raw["data_version"]))
    report = run_grid(config, requests, HFChatBackend, logger, owner=args.owner, sha=sha, extra_tags=tags)
    return finish(report, logger, Path(config.raw["out"]), args.owner, sha, tags)


def finish(report: dict[str, Any], logger: RunLogger, out: Path, owner: str, sha: str,
           extra_tags: dict[str, str]) -> int:
    """Write the outputs, log them as one `grid-summary` run (review F3), print the outcome.

    Exits 2 when no configuration passed the selection rule, so a script cannot mistake that for success.
    """
    paths = write_outputs(report, out)
    # The section 2 tags too (review F16): this is the run that holds metrics.json.
    summary_tags = {"stage": "story", "owner": owner, "git_sha": sha, "gate_group": "story", "where": "local",
                    "data_version": str(report["metrics"]["data_version"]),
                    "base_model": str(report["metrics"]["model_uri"]),
                    "kind": "grid-summary", "selected": str(report["selected"]), "baseline": report["baseline"],
                    **extra_tags}
    summary_id = logger.log_files("grid-summary", summary_tags, paths)
    for path in paths:
        print(path)
    print(f"summary run: {summary_id}")
    if report["selected"] is None:
        print(f"no configuration passed the selection rule (baseline: {report['baseline']})")
        return 2
    print(f"selected: {report['selected']} (baseline: {report['baseline']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
