"""The grid end to end with stub models: runs, MLflow tags, the selection rule, and outputs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from story_fakes import WordEchoBackend

from primer.story.data import ValidationRequest
from primer.story.grid import GridConfig, load_config, run_grid, select, write_outputs
from primer.story.sensitivity import explain

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "src" / "primer" / "story" / "configs" / "story_grid.yaml"
REQUIRED_TAGS = {"stage", "owner", "data_version", "git_sha", "base_model", "gate_group", "where"}


class RecordingLogger:
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def log_run(self, name, tags, params, metrics, artifact):
        self.runs.append({"name": name, "tags": tags, "params": params, "metrics": metrics, "artifact": artifact})
        return f"run-{len(self.runs)}"


def small_config(**overrides) -> GridConfig:
    raw = load_config(CONFIG).raw | {"grid": {"temperature": [0.3, 0.7], "max_new_tokens": [64]}}
    raw["baseline"] = {"model": "qwen2.5-0.5b-instruct", "temperature": 0.7, "max_new_tokens": 64}
    raw.update(overrides)
    return GridConfig(raw)


REQUESTS = [ValidationRequest(request_id=f"r{i}", book_key=f"tinystories/{i}", source="tinystories", age=3 + i % 3,
                              target_words=("ball", "cat"), character_name=None, setting="the park",
                              reading_level=0.5, book_level=2.0, level_source="fk_formula@1", license="x", attribution="y")
            for i in range(4)]


def backend_for(spec):
    """The 0.5B stub fumbles its first JSON every time; the 1.5B stub never does."""
    return WordEchoBackend(invalid_first=spec.name.startswith("qwen2.5-0.5b"))


def test_the_shipped_config_is_well_formed():
    config = load_config(CONFIG)
    assert all(re.fullmatch(r"[0-9a-f]{40}", m.revision) for m in config.models), "revisions must be pinned"
    base = config.raw["baseline"]
    assert (base["model"], float(base["temperature"]), int(base["max_new_tokens"])) in {
        (m.name, t, n) for m, t, n in config.points()}, "the baseline must be one of the grid points"
    uri = config.raw["mlflow_tracking_uri"]
    assert uri.startswith("sqlite:///"), "MLflow 3 refuses the file store; the local store is SQLite"
    for path in (uri.removeprefix("sqlite:///"), config.raw["mlflow_artifact_location"]):
        ignored = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT, check=False)
        assert ignored.returncode == 0, f"{path} must be git-ignored: runs are local working notes"


def test_every_grid_point_is_one_tagged_run():
    logger = RecordingLogger()
    report = run_grid(small_config(), REQUESTS, backend_for, logger, owner="tester", sha="abc1234")
    assert len(logger.runs) == 4  # 2 models x 2 temperatures x 1 token budget
    for run in logger.runs:
        assert REQUIRED_TAGS <= set(run["tags"]), REQUIRED_TAGS - set(run["tags"])
        assert run["tags"]["where"] == "local" and run["tags"]["level_source"] == "fk_formula@1"
        assert "@" in run["tags"]["base_model"], "base_model carries the pinned revision"
        assert "target_word_inclusion" in run["metrics"]
        assert any(key.startswith("slice/") for key in run["metrics"])
        assert all("=" not in key for key in run["metrics"]), "MLflow refuses '=' in metric names"
    assert len(report["runs"]) == 4


def test_the_rule_prefers_fewer_retries_when_inclusion_ties():
    """Both stubs reach 100% inclusion; the 1.5B never needs a retry, so the first tie-breaker picks it."""
    report = run_grid(small_config(), REQUESTS, backend_for, RecordingLogger(), owner="t", sha="s")
    assert report["selected"].startswith("qwen2.5-1.5b")
    assert report["baseline"] == "qwen2.5-0.5b-instruct-t0.7-n64"
    metrics = report["metrics"]
    assert metrics["overall"]["target_word_inclusion"] == 1.0
    assert metrics["baseline"]["target_word_inclusion"] == 1.0
    assert metrics["details"]["selected"] == report["selected"]


def run(name, model, **metrics):
    base = {"success_rate": 1.0, "returned_safety_violation_rate": 0.0, "target_word_inclusion": 0.9,
            "first_attempt_valid_json_rate": 0.5, "format_compliance_rate": 0.5, "p95_latency_s": 1.0}
    return {"name": name, "params": {"model": model}, "metrics": base | metrics}


RULE = load_config(CONFIG).raw["selection"]
SIZES = {"small": 0, "big": 1}


def test_the_primary_metric_wins_outside_the_tie_margin():
    chosen = select([run("a", "small", target_word_inclusion=0.90), run("b", "big", target_word_inclusion=0.95,
                                                                         first_attempt_valid_json_rate=0.0)],
                    RULE, SIZES)
    assert chosen["name"] == "b"


def test_ineligible_runs_are_never_chosen():
    chosen = select([run("unsafe", "small", target_word_inclusion=1.0, returned_safety_violation_rate=0.1),
                     run("flaky", "small", target_word_inclusion=1.0, success_rate=0.5),
                     run("ok", "big", target_word_inclusion=0.5)], RULE, SIZES)
    assert chosen["name"] == "ok"
    assert select([run("unsafe", "small", returned_safety_violation_rate=0.1)], RULE, SIZES) is None


def test_the_smaller_model_wins_a_complete_tie():
    assert select([run("b", "big"), run("a", "small")], RULE, SIZES)["name"] == "a"


def test_a_baseline_outside_the_grid_is_an_error():
    config = small_config(baseline={"model": "qwen2.5-0.5b-instruct", "temperature": 0.1, "max_new_tokens": 64})
    with pytest.raises(ValueError, match="baseline"):
        run_grid(config, REQUESTS, backend_for, RecordingLogger(), owner="t", sha="s")


def test_sensitivity_spreads():
    runs = [{"params": {"model": "a", "temperature": 0.3, "max_new_tokens": 64}, "metrics": {"success_rate": 1.0}},
            {"params": {"model": "a", "temperature": 0.7, "max_new_tokens": 64}, "metrics": {"success_rate": 0.6}}]
    spread = explain(runs)
    assert spread["temperature"]["success_rate"] == 0.4
    assert spread["model"]["success_rate"] is None, "one value cannot have a spread"


def test_outputs_are_written(tmp_path):
    report = run_grid(small_config(), REQUESTS, backend_for, RecordingLogger(), owner="t", sha="s")
    written = write_outputs(report, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["stage"] == "story" and "target_word_inclusion" in metrics["overall"]
    grid = json.loads((tmp_path / "grid_results.json").read_text())
    assert grid["selected"] == report["selected"] and "sensitivity" in grid
    assert tmp_path / "metrics.json" in written


def test_charts_are_drawn_when_matplotlib_is_installed(tmp_path):
    pytest.importorskip("matplotlib", reason="matplotlib comes from the `story` extra; CI omits it")
    from primer.story.charts import write_charts

    report = run_grid(small_config(), REQUESTS, backend_for, RecordingLogger(), owner="t", sha="s")
    paths = write_charts(report["runs"], tmp_path)
    assert [p.name for p in paths] == ["comparison.png", "temperature_sweep.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_importing_the_story_area_loads_no_ml_framework():
    """Weights and heavy libraries load lazily: serving imports every area, and so do these tests."""
    code = ("import sys, primer.story.model, primer.story.grid, primer.story.writer, primer.story.interface; "
            "print(sorted(m for m in ('torch', 'transformers', 'mlflow', 'matplotlib') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120)
    assert out.stdout.strip() == "[]"
