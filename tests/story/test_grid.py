"""The grid end to end with stub models: runs, MLflow tags, the selection rule, and outputs."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from story_fakes import WordEchoBackend

from primer.story.data import ValidationRequest, write_requests
from primer.story.grid import (
    GridConfig,
    config_params,
    dvc_tags,
    finish,
    limit_requests,
    load_config,
    request_inputs,
    requests_tags,
    run_grid,
    select,
    write_outputs,
)
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

    def log_files(self, name, tags, paths):
        self.runs.append({"name": name, "tags": tags, "files": list(paths)})
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
    report = run_grid(small_config(), REQUESTS, backend_for, logger, owner="tester", sha="abc1234", clock=itertools.count().__next__)
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
    report = run_grid(small_config(), REQUESTS, backend_for, RecordingLogger(), owner="t", sha="s", clock=itertools.count().__next__)
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
        run_grid(config, REQUESTS, backend_for, RecordingLogger(), owner="t", sha="s", clock=itertools.count().__next__)


def test_sensitivity_spreads():
    runs = [{"params": {"model": "a", "temperature": 0.3, "max_new_tokens": 64}, "metrics": {"success_rate": 1.0}},
            {"params": {"model": "a", "temperature": 0.7, "max_new_tokens": 64}, "metrics": {"success_rate": 0.6}}]
    spread = explain(runs)
    assert spread["temperature"]["success_rate"] == 0.4
    assert spread["model"]["success_rate"] is None, "one value cannot have a spread"


def test_outputs_are_written(tmp_path):
    report = run_grid(small_config(), REQUESTS, backend_for, RecordingLogger(), owner="t", sha="s", clock=itertools.count().__next__)
    written = write_outputs(report, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["stage"] == "story" and "target_word_inclusion" in metrics["overall"]
    grid = json.loads((tmp_path / "grid_results.json").read_text())
    assert grid["selected"] == report["selected"] and "sensitivity" in grid
    assert tmp_path / "metrics.json" in written


def test_charts_are_drawn_when_matplotlib_is_installed(tmp_path):
    pytest.importorskip("matplotlib", reason="matplotlib comes from the `story` extra; CI omits it")
    from primer.story.charts import write_charts

    report = run_grid(small_config(), REQUESTS, backend_for, RecordingLogger(), owner="t", sha="s", clock=itertools.count().__next__)
    paths = write_charts(report["runs"], tmp_path)
    assert [p.name for p in paths] == ["comparison.png", "temperature_sweep.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_importing_the_story_area_loads_no_ml_framework():
    """Weights and heavy libraries load lazily: serving imports every area, and so do these tests."""
    code = ("import sys, primer.story.model, primer.story.grid, primer.story.writer, primer.story.interface; "
            "print(sorted(m for m in ('torch', 'transformers', 'mlflow', 'matplotlib') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120)
    assert out.stdout.strip() == "[]"


def test_the_request_path_does_not_import_primer_data():
    """Review F13: nothing on the request path may import primer.data (model-development section 2).

    Only the modules serving would load -- not grid or data, which are offline and do import it.
    """
    code = ("import sys, primer.story.model, primer.story.writer, primer.story.interface, primer.story.checks; "
            "print(sorted(m for m in sys.modules if m == 'primer.data' or m.startswith('primer.data.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120)
    assert out.stdout.strip() == "[]"


def test_a_safety_hungry_configuration_is_not_eligible():
    """Review Q2: the raw flag rate is the safety filter that can actually exclude a run today."""
    rule = load_config(CONFIG).raw["selection"]
    chosen = select([run("tries-unsafe", "small", target_word_inclusion=1.0, raw_safety_flag_rate=0.5),
                     run("clean", "big", target_word_inclusion=0.8, raw_safety_flag_rate=0.0)], rule, SIZES)
    assert chosen["name"] == "clean"


def test_no_eligible_run_leaves_overall_empty_and_exits_non_zero(tmp_path):
    """Review F6: the baseline's numbers must never be reported as if a model had been selected."""
    config = small_config()
    config.raw["selection"] = config.raw["selection"] | {"min_success_rate": 1.01}  # nothing can pass
    logger = RecordingLogger()
    report = run_grid(config, REQUESTS, backend_for, logger, owner="t", sha="s", clock=itertools.count().__next__)
    assert report["selected"] is None
    assert report["metrics"]["overall"] == {}
    assert "target_word_inclusion" in report["metrics"]["not_measured"]
    assert report["metrics"]["baseline"]["target_word_inclusion"] == 1.0, "the baseline is still reported"
    assert finish(report, logger, tmp_path, "t", "s", {}) == 2


def test_metrics_and_charts_are_logged_as_one_summary_run(tmp_path):
    """Review F3: metrics.json (and the charts, when drawn) are MLflow artifacts, not only local files."""
    logger = RecordingLogger()
    report = run_grid(small_config(), REQUESTS, backend_for, logger, owner="t", sha="s",
                      clock=itertools.count().__next__)
    assert finish(report, logger, tmp_path, "t", "s", {"dvc/stories_text": "abc"}) == 0
    summary = logger.runs[-1]
    assert summary["name"] == "grid-summary" and summary["tags"]["kind"] == "grid-summary"
    assert REQUIRED_TAGS <= set(summary["tags"]), REQUIRED_TAGS - set(summary["tags"])  # review F16
    assert summary["tags"]["base_model"].startswith("hf://Qwen/")
    assert summary["tags"]["dvc/stories_text"] == "abc"
    assert tmp_path / "metrics.json" in summary["files"]


def test_the_requests_scored_are_named_by_hash_and_logged(tmp_path):
    """PR #36 review F5: every run names the requests file (hash, how many used, the TinyStories subset),
    and the summary run carries the file and its manifest as artifacts."""
    requests_path = tmp_path / "validation_requests.jsonl"
    write_requests(REQUESTS, requests_path)
    manifest = requests_path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"tinystories_files": [{"path": "p", "generation": "179"}]}))
    tags = requests_tags(requests_path, used=3)
    assert tags["requests_sha256"] == hashlib.sha256(requests_path.read_bytes()).hexdigest()[:16]
    assert tags["requests_used"] == "3"
    assert tags["tinystories_files"] == "1" and tags["tinystories_generations"] == "179"

    logger = RecordingLogger()
    report = run_grid(small_config(), REQUESTS, backend_for, logger, owner="t", sha="s", extra_tags=tags,
                      clock=itertools.count().__next__)
    finish(report, logger, tmp_path / "out", "t", "s", tags, inputs=request_inputs(requests_path))
    assert all(run["tags"]["requests_sha256"] == tags["requests_sha256"] for run in logger.runs)
    assert {requests_path, manifest} <= set(logger.runs[-1]["files"])


def test_a_changed_requests_file_changes_its_hash(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_requests(REQUESTS, first)
    write_requests(REQUESTS[:2], second)
    assert requests_tags(first, 4)["requests_sha256"] != requests_tags(second, 2)["requests_sha256"]
    assert requests_tags(first, 4)["tinystories_files"] == "unknown", "no manifest beside it"


def test_main_wires_the_requests_tags_into_every_run(tmp_path, monkeypatch):
    """The wiring in grid.main, not just the helpers: every logged run carries the requests file's
    hash. MlflowLogger and HFChatBackend are swapped for the recording logger and the stub model."""
    import yaml

    from primer.story import grid

    requests_path = tmp_path / "requests.jsonl"
    write_requests(REQUESTS, requests_path)
    config = small_config().raw | {"requests": str(requests_path), "out": str(tmp_path / "out"),
                                   "max_requests": 3}
    config_path = tmp_path / "grid.yaml"
    config_path.write_text(yaml.safe_dump(config))
    logger = RecordingLogger()
    monkeypatch.setattr(grid, "MlflowLogger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(grid, "HFChatBackend", backend_for)

    assert grid.main(["--config", str(config_path), "--owner", "t"]) == 0
    expected = requests_tags(requests_path, 3)["requests_sha256"]
    assert logger.runs and all(run["tags"]["requests_sha256"] == expected for run in logger.runs)
    assert all(run["tags"]["requests_used"] == "3" for run in logger.runs)
    assert requests_path in logger.runs[-1]["files"], "the summary run carries the requests file"


def test_every_run_logs_the_whole_config_and_the_extra_tags():
    """Review Q3 and the plan's DVC tags: params carry the whole config; extra tags reach every run."""
    logger = RecordingLogger()
    run_grid(small_config(), REQUESTS, backend_for, logger, owner="t", sha="s",
             extra_tags={"dvc/word_lists": "x"}, clock=itertools.count().__next__)
    for logged in logger.runs:
        assert json.loads(logged["params"]["config/selection"])["primary"] == "target_word_inclusion"
        assert "config/models" in logged["params"] and logged["tags"]["dvc/word_lists"] == "x"


def test_the_backend_is_created_once_per_model():
    """Review F11: setdefault called the factory at every grid point."""
    made: list[str] = []

    def counting(spec):
        made.append(spec.name)
        return backend_for(spec)

    run_grid(small_config(), REQUESTS, counting, RecordingLogger(), owner="t", sha="s",
             clock=itertools.count().__next__)
    assert sorted(made) == ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"]


def test_the_shipped_grid_is_the_reduced_mac_grid():
    """The hardware-constrained grid recorded in CARD.md: 2 models x 2 temperatures x 1 token limit."""
    config = load_config(CONFIG)
    assert len(config.points()) == 4
    assert config.raw["max_requests"] == 20


def test_limit_requests_takes_a_prefix():
    assert limit_requests(REQUESTS, 2) == REQUESTS[:2]
    assert limit_requests(REQUESTS, None) == REQUESTS
    assert limit_requests(REQUESTS, 99) == REQUESTS
    with pytest.raises(ValueError):
        limit_requests(REQUESTS, 0)


def test_config_params_and_dvc_tags_shapes():
    params = config_params({"name": "x", "grid": {"temperature": [0.3]}})
    assert params == {"name": "x", "grid": '{"temperature": [0.3]}'}
    tags = dvc_tags("v1.1")
    assert set(tags) == {"dvc/stories_text", "dvc/storybook_pages", "dvc/word_lists"}
    for key, value in tags.items():
        dataset = key.removeprefix("dvc/")
        exists = subprocess.run(["git", "cat-file", "-e", f"data-v1.1:data/{dataset}.dvc"], cwd=REPO_ROOT,
                                capture_output=True, check=False).returncode == 0
        # Where the pointer exists at the tag, a real hash is required (review F18): "unavailable"
        # everywhere would otherwise pass. A clone without the tag may still say "unavailable".
        assert re.fullmatch(r"[0-9a-f]{12}", value) if exists else value == "unavailable", (key, value)
    assert dvc_tags("v0.0-no-such-version") == dict.fromkeys(tags, "unavailable")
