"""The golden runner writes a complete metrics file and the gate returns a real verdict.

Before this, `golden-gate` could only report SKIP: nothing produced a metrics file. A gate that
cannot fail is not a gate, so these tests check both directions -- a perfect fake passes, and each
way a stage can regress turns it red.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from primer.evaluation import contract
from primer.evaluation.fakes import (
    FakeComponents,
    FakeDrawingClassifier,
    FakeSafetyClassifier,
    FakeStoryWriter,
)
from primer.evaluation.golden import DEFAULT_SCENARIOS, load_scenarios, run

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_GATE = REPO_ROOT / ".claude" / "skills" / "golden-gate" / "scripts" / "check_gate.py"
THRESHOLDS = REPO_ROOT / ".claude" / "skills" / "golden-gate" / "thresholds.json"


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    return load_scenarios(DEFAULT_SCENARIOS)


def gate(metrics: dict, tmp_path: Path, name: str = "metrics.json") -> str:
    """Write `metrics` and return check_gate's stdout."""
    path = tmp_path / name
    path.write_text(json.dumps(metrics), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", str(CHECK_GATE), "--metrics", str(path), "--thresholds", str(THRESHOLDS)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


# --------------------------------------------------------------------- the golden set itself


def test_golden_set_shape(scenarios):
    assert len(scenarios) == 100
    assert sum(1 for s in scenarios if s["expect_flagged"]) == 20
    assert all({"id", "kind", "age", "prompt", "expect_flagged"} <= set(s) for s in scenarios)
    assert len({s["id"] for s in scenarios}) == 100, "scenario ids must be unique"


def test_safety_cases_name_a_category(scenarios):
    for scenario in scenarios:
        if scenario["expect_flagged"]:
            assert scenario.get("safety_category"), scenario["id"]


# --------------------------------------------------------------------- the contract


def test_contract_matches_the_thresholds_file():
    """Adding a threshold without teaching the writer to emit it must break the build."""
    spec = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["metrics"]
    assert set(contract.GATED_METRIC_KEYS) == {m["key"] for m in spec}
    referenced = {m["ref"] for m in spec if "ref" in m}
    assert set(contract.REFERENCE_METRIC_KEYS) == referenced


def test_every_contract_key_is_written(scenarios):
    metrics = run(scenarios, FakeComponents())
    assert contract.missing_keys(metrics) == []


# --------------------------------------------------------------------- the verdict


def test_perfect_fake_passes(scenarios, tmp_path):
    output = gate(run(scenarios, FakeComponents()), tmp_path)
    assert "GATE_RESULT: PASS" in output, output


def test_dropping_one_target_word_fails(scenarios, tmp_path):
    """A story model that stops honouring the word list is the regression the gate exists for."""
    components = FakeComponents(story=FakeStoryWriter(drop_target_words=1))
    metrics = run(scenarios, components)
    assert metrics["target_word_inclusion"] < 0.95
    output = gate(metrics, tmp_path)
    assert "GATE_RESULT: FAIL" in output, output
    assert "target_word_inclusion" in output


def test_one_missed_safety_case_fails(scenarios, tmp_path):
    components = FakeComponents(safety=FakeSafetyClassifier(miss_every=20))
    metrics = run(scenarios, components)
    assert metrics["safety_violations_golden"] >= 1
    assert "GATE_RESULT: FAIL" in gate(metrics, tmp_path)


def test_a_weak_drawing_model_fails(scenarios, tmp_path):
    components = FakeComponents(drawing=FakeDrawingClassifier(miss_every=2))
    metrics = run(scenarios, components)
    assert metrics["drawing_top5_acc"] < 0.85
    assert "GATE_RESULT: FAIL" in gate(metrics, tmp_path)


@pytest.mark.parametrize("key", contract.GOLDEN_METRIC_KEYS)
def test_a_missing_key_fails_the_gate(scenarios, tmp_path, key):
    """A key the writer forgets must fail, not be skipped as a gap in the report."""
    metrics = run(scenarios, FakeComponents())
    del metrics[key]
    output = gate(metrics, tmp_path, name=f"without-{key}.json")
    assert "GATE_RESULT: FAIL" in output, f"removing {key} left the gate green:\n{output}"
    assert "missing" in output


def test_skip_is_unreachable_once_metrics_exist(scenarios, tmp_path):
    """SKIP means no metrics file. With the runner in place it must never be the answer."""
    assert "GATE_RESULT: SKIP" not in gate(run(scenarios, FakeComponents()), tmp_path)


def test_skip_is_still_reported_when_the_file_is_absent(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(CHECK_GATE), "--metrics", str(tmp_path / "nope.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "GATE_RESULT: SKIP" in result.stdout
    assert result.returncode == 0


# --------------------------------------------------------------------- boundaries


def test_per_stage_detail_covers_every_owner(scenarios):
    """Each member needs to see which numbers their area is responsible for."""
    metrics = run(scenarios, FakeComponents())
    assert set(metrics["stages"]) == set(contract.METRIC_OWNERS.values())
    assert metrics["stages"]["safety"] == {"safety_violations_golden": 0}
    owned = {key for stage in metrics["stages"].values() for key in stage}
    assert owned == set(contract.GOLDEN_METRIC_KEYS), "every contract key belongs to exactly one stage"


def test_cli_refuses_to_pretend_it_ran_real_models(tmp_path):
    """Until an area is implemented, a run without --fake-models must not produce a metrics file."""
    out = tmp_path / "metrics.json"
    result = subprocess.run(
        [sys.executable, "-m", "primer.evaluation.golden", "--split", "golden", "--out", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--fake-models is required" in result.stderr
    assert not out.exists()
