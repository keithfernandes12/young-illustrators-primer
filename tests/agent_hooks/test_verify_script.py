"""verify.py is the PR's headline deliverable and had no test of its own.

It is exercised on every commit, which is why its happy path is safe. The parts nothing touches
are the two that were wrong before: `changed_tests()` (the `--quick` selection) and the
prerequisite branch, which exists only because verify once printed PASS while silently skipping a
required gate. That is exactly the kind of fix that regresses back to a false PASS unnoticed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".claude" / "skills" / "verify" / "scripts" / "verify.py"
spec = importlib.util.spec_from_file_location("verify", SCRIPT)
assert spec and spec.loader
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_root_resolves_to_the_repository():
    """ROOT is four parents up; a moved script would silently check the wrong tree."""
    assert verify.ROOT == REPO_ROOT
    assert (verify.ROOT / "pyproject.toml").exists()
    assert verify.PRIVACY.exists(), "the privacy gate verify runs must exist"


def test_quick_selection_maps_source_changes_to_test_areas(monkeypatch):
    """`--quick` picks tests near the diff. Mapping src/primer/<area>/ to tests/<area>/ is the rule."""
    diff = "src/primer/data/profiling.py\ntests/agent_hooks/test_guard_cloud.py\nREADME.md\n"

    class Result:
        stdout = diff

    monkeypatch.setattr(verify.subprocess, "run", lambda *a, **k: Result())
    selected = verify.changed_tests()

    assert "tests/agent_hooks/test_guard_cloud.py" in [s.replace("\\", "/") for s in selected]
    assert any(s.replace("\\", "/") == "tests/data" for s in selected), selected
    # A changed README selects nothing, and nothing outside tests/ is ever returned as a path.
    assert all("README" not in s for s in selected)


def test_quick_selection_ignores_tests_that_no_longer_exist(monkeypatch):
    """A deleted test file in the diff must not be handed to pytest as an argument."""

    class Result:
        stdout = "tests/does_not_exist/test_gone.py\n"

    monkeypatch.setattr(verify.subprocess, "run", lambda *a, **k: Result())
    assert verify.changed_tests() == []


def test_a_missing_executable_is_a_failed_prerequisite_not_a_skip(monkeypatch, capsys):
    """Terraform files present but no terraform binary must fail, not quietly drop the gate.

    Before this, verify reported PASS without having run a required check.
    """
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    monkeypatch.setattr(verify.Path, "rglob", lambda self, pattern: iter([Path("infra/main.tf")]))
    monkeypatch.setattr(verify, "run", lambda name, cmd, cwd=None: True)
    monkeypatch.setattr(verify, "changed_tests", list)

    exit_code = verify.main([])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "VERIFY_RESULT: FAIL" in output
    assert "prerequisite" in output
    assert "terraform" in output


def test_every_check_runs_even_after_one_fails(monkeypatch, capsys):
    """verify reports every gate rather than stopping at the first failure."""
    attempted: list[str] = []

    def fake_run(name, cmd, cwd=None):
        attempted.append(name)
        return name != "ruff"

    monkeypatch.setattr(verify, "run", fake_run)
    monkeypatch.setattr(verify, "changed_tests", list)
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    monkeypatch.setattr(verify.Path, "rglob", lambda self, pattern: iter([]))

    exit_code = verify.main([])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert {"ruff", "mypy", "pytest"} <= set(attempted), attempted
    assert "VERIFY_RESULT: FAIL (1 failing: ruff)" in output


def test_quick_says_it_ran_a_subset(monkeypatch, capsys):
    """A --quick PASS must not read as a full PASS."""
    monkeypatch.setattr(verify, "run", lambda name, cmd, cwd=None: True)
    monkeypatch.setattr(verify, "changed_tests", list)
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    monkeypatch.setattr(verify.Path, "rglob", lambda self, pattern: iter([]))

    assert verify.main(["--quick"]) == 0
    output = capsys.readouterr().out
    assert "NOTE" in output and "subset" in output
    assert "VERIFY_RESULT: PASS" in output


@pytest.mark.parametrize("flag", ["--no-privacy"])
def test_privacy_can_be_turned_off_explicitly(monkeypatch, flag):
    """The gate is skippable only by saying so on the command line, never silently."""
    names: list[str] = []
    monkeypatch.setattr(verify, "run", lambda name, cmd, cwd=None: names.append(name) or True)
    monkeypatch.setattr(verify, "changed_tests", list)
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    monkeypatch.setattr(verify.Path, "rglob", lambda self, pattern: iter([]))

    verify.main([flag])
    assert "privacy" not in names
