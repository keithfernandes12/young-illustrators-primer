"""The deploy scripts must work under Git Bash on Windows, where `python3` is a Store alias stub.

`_common.sh` parsed gcloud's JSON with a hardcoded `python3`. On a Windows laptop that resolves to
the Microsoft Store app-execution alias, which prints "Python was not found" and exits non-zero, so
tag_url() and serving_revision() returned empty strings and every smoke test failed. The repo's own
hooks already pick an interpreter; these scripts now do the same.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / ".claude" / "skills" / "deploy-cloud-run" / "scripts"
COMMON = SCRIPTS / "_common.sh"


def find_bash() -> str | None:
    """Git Bash on Windows, /bin/bash elsewhere.

    `bash` on the Windows PATH is usually C:\\Windows\\System32\\bash.exe, the WSL launcher, which
    fails with a store error when no distribution is installed. Claude Code has the same problem and
    solves it with CLAUDE_CODE_GIT_BASH_PATH, so that is honoured here too.
    """
    if sys.platform != "win32":
        return shutil.which("bash")
    override = os.environ.get("CLAUDE_CODE_GIT_BASH_PATH")
    candidates = [override] if override else []
    candidates += [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


BASH = find_bash()
BASH_PATH = str(Path(BASH).parent) if BASH else ""   # so the scripts can find git
bash_required = pytest.mark.skipif(BASH is None, reason="needs Git Bash")


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
@bash_required
def test_scripts_parse(script):
    result = subprocess.run([BASH, "-n", str(SCRIPTS / script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_no_script_pipes_into_a_bare_python3():
    """A bare `python3` is the bug; `PY=python3` followed by a fallback is the fix."""
    for script in SCRIPTS.glob("*.sh"):
        text = script.read_text(encoding="utf-8")
        assert "| python3" not in text, script.name
        assert "python3 -c" not in text, script.name


def test_common_defines_an_interpreter_fallback():
    text = COMMON.read_text(encoding="utf-8")
    assert 'PY=python3' in text
    assert '"$PY" -c "" 2>/dev/null || PY=python' in text


def test_common_uses_the_single_project_variable():
    text = COMMON.read_text(encoding="utf-8")
    assert "YIP_PROJECT" in text
    # The staging/prod split is gone: there is one project.
    assert "YIP_PROD_PROJECT" not in text


@bash_required
def test_common_refuses_to_run_without_a_project(tmp_path):
    result = subprocess.run(
        [BASH, str(COMMON)],
        cwd=tmp_path,
        env={"PATH": BASH_PATH},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "YIP_PROJECT" in result.stderr


@bash_required
def test_common_reports_the_project_it_will_use(tmp_path):
    result = subprocess.run(
        [BASH, str(COMMON)],
        cwd=tmp_path,
        env={"PATH": BASH_PATH, "YIP_PROJECT": "young-illustrators-primer"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert "project=young-illustrators-primer" in result.stderr
    assert result.returncode == 0


@bash_required
def test_older_staging_variable_still_works(tmp_path):
    result = subprocess.run(
        [BASH, str(COMMON)],
        cwd=tmp_path,
        env={"PATH": BASH_PATH, "YIP_STAGING_PROJECT": "young-illustrators-primer"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert "project=young-illustrators-primer" in result.stderr
