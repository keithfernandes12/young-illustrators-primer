"""The deploy scripts must work under Git Bash on Windows, where `python3` is a Store alias stub.

`_common.sh` parsed gcloud's JSON with a hardcoded `python3`. On a Windows laptop that resolves to
the Microsoft Store app-execution alias, which prints "Python was not found" and exits non-zero, so
tag_url() and serving_revision() returned empty strings and every smoke test failed. The repo's own
hooks already pick an interpreter; these scripts now do the same.
"""

import json
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
    assert "YIP_SHARED_PROJECT" in text, "the served service lives in the shared team project"
    assert "YIP_PROJECT" in text, "the single-project fallback must keep working"
    # The staging/prod split is gone; there was never a prod project.
    assert "YIP_PROD_PROJECT" not in text


# ------------------------------------------------- findings from the Copilot review of PR #11

CANARY = SCRIPTS / "deploy_canary.sh"
PROMOTE = SCRIPTS / "promote.sh"
SMOKE = SCRIPTS / "smoke_test.sh"


def code_of(script: Path) -> str:
    """The script without comment lines.

    These scripts explain in comments why they do *not* call `update-traffic` or `--to-latest`, so
    a plain substring search finds the prose and not the behaviour.
    """
    return "\n".join(
        line for line in script.read_text(encoding="utf-8").splitlines() if not line.strip().startswith("#")
    )


def test_the_canary_never_shifts_traffic():
    """It used to send 10% of live requests, which guard_cloud.py now refuses outright."""
    code = code_of(CANARY)
    assert "update-traffic" not in code, "the canary must not move traffic at all"
    assert "--no-traffic" in code and "--tag canary" in code
    assert "CANARY_PERCENT: 0" in code


def test_the_canary_fails_closed_when_the_service_cannot_be_described():
    """A describe failure can be permissions or a transient error, not just an absent service.

    The old fallback deployed at 100% traffic in every one of those cases.
    """
    text = CANARY.read_text(encoding="utf-8")
    assert "exit 1" in text
    # The old bootstrap branch deployed without --no-traffic. It must be gone.
    assert "does not exist yet; first deploy takes 100%" not in text


def test_the_canary_records_the_revision_it_tagged():
    assert "tagged_revision" in CANARY.read_text(encoding="utf-8")
    assert "tagged_revision()" in COMMON.read_text(encoding="utf-8").replace("tagged_revision() {", "tagged_revision()")


def test_promote_targets_the_tested_revision_not_the_latest():
    """`--to-latest` promotes whatever is newest, which may not be what was smoke-tested."""
    code = code_of(PROMOTE)
    assert "--to-latest" not in code
    assert "--to-revisions" in code
    assert ".canary" in code, "it must read the revision deploy_canary.sh recorded"
    assert "refusing" in code, "it must refuse when the canary tag moved since the test"


def test_smoke_test_fails_when_its_fixture_is_missing():
    """A missing fixture used to skip the POST, so /healthz alone could report PASS."""
    text = SMOKE.read_text(encoding="utf-8")
    assert "missing smoke fixture" in text
    assert "SMOKE_RESULT: FAIL" in text


def test_the_smoke_fixture_exists_and_carries_no_child_data():
    fixture = REPO_ROOT / "tests" / "fixtures" / "smoke_book_request.json"
    assert fixture.exists(), "smoke_test.sh references this file"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    assert data["options"]["dry_run"] is True
    # A pseudonymous id and a reference, never a name or an uploaded image.
    assert set(data["child"]) == {"child_id", "age"}
    assert data["drawing"]["drawing_ref"].startswith("fixture://")


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
    assert "YIP_SHARED_PROJECT" in result.stderr


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
def test_shared_project_is_preferred_over_the_fallbacks(tmp_path):
    """With both set, the deploy scripts must target the shared project, not the compute one."""
    result = subprocess.run(
        [BASH, str(COMMON)],
        cwd=tmp_path,
        env={
            "PATH": BASH_PATH,
            "YIP_SHARED_PROJECT": "young-illustrators-primer",
            "YIP_COMPUTE_PROJECT": "member-dev",
            "YIP_PROJECT": "should-not-win",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert "project=young-illustrators-primer" in result.stderr, result.stderr
    assert "member-dev" not in result.stderr
    assert "should-not-win" not in result.stderr


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
