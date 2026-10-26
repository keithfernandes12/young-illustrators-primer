"""The hook commands in .claude/settings.json, run the way Claude Code runs them (through bash).

Testing the Python functions is not enough: an earlier `python3 … || python …` command printed
"BLOCKED" but exited 0, because the fallback re-ran the hook with its input already consumed, so
the refusal never reached Claude Code. These tests run the real command strings.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt",
                                reason="needs a POSIX shell to fake the Windows python3 stub")


def hook_command(event: str) -> str:
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    return settings["hooks"][event][0]["hooks"][0]["command"]


@pytest.fixture(params=["normal", "windows-python3-stub"])
def path_env(request, tmp_path):
    """PATH with a working python3, or with a broken python3 (like the Windows Store stub) and a working python."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if request.param == "normal":
        (bin_dir / "python3").symlink_to(sys.executable)
    else:
        stub = bin_dir / "python3"
        stub.write_text("#!/bin/sh\necho 'Python was not found' >&2\nexit 49\n")
        stub.chmod(0o755)
        (bin_dir / "python").symlink_to(sys.executable)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "CLAUDE_PROJECT_DIR": str(REPO_ROOT)}


def run(event: str, payload: dict, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", hook_command(event)], input=json.dumps(payload), env=env,
                          capture_output=True, text=True, timeout=30, check=False)


def test_guard_refusal_reaches_claude_code(path_env):
    result = run("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 3"}, "cwd": "."}, path_env)
    assert result.returncode == 2, result
    assert "Agents never merge" in result.stderr


def test_guard_allows_ordinary_commands(path_env):
    result = run("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "pytest -q"}, "cwd": "."}, path_env)
    assert result.returncode == 0, result


def test_session_start_prints_once(path_env):
    session = "65a9edd9-522a-48f8-bbc4-24b92bd4eaa5"
    result = run("SessionStart", {"session_id": session}, path_env)
    assert result.returncode == 0, result
    assert result.stdout.count(f"- Claude Code session: {session}") == 1
