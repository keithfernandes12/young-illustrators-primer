"""The guards also cover the PowerShell tool, which Claude Code uses on Windows for claude.ai accounts.

Without this, a Windows session could run `gh pr merge` through PowerShell and neither the hooks
nor the Bash(...) deny rules would see it.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))


def test_powershell_tool_turned_off_for_the_project():
    assert SETTINGS["env"]["CLAUDE_CODE_USE_POWERSHELL_TOOL"] == "0"


def test_shell_guards_match_powershell_too():
    assert SETTINGS["hooks"]["PreToolUse"][0]["matcher"] == "Bash|PowerShell"


def test_every_bash_deny_rule_has_a_powershell_twin():
    deny = SETTINGS["permissions"]["deny"]
    for rule in deny:
        if rule.startswith("Bash("):
            inner = rule[len("Bash("):-1].removesuffix(":*")
            assert f"PowerShell({inner} *)" in deny, rule


def test_guard_bash_refuses_merge_from_powershell():
    payload = {"tool_name": "PowerShell", "tool_input": {"command": "gh pr merge 3"}, "cwd": str(REPO_ROOT)}
    result = subprocess.run([sys.executable, str(REPO_ROOT / ".claude" / "hooks" / "guard_bash.py")],
                            input=json.dumps(payload), capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 2, result
    assert "Agents never merge" in result.stderr
