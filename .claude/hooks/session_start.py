#!/usr/bin/env python3
"""Runs when a Claude Code session starts (see .claude/settings.json).

Whatever this prints is added to the agent's context. It does two things:

1. Gives the agent its session id, to record in the PR's "Agent provenance" section, so a bad
   change can be traced back to the session (and prompt) that produced it. The id only resolves
   on the machine that ran the session; it does not publish the transcript.
2. Checks this clone's one-time setup and, if a step is missing, tells the agent to ask the person
   to run it. Teammates never have to read a setup document; the agent prompts them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SESSION_ID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
REPO_ROOT = Path(__file__).resolve().parents[2]


def provenance(session_id: str) -> str:
    line = f"- Claude Code session: {session_id}"
    return (
        f"Claude Code session id: {session_id}\n"
        "When you open a pull request or update its body, add this exact line under "
        "`## Agent provenance` if it is not already there, keep every earlier line, and remove "
        f"the `N/A` placeholder when adding the first one:\n{line}"
    )


def git_hooks_path(root: Path) -> str:
    try:
        result = subprocess.run(["git", "config", "--get", "core.hooksPath"], cwd=root,
                                capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def missing_setup(root: Path, hooks_path: str, has_cli: bool) -> list[str]:
    """One-time steps this clone still needs, as commands for the person to run."""
    steps = []
    if hooks_path != ".githooks":
        steps.append("`git config core.hooksPath .githooks` (turns on the commit-message check)")
    if not has_cli:
        steps.append('`pip install -e ".[dev,dvc]"` in the project\'s virtual environment '
                     "(installs `primer-data`, pytest, ruff, mypy)")
    return steps


def setup_message(steps: list[str]) -> str:
    if not steps:
        return ""
    listed = "\n".join(f"- {step}" for step in steps)
    return (
        "This clone is missing one-time setup. Before other work, tell the person and ask them to run "
        f"(they can type `!` followed by the command in this session):\n{listed}\n"
        "If Jira tools are unavailable, ask them to run `/mcp` and authenticate `atlassian`."
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    parts = []
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    if isinstance(session_id, str) and SESSION_ID.fullmatch(session_id):
        parts.append(provenance(session_id))
    steps = missing_setup(REPO_ROOT, git_hooks_path(REPO_ROOT), shutil.which("primer-data") is not None)
    if steps:
        parts.append(setup_message(steps))
    if parts:
        print("\n\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
