#!/usr/bin/env python3
"""Checks a pull request's title, body and commit subjects (run by .github/workflows/pr-rules.yml).

Reads PR_TITLE, PR_BODY, BASE_SHA and HEAD_SHA from the environment. Uses the same subject rule as
the local commit-msg hook (.githooks/commit_msg.py), so the two cannot disagree. The title is checked
too, because a squash merge turns the PR title into the commit on main.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JIRA_KEY = re.compile(r"\bKAN-\d+\b")


def load_commit_msg():
    path = str(ROOT / ".githooks" / "commit_msg.py")
    loader = importlib.machinery.SourceFileLoader("commit_msg", path)
    spec = importlib.util.spec_from_file_location("commit_msg", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commit_subjects(base: str, head: str) -> list[str]:
    out = subprocess.run(["git", "log", "--no-merges", "--format=%s", f"{base}..{head}"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def problems(title: str, body: str, subjects: list[str]) -> list[str]:
    rule = load_commit_msg()
    found: list[str] = []
    if rule.problem(title):
        found.append(f"PR title does not follow `type(scope): summary (KAN-<n>)`: {title!r}")
    if not JIRA_KEY.search(body or ""):
        found.append("PR body has no Jira key (KAN-<n>) — fill in the Jira section of the template.")
    for subject in subjects:
        if rule.problem(subject):
            found.append(f"Commit subject does not follow the format: {subject!r}")
    return found


def main() -> int:
    subjects = commit_subjects(os.environ["BASE_SHA"], os.environ["HEAD_SHA"])
    found = problems(os.environ.get("PR_TITLE", ""), os.environ.get("PR_BODY", ""), subjects)
    for line in found:
        print(f"::error::{line}")
    if found:
        print("\nRules: AGENTS.md -> Git and pull requests. Fix commit subjects with `git rebase -i` "
              "(or the commit-reorganize skill) and push with --force-with-lease.")
        return 1
    print(f"PR title, body and {len(subjects)} commit subject(s) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
