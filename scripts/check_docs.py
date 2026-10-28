#!/usr/bin/env python3
"""Checks the frontmatter of every document under docs/<type>/.

Usage:  python scripts/check_docs.py        (exit 1 and a list of problems if any)

The rules are written out for people in docs/engineering/documentation.md. This script is what
keeps them true: tests/test_docs.py runs it, so CI fails on a malformed document. Without a check
like this, status values and field names drift within weeks (one team's plans ended up with
`completed`, `shipped` and `draft-speculative` alongside the official values).

Files that are skipped: AGENTS.md, README.md, templates (names starting with `_`), and documents
directly in docs/ (the course submissions such as docs/project_scoping.md).
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parents[1] / "docs"

# folder -> (type value, allowed status values)
TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "plans": ("plan", ("draft", "active", "closed", "superseded")),
    "adrs": ("adr", ("draft", "accepted", "deprecated", "superseded")),
    "briefs": ("brief", ("draft", "decided", "parked")),
    "runbooks": ("runbook", ("active", "deprecated")),
    "solutions": ("solution", ("active", "deprecated")),
    "engineering": ("engineering", ("active", "deprecated")),
    "meeting-notes": ("meeting-note", ("draft", "final")),
}
REQUIRED = ("title", "type", "created", "status", "author")
OPTIONAL = ("updated", "jira", "tags", "related", "supersedes", "superseded_by")
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
DATED_SLUG = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(-[a-z0-9]+)*$")
NUMBER_PREFIX = re.compile(r"^\d+-")  # dates (2026-10-25-…) and manual numbers (001-…)
PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")
AUTHOR = re.compile(r"^[A-Za-z0-9](-?[A-Za-z0-9])*$")
JIRA = re.compile(r"^KAN-\d+$")
SKIPPED = {"AGENTS.md", "README.md"}


def documents(root: Path) -> list[Path]:
    found = []
    for folder in TYPES:
        for path in sorted((root / folder).rglob("*.md")):
            if path.name not in SKIPPED and not path.name.startswith("_"):
                found.append(path)
    return found


def frontmatter(text: str) -> dict | None:
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.DOTALL)
    if not match:
        return None
    data = yaml.safe_load(match.group(1))
    return data if isinstance(data, dict) else None


def is_date(value: object) -> bool:
    if isinstance(value, dt.date):
        return True
    if isinstance(value, str):
        try:
            dt.date.fromisoformat(value)
            return True
        except ValueError:
            return False
    return False


def as_list(value: object) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def check_document(path: Path, root: Path, slugs: set[str]) -> list[str]:
    rel = path.relative_to(root.parent).as_posix()
    folder = path.relative_to(root).parts[0]
    expected_type, statuses = TYPES[folder]
    problems: list[str] = []

    if folder == "meeting-notes":
        if not DATED_SLUG.match(path.stem):
            problems.append(f"{rel}: file name must be YYYY-MM-DD-kebab-case")
    elif not SLUG.match(path.stem) or NUMBER_PREFIX.match(path.stem):
        problems.append(f"{rel}: file name must be a kebab-case slug with no date or number prefix")

    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER.search(text):
        problems.append(f"{rel}: has an unfilled placeholder ({PLACEHOLDER.search(text).group(0)})")
        return problems
    try:
        meta = frontmatter(text)
    except yaml.YAMLError as exc:
        return problems + [f"{rel}: frontmatter is not valid YAML ({exc.__class__.__name__})"]
    if meta is None:
        return problems + [f"{rel}: missing frontmatter (a --- block at the very top)"]

    for key in REQUIRED:
        if meta.get(key) in (None, ""):
            problems.append(f"{rel}: missing required field `{key}`")
    for key in meta:
        if key not in REQUIRED and key not in OPTIONAL:
            problems.append(f"{rel}: unknown field `{key}` (allowed: {', '.join(REQUIRED + OPTIONAL)})")

    if meta.get("type") not in (None, expected_type):
        problems.append(f"{rel}: type must be `{expected_type}` in docs/{folder}/")
    if meta.get("status") not in (None, *statuses):
        problems.append(f"{rel}: status `{meta.get('status')}` is not one of: {', '.join(statuses)}")
    for key in ("created", "updated"):
        if meta.get(key) is not None and not is_date(meta[key]):
            problems.append(f"{rel}: `{key}` must be a date like 2026-10-25")
    if meta.get("author") is not None and not AUTHOR.match(str(meta["author"])):
        problems.append(f"{rel}: `author` must be a GitHub username, not a display name")
    for key in as_list(meta.get("jira")):
        if not JIRA.match(str(key)):
            problems.append(f"{rel}: `jira` entries look like KAN-42 (got {key!r})")
    for key in ("related", "supersedes", "superseded_by"):
        for target in as_list(meta.get(key)):
            if str(target) not in slugs:
                problems.append(f"{rel}: `{key}` names {target!r}, which is not the bare slug of a document in docs/")
    if meta.get("status") == "superseded" and not meta.get("superseded_by"):
        problems.append(f"{rel}: a superseded document must name its successor in `superseded_by`")
    return problems


def check(root: Path = DOCS) -> list[str]:
    paths = documents(root)
    slugs = {p.stem for p in paths}
    problems: list[str] = []
    for path in paths:
        problems.extend(check_document(path, root, slugs))
    return problems


def main() -> int:
    problems = check()
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} problem(s). Rules: docs/engineering/documentation.md")
        return 1
    print(f"docs: {len(documents(DOCS))} document(s) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
