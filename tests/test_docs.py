"""Every document in docs/<type>/ follows the frontmatter rules (scripts/check_docs.py)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_docs", ROOT / "scripts" / "check_docs.py")
check_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_docs)

GOOD = """---
title: Example plan
type: plan
created: 2026-10-25
status: active
author: rishiparekh
jira: [KAN-42]
related: [other-plan]
---

# Example plan
"""


def test_repository_docs_are_valid():
    assert check_docs.check() == []


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    write(root, "plans/other-plan.md", GOOD.replace("related: [other-plan]\n", ""))
    write(root, "plans/_template.md", "no frontmatter needed in templates")
    write(root, "plans/AGENTS.md", "no frontmatter needed here")
    write(root, "project_scoping.md", "course submission, exempt")
    return root


def test_a_valid_document_passes(docs):
    write(docs, "plans/example-plan.md", GOOD)
    assert check_docs.check(docs) == []


@pytest.mark.parametrize("rel, text, expected", [
    ("plans/Example_Plan.md", GOOD, "file name"),
    ("plans/2026-10-25-plan.md", GOOD, "file name"),
    ("plans/001-example.md", GOOD, "file name"),
    ("plans/example-plan.md", GOOD.replace("rishiparekh", "{{GITHUB_USERNAME}}"), "unfilled placeholder"),
    ("meeting-notes/weekly-sync.md", GOOD.replace("type: plan", "type: meeting-note").replace("active", "final"), "file name"),
    ("plans/example-plan.md", "# no frontmatter\n", "missing frontmatter"),
    ("plans/example-plan.md", GOOD.replace("author: rishiparekh\n", ""), "missing required field `author`"),
    ("plans/example-plan.md", GOOD.replace("type: plan", "type: adr"), "type must be `plan`"),
    ("plans/example-plan.md", GOOD.replace("status: active", "status: shipped"), "is not one of"),
    ("plans/example-plan.md", GOOD.replace("created: 2026-10-25", "created: last week"), "must be a date"),
    ("plans/example-plan.md", GOOD.replace("author: rishiparekh", "author: Rishi Parekh"), "GitHub username"),
    ("plans/example-plan.md", GOOD.replace("[KAN-42]", "[sc-42]"), "KAN-42"),
    ("plans/example-plan.md", GOOD.replace("[other-plan]", "[other-plan.md]"), "bare slug"),
    ("plans/example-plan.md", GOOD.replace("jira:", "owner: x\njira:"), "unknown field `owner`"),
    ("plans/example-plan.md", GOOD.replace("status: active", "status: superseded"), "superseded_by"),
])
def test_problems_are_reported(docs, rel, text, expected):
    write(docs, rel, text)
    problems = check_docs.check(docs)
    assert any(expected in p for p in problems), problems


def test_dated_meeting_note_name_is_accepted(docs):
    note = GOOD.replace("type: plan", "type: meeting-note").replace("status: active", "status: final")
    write(docs, "meeting-notes/2026-10-25-weekly-sync.md", note)
    assert check_docs.check(docs) == []
