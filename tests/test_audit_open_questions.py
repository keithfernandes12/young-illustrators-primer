"""The open-questions audit (scripts/audit_open_questions.py, cross-slice-coherence S8)."""

import io
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audit(load_script):
    return load_script("scripts/audit_open_questions.py", "audit_open_questions")


PLAN = """---
title: A plan
---

# A plan

## Slices

- S1 is a slice, not a question, and is never read.

## Open questions

- **Which region?** Recommended: us-central1.
  *Decides:* the owner. Tracked as KAN-12.
- **Who owns the cache?** Answered in S2 of this plan.
- **Is the threshold right?** Nobody has said.
  It continues on a second line.

## Out of scope

- A bare bullet here is not an open question.
"""


def _plans(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_only_the_question_with_neither_key_nor_slice_is_untracked(audit, tmp_path):
    found = audit.audit(_plans(tmp_path, {"a-plan.md": PLAN}))

    assert [q.tracked for q in found] == [True, True, False]
    untracked = found[2]
    assert untracked.text.startswith("**Is the threshold right?**")
    assert "second line" in untracked.text, "continuation lines belong to their bullet"
    assert untracked.line == 16, "the bullet's first line, for a clickable path:line"


def test_giving_a_question_a_key_removes_it_from_the_list(audit, tmp_path):
    keyed = PLAN.replace("Nobody has said.", "Nobody has said. KAN-99.")

    found = audit.audit(_plans(tmp_path, {"a-plan.md": keyed}))

    assert all(q.tracked for q in found)


def test_bullets_outside_open_questions_and_the_folder_rules_are_ignored(audit, tmp_path):
    root = _plans(tmp_path, {"a-plan.md": PLAN, "AGENTS.md": PLAN, "_template.md": PLAN})

    assert {q.plan for q in audit.audit(root)} == {"a-plan.md"}
    assert len(audit.audit(root)) == 3, "Slices and Out of scope bullets are not questions"


def test_a_plan_with_no_open_questions_section_has_none(audit, tmp_path):
    assert audit.audit(_plans(tmp_path, {"b.md": "# B\n\n## Slices\n\n- S1\n"})) == []


def test_the_cli_always_exits_zero_and_json_lists_only_the_untracked(audit, tmp_path, capsys):
    root = _plans(tmp_path, {"a-plan.md": PLAN})

    assert audit.main(["--plans", str(root)]) == 0
    text = capsys.readouterr().out
    assert "a-plan.md:16:" in text and "1 of 3 open questions" in text

    assert audit.main(["--plans", str(root), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["total"] == 3 and [q["line"] for q in report["untracked"]] == [16]


def test_it_runs_on_the_real_plans(audit):
    # Not a count assertion: plans change every week. It must parse every plan and find questions.
    found = audit.audit()
    assert found and all(q.plan.endswith(".md") for q in found)


def test_a_tab_indented_continuation_belongs_to_its_bullet(audit, tmp_path):
    plan = "# P\n\n## Open questions\n\n- **Who decides?**\n\tSee KAN-7 for the owner.\n"

    found = audit.audit(_plans(tmp_path, {"a-plan.md": plan}))

    assert len(found) == 1 and found[0].tracked


def test_text_mode_survives_a_console_that_cannot_encode_the_question(audit, tmp_path, monkeypatch):
    plan = "# P\n\n## Open questions\n\n- **Which → which?** Nobody has said.\n"
    root = _plans(tmp_path, {"a-plan.md": plan})
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", console)

    assert audit.main(["--plans", str(root)]) == 0

    console.flush()
    assert b"Which ? which?" in console.buffer.getvalue()


def test_an_empty_audit_warns_on_stderr_and_still_exits_zero(audit, tmp_path, capsys):
    assert audit.main(["--plans", str(tmp_path / "no-such-folder")]) == 0

    captured = capsys.readouterr()
    assert "no open questions found" in captured.err
    assert "0 of 0 open questions" in captured.out
