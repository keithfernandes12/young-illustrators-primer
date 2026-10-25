"""The session-start hook and the commit-message hook."""

import io
import json

import pytest

SESSION = "65a9edd9-522a-48f8-bbc4-24b92bd4eaa5"


def run_session_start(session_start, monkeypatch, capsys, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert session_start.main() == 0
    return capsys.readouterr().out


def test_session_start_prints_the_provenance_line(session_start, monkeypatch, capsys):
    out = run_session_start(session_start, monkeypatch, capsys, json.dumps({"session_id": SESSION}))
    assert f"- Claude Code session: {SESSION}" in out


@pytest.mark.parametrize("payload", ["not json", json.dumps({"session_id": "../etc"}), json.dumps([1])])
def test_session_start_ignores_a_bad_session_id(session_start, monkeypatch, capsys, payload):
    assert "Claude Code session" not in run_session_start(session_start, monkeypatch, capsys, payload)


def test_setup_check_lists_missing_steps(session_start, tmp_path):
    steps = session_start.missing_setup(tmp_path, hooks_path="", has_cli=False)
    assert any("core.hooksPath .githooks" in s for s in steps)
    assert any("pip install" in s for s in steps)
    assert "ask them to run" in session_start.setup_message(steps)


def test_setup_check_is_quiet_when_done(session_start, tmp_path):
    assert session_start.missing_setup(tmp_path, hooks_path=".githooks", has_cli=True) == []
    assert session_start.setup_message([]) == ""


@pytest.mark.parametrize("subject", [
    "feat(data): add Amateur Drawings ingestion (KAN-42)",
    "fix: handle an empty batch (KAN-7)",
    "chore(agents)!: replace the guard hook (KAN-100)",
    "Merge branch 'main' into feat/KAN-42-x",
    "Revert \"feat(data): add x (KAN-1)\"",
    "fixup! feat(data): add x (KAN-1)",
])
def test_commit_subject_accepted(commit_msg, subject):
    assert commit_msg.problem(subject) is None


@pytest.mark.parametrize("subject", [
    "Add dataset versioning with DVC (data v0.1)",
    "feat(data): add x",
    "feat(data): add x KAN-42",
    "Feat(data): add x (KAN-42)",
    "feature(data): add x (KAN-42)",
    "feat(Data): add x (KAN-42)",
    "feat(data):add x (KAN-42)",
    "",
])
def test_commit_subject_refused(commit_msg, subject):
    assert commit_msg.problem(subject)


def test_commit_msg_reads_the_first_real_line(commit_msg, tmp_path, capsys):
    good = tmp_path / "good"
    good.write_text("# comment\n\nfeat(data): add x (KAN-42)\n\nbody\n", encoding="utf-8")
    bad = tmp_path / "bad"
    bad.write_text("add x\n", encoding="utf-8")
    assert commit_msg.main(["commit-msg", str(good)]) == 0
    assert commit_msg.main(["commit-msg", str(bad)]) == 1
    assert "KAN-<n>" in capsys.readouterr().err
