"""The CI pull-request check (scripts/check_pr.py)."""

import pytest


def test_a_good_pr_passes(check_pr):
    assert check_pr.problems("feat(data): add x (KAN-42)", "## Jira\n- KAN-42", ["feat(data): add x (KAN-42)"]) == []


@pytest.mark.parametrize("title, body, subjects, expected", [
    ("Add x", "KAN-42", [], "PR title"),
    ("feat(data): add x (KAN-42)", "no key here", [], "no Jira key"),
    ("feat(data): add x (KAN-42)", None, [], "no Jira key"),
    ("feat(data): add x (KAN-42)", "KAN-42", ["wip"], "Commit subject"),
])
def test_problems(check_pr, title, body, subjects, expected):
    assert any(expected in p for p in check_pr.problems(title, body, subjects))


def _git(repo, *args):
    import subprocess
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                           *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, subject):
    _git(repo, "commit", "--allow-empty", "--no-verify", "-m", subject)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path):
    """base -> two pre-format commits (the second is the exempt tip) -> two newer commits."""
    _git(tmp_path, "init", "-q")
    base = _commit(tmp_path, "chore: base (KAN-1)")
    _commit(tmp_path, "Add old thing")
    tip = _commit(tmp_path, "Add older format thing")
    _commit(tmp_path, "feat(data): new thing (KAN-2)")
    head = _commit(tmp_path, "bad new subject")
    return tmp_path, base, tip, head


def test_commits_up_to_the_exempt_tip_are_not_checked(check_pr, history):
    repo, base, tip, head = history
    assert check_pr.commit_subjects(base, head, cwd=repo, exempt_tip=tip) == [
        "bad new subject", "feat(data): new thing (KAN-2)"]
    found = check_pr.problems("feat(data): x (KAN-2)", "KAN-2",
                              check_pr.commit_subjects(base, head, cwd=repo, exempt_tip=tip))
    assert found == ["Commit subject does not follow the format: 'bad new subject'"]   # still enforced after


def test_an_unknown_exempt_tip_exempts_nothing(check_pr, history):
    repo, base, _, head = history
    subjects = check_pr.commit_subjects(base, head, cwd=repo, exempt_tip="0" * 40)
    assert len(subjects) == 4 and "Add old thing" in subjects


def test_the_real_exempt_tip_is_the_dvc_versioning_commit(check_pr):
    """Guards against editing the constant to a different commit by mistake."""
    assert check_pr.PRE_FORMAT_TIP == "6240f42e3a5041aed15fa7d12b5332a45b176ed6"
