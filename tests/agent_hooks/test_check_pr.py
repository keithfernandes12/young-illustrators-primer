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
