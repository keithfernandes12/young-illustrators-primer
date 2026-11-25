import pytest


@pytest.fixture(scope="session")
def guard(load_script):
    return load_script(".claude/hooks/guard_bash.py", "guard_bash")


@pytest.fixture(scope="session")
def session_start(load_script):
    return load_script(".claude/hooks/session_start.py", "session_start")


@pytest.fixture(scope="session")
def commit_msg(load_script):
    return load_script(".githooks/commit_msg.py", "commit_msg")


@pytest.fixture(scope="session")
def check_pr(load_script):
    return load_script("scripts/check_pr.py", "check_pr")
