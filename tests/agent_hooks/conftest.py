import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(relative_path: str, name: str):
    """Import a script that is not part of the package (hooks live under .claude/ and .githooks/).

    An explicit loader is needed because git hooks have no .py extension.
    """
    path = str(REPO_ROOT / relative_path)
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def guard():
    return load_script(".claude/hooks/guard_bash.py", "guard_bash")


@pytest.fixture(scope="session")
def session_start():
    return load_script(".claude/hooks/session_start.py", "session_start")


@pytest.fixture(scope="session")
def commit_msg():
    return load_script(".githooks/commit_msg.py", "commit_msg")


@pytest.fixture(scope="session")
def check_pr():
    return load_script("scripts/check_pr.py", "check_pr")
