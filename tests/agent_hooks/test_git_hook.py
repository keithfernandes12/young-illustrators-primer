"""The commit-msg hook, run by git itself, with a working python3 and with a broken one.

On Windows `python3` is often the Microsoft Store stub, which fails. A hook that started with
`#!/usr/bin/env python3` would then break every commit, so `.githooks/commit-msg` is a shell script
that picks a working interpreter first. These tests make git run it.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("git") is None or os.name == "nt",
                                reason="needs a POSIX shell to fake the Windows python3 stub")


@pytest.fixture(params=["normal", "windows-python3-stub"])
def repo(request, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if request.param == "normal":
        (bin_dir / "python3").symlink_to(sys.executable)
    else:
        stub = bin_dir / "python3"
        stub.write_text("#!/bin/sh\necho 'Python was not found' >&2\nexit 49\n")
        stub.chmod(0o755)
        (bin_dir / "python").symlink_to(sys.executable)
    git_dir = os.path.dirname(shutil.which("git"))
    env = {**os.environ, "PATH": os.pathsep.join([str(bin_dir), git_dir, "/usr/bin", "/bin"]),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.test",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.test"}
    work = tmp_path / "work"
    work.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=work, env=env, capture_output=True, text=True,
                              timeout=30, check=False)

    git("init", "-q")
    git("config", "core.hooksPath", str(REPO_ROOT / ".githooks"))
    return git


def test_bad_subject_is_refused(repo):
    result = repo("commit", "-q", "--allow-empty", "-m", "Add x")
    assert result.returncode != 0
    assert "KAN-<n>" in result.stderr


def test_good_subject_is_accepted(repo):
    result = repo("commit", "-q", "--allow-empty", "-m", "feat(data): add x (KAN-42)")
    assert result.returncode == 0, result.stderr
