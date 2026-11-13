"""Every tracked file with a shebang must be executable in git.

Ruff's EXE001 enforces this, but only where the filesystem reports an executable bit. On Windows
`core.fileMode` is false and git stores 100644 for anything newly added, so `ruff check .` passes
locally and then fails on the Linux CI runner. This test reads the mode out of the git index, so it
catches the divergence on either platform.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def tracked_files() -> list[tuple[str, str]]:
    """(mode, path) for every file in the index."""
    result = subprocess.run(
        ["git", "ls-files", "--stage"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    rows = []
    for line in result.stdout.splitlines():
        meta, _, path = line.partition("\t")
        if path:
            rows.append((meta.split()[0], path))
    return rows


def has_shebang(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"#!"
    except OSError:
        return False


def test_shebang_files_are_executable():
    offenders = [
        path
        for mode, path in tracked_files()
        if mode == "100644" and has_shebang(REPO_ROOT / path)
    ]
    assert not offenders, (
        "These files start with a shebang but are mode 100644 in git, which fails ruff EXE001 on "
        "Linux CI. Fix with: git update-index --chmod=+x <path>\n  " + "\n  ".join(offenders)
    )


def test_executable_files_have_a_shebang():
    """The other direction: an executable bit on a file nothing can run is noise."""
    offenders = [
        path
        for mode, path in tracked_files()
        if mode == "100755" and (REPO_ROOT / path).suffix in {".py", ".sh"}
        and not has_shebang(REPO_ROOT / path)
    ]
    assert not offenders, offenders
