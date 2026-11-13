"""Every tracked file with a shebang must be executable in git.

Ruff's EXE001 enforces this, but only where the filesystem reports an executable bit. On Windows
`core.fileMode` is false and git stores 100644 for anything newly added, so `ruff check .` passes
locally and then fails on the Linux CI runner. This test reads the mode out of the git index, so it
catches the divergence on either platform.
"""

import re
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


# --------------------------------------------------------------- portability of the skill scripts
#
# Every teammate is primarily on Windows and the model path has to work from Git Bash and from WSL
# (docs/engineering/model-containers-on-windows.md). Two mistakes would each work on one person's
# machine and nowhere else, so they are checked here rather than found by the next person.

SKILL_SCRIPTS = sorted((REPO_ROOT / ".claude" / "skills").glob("*/scripts/*"))

#: An absolute path baked into a script. `/c/...` is the Git Bash form, `C:\...` the Windows one.
ABSOLUTE_PATH = re.compile(r"""(?:^|[\s"'=(])(?:/[a-zA-Z]/[\w.-]|[a-zA-Z]:[\\/])""")


def test_skill_scripts_exist():
    """A guard against the globs below silently matching nothing."""
    assert SKILL_SCRIPTS, "no scripts found under .claude/skills/*/scripts/"


@pytest.mark.parametrize("script", SKILL_SCRIPTS, ids=lambda p: p.name)
def test_no_skill_script_hardcodes_python3(script):
    """`python3` on Windows is the Microsoft Store alias stub, which fails.

    The scripts pick an interpreter instead; see test_deploy_scripts.py for that fallback. Putting a
    bare `python3` back into any of these must fail this test.
    """
    text = script.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):  # a shebang is only read on Unix, where python3 is correct
            continue
        if stripped.startswith("#"):  # prose about the problem is allowed to name it
            continue
        assert "python3" not in stripped or "PY=python3" in stripped, f"{script.name}: {stripped}"


@pytest.mark.parametrize("script", SKILL_SCRIPTS, ids=lambda p: p.name)
def test_no_skill_script_embeds_an_absolute_path(script):
    """A `/c/Users/...` or `C:\\Users\\...` path is one machine's layout, not a shared script."""
    offenders = [
        line.strip()
        for line in script.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("#") and ABSOLUTE_PATH.search(line)
    ]
    assert not offenders, f"{script.name} embeds an absolute path:\n  " + "\n  ".join(offenders)
