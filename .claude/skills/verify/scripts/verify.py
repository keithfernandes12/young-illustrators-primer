#!/usr/bin/env python3
"""One verification command for loops: the CI checks plus the project's extra gates.

Usage:  python .claude/skills/verify/scripts/verify.py [--quick] [--no-privacy]

Runs, in order, and stops at nothing (every check reports):
  1. ruff check .            (CI)
  2. mypy src                (CI)
  3. pytest                  (CI; --quick runs only tests near changed files, then full pytest is still
                              required before the PR is marked ready)
  4. privacy lint on src/ and app/
  5. terraform fmt -check and validate, only when infra/ contains *.tf files
  6. npm run lint / test / build in app/, only when app/package.json exists

Prints one line per check and a final line that a /goal evaluator can read:
  VERIFY_RESULT: PASS
  VERIFY_RESULT: FAIL (<n> failing: <names>)
Python only, so it runs the same on Windows (Git Bash), macOS and Linux.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PRIVACY = ROOT / ".claude" / "skills" / "privacy-guard" / "scripts" / "privacy_lint.py"


def changed_tests() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    out += subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    tests: set[str] = set()
    for f in out:
        p = Path(f)
        if p.parts[:1] == ("tests",) and p.suffix == ".py" and (ROOT / p).exists():
            tests.add(str(p))
        elif p.parts[:2] == ("src", "primer") and len(p.parts) > 3:
            area = ROOT / "tests" / p.parts[2]
            if area.is_dir():
                tests.add(str(area.relative_to(ROOT)))
    return sorted(tests)


def run(name: str, cmd: list[str], cwd: Path = ROOT) -> bool:
    print(f"--- {name}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False)
    print(f"{'PASS' if result.returncode == 0 else 'FAIL'}  {name}", flush=True)
    return result.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick", action="store_true", help="pytest only the areas you changed"
    )
    ap.add_argument("--no-privacy", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    failed: list[str] = []

    checks: list[tuple[str, list[str], Path]] = [
        ("ruff", [py, "-m", "ruff", "check", "."], ROOT),
        ("mypy", [py, "-m", "mypy", "src"], ROOT),
    ]
    tests = changed_tests() if args.quick else []
    checks.append(("pytest", [py, "-m", "pytest", "-q", *tests], ROOT))
    if not args.no_privacy and PRIVACY.exists():
        targets = [t for t in ("src", "app") if (ROOT / t).is_dir()]
        checks.append(("privacy", [py, str(PRIVACY), *targets], ROOT))
    if list((ROOT / "infra").rglob("*.tf")) and shutil.which("terraform"):
        checks.append(
            (
                "terraform-fmt",
                ["terraform", "-chdir=infra", "fmt", "-check", "-recursive"],
                ROOT,
            )
        )
        checks.append(
            ("terraform-validate", ["terraform", "-chdir=infra", "validate"], ROOT)
        )
    if (ROOT / "app" / "package.json").exists() and shutil.which("npm"):
        for script in ("lint", "test", "build"):
            checks.append(
                (f"app-{script}", ["npm", "run", "--if-present", script], ROOT / "app")
            )

    for name, cmd, cwd in checks:
        if not run(name, cmd, cwd):
            failed.append(name)

    if args.quick:
        print(
            "NOTE  --quick ran a subset of tests; run without --quick before marking a PR ready."
        )
    if failed:
        print(f"VERIFY_RESULT: FAIL ({len(failed)} failing: {', '.join(failed)})")
        return 1
    print("VERIFY_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
