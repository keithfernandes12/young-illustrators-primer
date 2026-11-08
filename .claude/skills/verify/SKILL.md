---
name: verify
description: One command that runs everything CI runs (ruff check ., mypy src, pytest) plus the project's extra gates (privacy lint, terraform, app checks) and ends with VERIFY_RESULT PASS/FAIL. Use at the end of every loop iteration, before create-pr, before marking a PR ready, and as the success condition of any /goal or Ralph loop ("until verify passes").
---

# Verify

```bash
python .claude/skills/verify/scripts/verify.py            # full run; required before "ready for review"
python .claude/skills/verify/scripts/verify.py --quick    # inner-loop speed: only tests near changed files
```

The last line is always `VERIFY_RESULT: PASS` or `VERIFY_RESULT: FAIL (<n> failing: <names>)`.
Loops should state their goal in terms of that line, for example
`/goal verify.py prints VERIFY_RESULT: PASS on the full run, or stop after 15 turns`.

## Rules

- This adds to `AGENTS.md` -> Tests and CI; it never replaces it. CI is still the authority.
- A FAIL is fixed at the cause with `fix-ci` habits: read the failing output, reproduce the single
  test, fix, re-run. Never skip, xfail, `# noqa`, or `# type: ignore` your way to PASS.
- `--quick` is for iteration only. The PR is not ready until the full run passes.
- The script never touches GCS or the network beyond `git diff origin/main` (local refs).
