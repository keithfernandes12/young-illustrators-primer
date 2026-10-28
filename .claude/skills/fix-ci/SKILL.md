---
name: fix-ci
description: Diagnose and fix a failing CI run on a pull request — confirm the failure is current, read the failing step, reproduce it locally, classify it (code, test, flaky, tooling, cancelled), and fix the cause without weakening CI. Use when CI is red, a check failed, "why did CI fail", "fix the build", or a test passes locally but fails in CI.
---

# Fix a failing CI run

Rules: `docs/engineering/ci.md`. Never re-run a failed job hoping it passes, and never make CI green
by skipping a test, adding `# type: ignore`/`# noqa`, or removing a step.

## 1. Is the failure current?

```bash
gh pr view <n> --json headRefOid,statusCheckRollup
gh pr checks <n>
```

- The run must be for the PR's **latest** commit. Older runs do not matter.
- `cancelled` means a newer push superseded it — not a failure.
- A check that could not be read is unknown, not passing.

## 2. Read the failing step

```bash
gh run list --branch <branch> --limit 5
gh run view <run-id> --log-failed
```

Note the job, the step, and the first real error (not the last line).

## 3. Reproduce locally with the same command

The CI steps are the `AGENTS.md` commands: `ruff check .`, `mypy src`, `pytest`, and for `pr-rules`
`python scripts/check_pr.py` with `PR_TITLE`, `PR_BODY`, `BASE_SHA`, `HEAD_SHA` set. Make sure the
branch is up to date with what CI tested (`git fetch && git status`).

## 4. Classify and act

| Kind | Sign | Do |
|---|---|---|
| Code or test failure | reproduces locally | fix the cause; add or fix a test |
| Lint / type | `ruff` or `mypy` output | fix the code; do not add ignores |
| PR rules | title, commit or body message | fix the title/body in GitHub; for commit subjects use `commit-reorganize` |
| Passes locally, fails in CI | different OS, Python version, missing env, unpinned tool version | find the difference (compare `python --version`, `pip freeze`, the OS); fix it so both agree |
| Flaky | fails only sometimes | follow "Flaky tests" in `docs/engineering/ci.md`: prove it with a loop, compare with `main`, fix with a regression test, or quarantine with a Jira key in its own PR |
| CI infrastructure | runner/network error before any repo code ran | say so with the log line; a re-run is acceptable **only** here, once |

## 5. Fix, verify, report

- Commit the fix (`fix(<area>): … (KAN-<n>)`), push, and wait for the new run on the new commit.
- Report: the failing step, the cause, the fix, and the result of the new run (with its run id).
- If the cause is in something this PR did not touch (it also fails on `main`), say so, and ask the
  person whether to fix it here or in a separate PR with its own Jira issue.

## Never

Re-run blindly; skip, xfail or delete a test; add ignores; edit `.github/workflows/` to get past a
failure (CI changes follow "Changing CI" in `docs/engineering/ci.md` and go in their own PR).
