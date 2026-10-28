---
title: CI hardening — safety, Windows coverage, one required check, test-quality gates
type: plan
created: 2026-10-25
status: draft
author: rishiparekh
jira: [KAN-19]
tags: [ci, testing, github-actions]
related: [ci, ai-coding-framework]
---

# CI hardening — safety, Windows coverage, one required check, test-quality gates

## Where things stand

CI is two jobs added by the AI coding framework (`docs/plans/ai-coding-framework.md`, S1): `test`
(`ruff check .` → `mypy src` → `pytest`) and `pr-rules` (title, commits, Jira key). The suite runs in
about 13 seconds. Neither job is required yet (framework S3). The principles this plan implements,
and what each slice changes for developers, are in `docs/engineering/ci.md`; update that document's
"Enforced / Planned" markers in the same PR as each slice.

## Sources read

- `.github/workflows/ci.yml`, `.github/workflows/pr-rules.yml`, `scripts/check_pr.py`.
- A study of a ~110-workflow production CI (summary in `docs/engineering/ci.md` → Principles).
- `docs/project_scoping.md` §8 (CI evaluation gate on the golden set).
- Not read: the team's GitHub plan limits beyond "public repository" (Actions minutes are free for
  public repositories).

## Contract

- **Target behaviour** — CI is safe by default (read-only token, pinned actions, cancelled
  superseded runs), runs on Linux and Windows, exposes one required check (`ci-pass`), and adds
  test-quality gates (changed-line coverage, "fix needs a test", no network in tests) after a
  warn-only period.
- **Not doing** — merge queue, affected-only selection, auto-approval, Sonar/Codecov.
- **Patterns to follow** — gates as small Python scripts under `scripts/` with tests in `tests/`, as
  `scripts/check_pr.py` is.
- **Boundaries** — CI runs the same commands as `AGENTS.md` → Commands; no CI-only behaviour.
- **Test plan** — script logic is unit-tested; workflow behaviour is proven on a throwaway PR (a
  deliberately failing commit must turn `ci-pass` red), recorded under Verified facts.

## Slices

### S1 · Safe defaults

*size: small*

Add `permissions: contents: read` and `concurrency: {group: ${{ github.workflow }}-${{ github.ref }},
cancel-in-progress: true}` to both workflows; pin ruff to a range (`ruff>=0.17,<0.18`) in
`pyproject.toml`.

- **Why separate** — no behaviour change for passing PRs; easy to review.
- **Touches** — `.github/workflows/*.yml`, `pyproject.toml`, `docs/engineering/ci.md` (principles 5, 6, 8).
- **Acceptance** — two quick pushes to one PR: the first run shows "cancelled"; the job token cannot
  write (a `gh api -X POST` comment step would fail — verify once on a scratch branch, then remove).
- **Testing** — none beyond CI itself; record the cancelled-run observation under Verified facts.
- **Depends on** — nothing.

### S2 · Windows runner and one required check

*size: small*

Run `test` as a matrix on `ubuntu-latest` and `windows-latest` (shell: bash, so the hooks run the way
they do for teammates). Add a `ci-pass` job with `needs: [test, pr-rules]`, `if: always()`, that fails
unless every needed job's result is `success`. Ask the admin to require `ci-pass` (instead of `test`
and `pr-rules`) in `protect-main`.

- **Why separate** — changes the required check name; needs an admin step.
- **Touches** — `.github/workflows/ci.yml` (and `pr-rules` either moved into it or referenced via a
  reusable workflow — required checks across workflows need care; prefer one workflow file).
- **Acceptance** — on a scratch PR: a failing test on Windows only turns `ci-pass` red; all green turns
  it green. The Windows run executes `tests/agent_hooks/` (the POSIX-only launcher tests skip;
  everything else runs). Closes open question 2 of the framework plan for CI.
- **Testing** — the scratch-PR observations above, recorded under Verified facts.
- **Depends on** — S1.

### S3 · Pinned actions and Dependabot

*size: small*

Pin every `uses:` to a full commit SHA with `# vN`; add `.github/dependabot.yml` for
`github-actions` and `pip`, weekly, grouped.

- **Acceptance** — `grep -E "uses: [^@]+@v[0-9]" .github/workflows/*.yml` prints nothing; Dependabot
  opens its first PR within a week.
- **Depends on** — nothing.

### S4 · No network in tests

*size: small*

Register a `network` marker in `pyproject.toml` and run `pytest -m "not network"` in CI. Add an
autouse guard in `tests/conftest.py` that blocks `socket.socket.connect` to non-localhost addresses
unless the test has the `network` marker.

- **Acceptance** — a test that calls `urllib.request.urlopen("https://example.com")` without the marker
  fails with a message naming the marker.
- **Testing** — a test proving the guard blocks an outbound connection, and one proving the marker
  allows it (connect is mocked; nothing leaves the machine).
- **Depends on** — nothing.

### S5 · Coverage of changed lines (warn-only, then required)

*size: medium*

Add `pytest-cov` and `diff-cover`; in CI, `pytest --cov=src --cov-report=xml` then
`diff-cover coverage.xml --compare-branch=origin/main --fail-under=80`, first with
`continue-on-error: true`.

- **Acceptance** — the PR shows the changed-line coverage figure; after two weeks warn-only, the
  step is made blocking.
- **Depends on** — S2 (so it lands inside the required `ci-pass`).

### S6 · A fix needs a test (warn-only, then required)

*size: small*

In `scripts/check_pr.py`: when the title starts with `fix`, require at least one changed file under
`tests/`, unless the PR has the `no-test-required` label.

- **Testing** — cases in `tests/agent_hooks/test_check_pr.py`: `fix` with a test change passes;
  without fails; with the label passes; `feat` without tests passes.
- **Depends on** — nothing.

### S7 · Format check

*size: small*

After the one-time `ruff format .` PR (framework plan, Later slices), add `ruff format --check .`
to CI.

- **Depends on** — the format PR.

### S8 · Locked dependencies

*size: medium*

Adopt a lockfile (`uv lock`, or `pip-tools` with `requirements*.txt`), install from it in CI, and key
the cache on its hash.

- **Acceptance** — two CI runs a week apart install identical versions.
- **Depends on** — a team decision on uv vs pip-tools (open question 1).

### Later

- Model evaluation workflow when models exist (`project_scoping` §8): golden-set quality, reading
  level and target-word constraints, safety filters. Runs on manual trigger or nightly and when model
  or prompt code changes; advisory first, required before deploys. GitHub-hosted runners have no GPU:
  call Vertex AI or use small CPU models.

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S1 | not started | | |
| S2 | not started | | needs admin for the ruleset |
| S3 | not started | | |
| S4 | not started | | |
| S5 | not started | | warn-only first |
| S6 | not started | | warn-only first |
| S7 | not started | | after the format PR |
| S8 | not started | | needs open question 1 |

## Decisions

- **2026-10-25** — One required check (`ci-pass`) instead of requiring each job, so adding the Windows
  runner or new jobs never needs a ruleset change. If S2 lands before the framework plan's S3, the
  admin requires `ci-pass` and skips requiring `test` and `pr-rules` separately.
- **2026-10-25** — Coverage is measured on changed lines, not as a repository-wide percentage.

## Verified facts

- **2026-10-25** — Full suite runtime about 13 s on Linux (174 tests); slowest single test about 1 s.

## Open questions

1. uv or pip-tools for the lockfile? *Recommended: uv (one tool for venv, lock and install; fast on
   Windows).* Team decides.
2. Changed-line coverage threshold: *recommended 80 %*, revisit after the warn-only period.
