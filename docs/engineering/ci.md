---
title: How CI works and the rules it enforces
type: engineering
created: 2026-10-25
status: active
author: rishiparekh
tags: [ci, testing, github-actions, process, agents]
related: [ci-hardening, engineering-loop, review-checklist, documentation]
---

# How CI works and the rules it enforces

CI is the one check every change passes through, whoever wrote it — a person, an agent, or a person
who skipped the local hooks. This document says what runs, what blocks a merge, the principles
behind it, and the rules people and agents follow so CI stays trustworthy. Planned improvements are
tracked in `docs/plans/ci-hardening.md`; this file describes what is true now and is updated in the
same PR whenever that changes.

Each rule below is marked:
- **Enforced** — a check fails if it is broken.
- **Convention** — reviewers and the `review-pr` skill check it; no automated check yet.
- **Planned** — not in place yet; the plan slice that adds it is named.

## What runs today

| Workflow / job | Runs on | What it checks | Blocks merge? |
|---|---|---|---|
| `ci.yml` / `test` | every PR, every push to `main` | `ruff check .` → `mypy src` → `pytest` (includes the hook tests, the no-GCS guard and the docs frontmatter check) | after `protect-main` requires it (framework plan, S3) |
| `pr-rules.yml` / `pr-rules` | PR opened, edited, new commits | PR title and every commit subject match `type(scope): summary (KAN-<n>)`; PR body has a Jira key | same |

The full suite takes about 13 seconds, so CI runs everything on every PR. There is no
"only affected tests" selection and none is needed at this size.

## Where each rule is enforced

A rule that matters is enforced in more than one place, because any single layer can be skipped:

| Layer | Catches | Can be skipped by |
|---|---|---|
| Agent guard hook (`.claude/hooks/guard_bash.py`) | agents merging, pushing to `main`, writing to the bucket, opening PRs without a Jira key | a person, or a session where the hook fails to start |
| Git hook (`.githooks/commit-msg`) | bad commit subjects, for everyone | a clone that never ran `git config core.hooksPath .githooks` |
| CI (`ci.yml`, `pr-rules.yml`) | failing tests, lint, types, bad titles/commits/bodies — whoever pushed | nothing, once required |
| Branch protection (`protect-main` ruleset) | merging without a passing CI and another member's approval | repository admins only |

So local hooks give fast feedback; CI and branch protection are what actually guarantee the rule.

## Principles

These come from a much larger production CI (about 110 workflows) and are kept here because they
hold at any size. Each says why, and how it applies here.

1. **Require one stable check name.** *Planned (ci-hardening S2).* Branch protection should require a
   single summary job (`ci-pass`) that waits for every other job and fails unless each one succeeded.
   Renaming a required check, or adding an OS to a matrix (which renames the job to
   `test (ubuntu-latest)`), otherwise leaves PRs waiting forever for a check that no longer exists.
2. **A skipped job must never count as passing when it should have run.** *Planned (S2).* On GitHub a
   skipped job looks green. The summary job treats "skipped because the step that decides what to run
   failed" as a failure.
3. **Cheapest checks first.** *Enforced.* `ruff` (seconds), then `mypy`, then `pytest`, so a formatting
   mistake fails fast instead of after the slowest step.
4. **Tests are hermetic: no network, no shared data, no clock or randomness they do not control.**
   - No GCS: *Enforced* (`tests/conftest.py` fails any test that opens a `gs://` filesystem).
   - No network: *Convention; planned marker (S4).*
   - Deterministic: *Convention.* The flaky CLEAR test failed 1 run in 40 because test spreadsheets
     embedded the time they were written; see `test_xlsx_bytes_is_deterministic`.
5. **Cancel superseded runs.** *Planned (S1).* Each push replaces the previous run for that branch.
6. **Least-privilege CI token; pinned actions; no untrusted code with secrets.** *Planned (S1, S3).*
   Workflows read the repo and nothing more unless a job needs more. Third-party actions are pinned to
   a commit, so a hijacked tag cannot run new code in CI. `pull_request_target` is never used with the
   PR's own code.
7. **Every job has a timeout.** *Enforced* (`timeout-minutes` on both jobs). A hang fails clearly.
8. **Reproducible installs.** *Partly.* pip downloads are cached; ruff is planned to be pinned (S1) and
   all dependencies locked (S8). Without that, a new release of any tool can turn CI red on unchanged
   code.
9. **Ratchet debt; do not demand zero.** *Available when needed.* If a rule cannot be met everywhere at
   once, record today's count and fail only increases. Today `mypy` and `ruff` are at zero, so they
   are plain gates.
10. **A fix comes with a test that fails without it.** *Convention; planned check (S6).*
11. **Roll out a new gate in warn-only mode first, then make it required.** *Convention.* Use
    `continue-on-error: true` (or a report-only script) for at least a week, fix what it finds, then
    remove it and add the check to the ruleset.
12. **Flaky tests are fixed or quarantined, never retried away.** *Convention.* A test that sometimes
    fails teaches everyone to ignore red builds.

## Rules for developers and agents

### Before marking a PR ready

- Run what CI runs: `ruff check .`, `mypy src`, `pytest`. All must pass locally.
- A red CI run on the PR's latest commit means not ready, whatever the cause.

### Writing tests

- No network calls and no real bucket in tests. Use the fake fetcher and the `store` fixture.
- No wall-clock time or randomness the test does not control. Fix the time, seed the random
  generator, or generate fixtures once and reuse them.
- No real children's data, names, drawings or audio in fixtures.
- A bug fix (`fix:` PR) includes a test that fails before the fix and passes after. Say so in the PR
  body ("the test fails on `main` with …"). If a test is genuinely impossible, say why.
- A test earns its place only if it would fail when the behaviour it covers is reverted.

### Never make CI green by weakening it

Each of these needs a reason in a comment and a reviewer's explicit agreement, and never goes in the
same PR as the code it unblocks:
- `pytest.mark.skip`, `xfail`, or deleting a failing test.
- `# type: ignore`, `# noqa`, or a new ruff/mypy exclusion.
- Lowering a threshold, removing a CI step, or adding `continue-on-error` to an existing gate.

### When CI is red

Use the `fix-ci` skill. In short:
1. Check the failing run is for the PR's **latest** commit (`gh pr checks <n>`). A cancelled run
   (superseded by a newer push) is not a failure.
2. Read the failing step's log (`gh run view <run-id> --log-failed`) before doing anything else.
3. Reproduce it locally with the same command. Fix the cause.
4. Never re-run a failed job hoping it passes (agents must ask before `gh run rerun`, set in
   `.claude/settings.json`). If a test fails only sometimes, it is flaky — follow the next section.

### Flaky tests

1. Prove it: run the test in a loop (`for i in $(seq 1 40); do pytest -q <test> || break; done`) and
   compare against `main`. If `main` also fails, the flake predates your change.
2. Find the cause (time, randomness, ordering, shared state, network) and fix it with a regression
   test that forces the bad condition, as `test_xlsx_bytes_is_deterministic` does.
3. If it cannot be fixed now: mark it `@pytest.mark.skip(reason="flaky: KAN-<n>")` in a PR of its own,
   with the Jira issue to fix it. Never leave a known-flaky test running.
4. Record the cause in `docs/solutions/` if it could recur elsewhere.

### Changing CI (`.github/`)

Workflow changes affect everyone and are owned in `.github/CODEOWNERS`.
- Keep job names stable; a required check that is renamed blocks every PR. Changing a required name
  needs the ruleset updated in the same change (admin).
- Every workflow sets `permissions:` (read-only unless a job needs more) and every job sets
  `timeout-minutes`.
- Pin third-party actions to a full commit SHA with the version as a comment
  (`uses: actions/checkout@<sha> # v4`).
- Pass event data to scripts through `env:`, never by writing `${{ github.event... }}` inside `run:`
  (a PR title can contain shell code).
- Never use `pull_request_target` to run the PR's code.
- New gates start warn-only (principle 11).
- Keep the CI commands identical to the local ones in `AGENTS.md`, so "passes locally" means
  "passes in CI".

## Deliberately not adopted

These are useful in large organisations and cost more than they return here: a merge queue;
"affected-only" test selection; automatic bot approval; several ratchets with self-tests; separate
quality platforms (Sonar, Codecov, CodeQL together); AI classifiers that skip CI; tiered end-to-end
suites. Revisit if the suite takes more than about 10 minutes or the team grows.
