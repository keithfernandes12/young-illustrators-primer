---
title: Autonomous loops — running Claude Code unattended inside the framework
type: plan
created: 2026-11-07
status: draft
author: rishiparekh
jira: [KAN-19]
tags: [agents, loops, gcp, ci]
related: [ai-coding-framework, ci-hardening, mvp-cloud-deploy, engineering-loop]
---

# Autonomous loops — running Claude Code unattended inside the framework

## Where things stand

The AI coding framework (`ai-coding-framework`, KAN-19) makes a single supervised session safe. This
plan adds what an unattended session needs: one command that proves a change (`verify`), quality and
privacy gates (`golden-gate`, `privacy-guard`), a wrapper that runs exactly one slice per iteration
(`loop-worker`), Google Cloud skills (`deploy-cloud-run`, `train-vertex`), a second guard hook that
keeps agents off the live serving path (`guard_cloud.py`), and loop prompts in `.claude/loops/`. The how-to is
the team tutorial that ships with this kit.

## Sources read

- `AGENTS.md`, `.claude/settings.json`, `.claude/hooks/guard_bash.py`, `tests/agent_hooks/`.
- Claude Code docs on `/goal`, `/loop`, routines, permission modes and Vertex AI (links in the tutorial).
- Not read: each teammate's Claude plan (Pro, Max or none), which decides who can use routines.

## Contract

- **Target behaviour** — a person can start `/goal` or `/loop` with a prompt from `.claude/loops/`, leave,
  and come back to at most a few draft PRs that pass `verify`, follow every `AGENTS.md` rule, and never
  touched the shared bucket, `main`, or production.
- **Not doing** — routines that merge, deploy to prod, or write to Jira; Superpowers' planning skills
  (they write plans to `docs/superpowers/`, a second plan location, which `docs/plans/AGENTS.md` forbids).
- **Parked, and what would unpark each** — `golden-gate` itself is no longer parked:
  `src/primer/evaluation/` and `configs/golden/scenarios.jsonl` landed with it, so the gate
  returns PASS or FAIL and SKIP now means only that the metrics file is missing. Two workflows
  stay out, each with the condition that would bring it in:

  | Parked | Why | Unpark when |
  |---|---|---|
  | `.github/workflows/nightly-golden-gate.yml` | Its metrics step is not `continue-on-error`, so one bad night fails the job instead of warning. | The metrics step is `continue-on-error` too, or the run is proven green for a week on a branch. Then add it as warn-only, per `docs/engineering/ci.md`. |
  | `.github/workflows/claude.yml` (`@claude` on PRs) | Needs Claude on Vertex AI or an API key; an Enterprise seat is not meant for CI. | The team has a Vertex project with Workload Identity Federation, or an API key the org is willing to bill. Until then it is a cost decision, not an engineering one. |

  Neither has a Jira issue: this whole branch is KAN-89, and these are recorded here rather
  than split into tickets nobody has scheduled.
- **Patterns to follow** — hooks in Python with the `python3`-then-`python` launcher; scripts outside the
  package loaded by file path in tests; skills directly in `.claude/skills/`.
- **Boundaries** — the new hook runs after `guard_bash.py` and never replaces it.
- **Test plan** — `tests/agent_hooks/test_guard_cloud.py`, which pins the repository-name collision
  (the project id is also the repo name) and the single-project configuration; the existing
  settings-command tests still read the first PreToolUse hook (`guard_bash.py`).

## Slices

### S1 · Loop skills, prod guard and permissions

*size: medium*

Adds `.claude/skills/{verify,golden-gate,privacy-guard,loop-worker,deploy-cloud-run,train-vertex}`,
`.claude/hooks/guard_cloud.py` and its test, `.claude/loop.md`, `.claude/loops/`, the permission
additions in `.claude/settings.json`, this plan and `mvp-cloud-deploy`, and a Skills row per new skill
in `AGENTS.md`.

- **Acceptance** — `pytest`, `ruff check .`, `mypy src` pass; `guard_cloud.py` refuses a traffic
  change, a delete and a foreign `--project` while allowing a `--no-traffic` canary and a `cd` into
  the repository; `verify.py` prints `VERIFY_RESULT: PASS`.
- **Depends on** — nothing. **Jira** — KAN-89.

### S2 · Per-person setup

*size: small · person-only*

Each teammate installs the user-level plugins and skills from the tutorial, sets
`YIP_COMPUTE_PROJECT` (their own project) and `YIP_SHARED_PROJECT` (the team project) in
`.claude/settings.local.json`, and runs one supervised `/goal` on a small slice.

- **Acceptance** — each person records their check under Verified facts (one line each).

### S3 · Claude in GitHub Actions on Vertex AI (own PR)

*size: small*

`.github/workflows/claude.yml`: `@claude` mentions on PRs answered by Claude through Vertex AI with
Workload Identity Federation; actions pinned to commits, read-only default token, `timeout-minutes`.

- **Depends on** — `mvp-cloud-deploy` S1 (WIF pool). **Jira** — to create.

### S4 · Nightly golden gate (own PR)

*size: small*

`.github/workflows/nightly-golden-gate.yml`, warn-only first.

- **Depends on** — `mvp-cloud-deploy` S2. **Jira** — to create.

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S1 | not started | | |
| S2 | not started | | person-only |
| S3 | not started | | |
| S4 | not started | | |

## Decisions

- **2026-11-07** — Loops run from the repository root, never from `.claude/worktrees/`, because
  permission rules do not load there (`AGENTS.md` -> Check your setup). Parallel loops use separate clones.
- **2026-11-07** — At most 3 open agent-authored PRs per person before loops pause; review is the bottleneck.
- **2026-11-07** — Deploy, training-job and traffic commands are `ask` rules, so auto mode still stops for a
  person; describe, list and log reads are allowed so a watcher loop can run alone.

## Verified facts

- **2026-11-07** — On a clone of `main` at 58737b1 with this slice applied: `pytest` 198 passed, 1 skipped;
  `ruff check .` and `mypy src` clean; `verify.py` runs ruff, mypy, pytest and the privacy lint.

## Open questions

- Claude subscription or Vertex AI billing per person? Recommended: subscription for interactive work and
  routines; Vertex AI for GitHub Actions if the GCP credit covers it. Each person decides.

## Out of scope

- Production deploys by agents: never; see `deploy-cloud-run`.
