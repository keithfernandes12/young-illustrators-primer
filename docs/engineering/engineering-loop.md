---
title: How a piece of work goes from request to merged PR
type: engineering
created: 2026-10-25
status: active
author: rishiparekh
tags: [process, agents, planning]
related: [documentation, review-checklist]
---

# How a piece of work goes from request to merged PR

This is the standard path for any change bigger than a one-file fix, whether a person or an agent
does it. The `engineering-loop` skill walks an agent through it. The point is that a reviewer — or
the next agent — can see what was intended, what was proven, and what was not, without asking.

## 1. Say what kind of work it is

The kind of work sets what "done" means:

| Kind | Done means |
|---|---|
| Shaping (vague idea → slices) | a brief in `docs/briefs/` the team can decide on |
| Planning | a plan in `docs/plans/` with slices, acceptance and tests |
| Prototype / notebook | a result and what it showed; not merged into `src/` as is |
| Implementation | a draft PR that meets its slice's acceptance, with tests |
| Fixing review feedback | every finding has a disposition (see the review checklist) |
| Debugging | the cause found and proven, then a fix through the normal path |

## 2. Collect the sources before changing anything

Read, and list in the plan or PR what you read:

- `AGENTS.md` and the nearest `AGENTS.md` for every folder you will touch.
- The Jira issue (acceptance criteria) and its parent epic.
- The plan for this work in `docs/plans/`, and any ADR it relates to.
- `docs/project_scoping.md` for the area (scope, architecture §9, failure analysis §8).
- The existing code and **its tests** for what you will change.
- `docs/solutions/` for earlier problems in the same area.

Also write down what you did **not** read and what you assumed. A guess written down can be
checked; a guess left out cannot.

## 3. Write the contract before the code

In the plan (or the PR body, for a single-PR change):

- **Target behaviour** — what will be true afterwards, from the outside.
- **Not doing** — what is deliberately out of scope.
- **Patterns to follow** — the existing code this should look like (for example
  `src/primer/data/sources/tinystories.py` for a new incremental source).
- **Boundaries** — which modules may depend on which; what must not change.
- **Test plan** — which tests prove the target behaviour, and where fakes replace GCS or network.
- **Proof table** — every file you expect to change, and the test (or the stated reason there is
  none) that covers it.

## 4. Build in slices

- One idea per PR. Structure changes (moves, renames) in their own commits, separate from
  behaviour changes.
- Find the existing tests before editing. Run the narrowest test first
  (`pytest tests/data/test_sources.py::test_x`), then `ruff check . && mypy src && pytest`.
- A test is worth keeping only if it would fail when the behaviour it covers is broken.
- "Not tested" must be a decision written in the proof table, never an omission.

## 5. Review yourself before anyone else does

Run the `review-pr` skill in local mode on your branch and deal with what it finds. Then open a
draft PR with the `create-pr` skill.

## 6. Finish by what is actually true

A PR is ready for review when: it is based on current `main`; CI is green on its latest commit;
every review finding has a disposition; the PR body matches the final diff; and the plan's status
table says what landed.

## 7. Keep the lesson

When the same mistake is caught twice, add a line to `docs/engineering/review-checklist.md` or the
relevant `AGENTS.md` in the PR that fixes it, so nobody pays for the same correction a third time.
