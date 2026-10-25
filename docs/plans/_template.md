---
title: <What is being built>
type: plan
created: YYYY-MM-DD
status: draft
author: <github-username>
jira: [KAN-<epic or feature>]
tags: []
related: []
---

# <What is being built>

> Copy this file to `docs/plans/<kebab-slug>.md`. Delete the guidance lines in `>` quotes as you
> fill each section. Read `docs/plans/AGENTS.md` first.

## Where things stand

> Two to five sentences: what exists today, what this plan adds, and the Jira epic. Link the
> scoping section (`docs/project_scoping.md` §…) and any ADR.

## Sources read

> What you read (files, docs, Jira issues), what you did not read, and what you assumed.

## Contract

- **Target behaviour** — what will be true afterwards, from the outside.
- **Not doing** — deliberately out of scope, and why.
- **Patterns to follow** — existing code this should look like.
- **Boundaries** — which modules may depend on which; what must not change.
- **Test plan** — which tests prove it; where fakes replace GCS, network or models.

## Slices

> One slice = one PR. Order them so each one leaves `main` working.

### S1 · <short name>

*size: small | medium | large*

<One paragraph: what this PR does.>

- **Why separate** — why this is its own PR.
- **Touches** — files or modules.
- **Acceptance** — what will be observably true; written so it could be shown false.
- **Testing** — the test cases, and the change to the code that would make each one fail.
- **Depends on** — earlier slices, or "nothing".
- **Jira** — KAN-<n>.

### S2 · <short name>

...

## Status

> Update in the same PR as the code. Newest state at the top of the table.

| Slice | State | PR | Notes |
|---|---|---|---|
| S1 | not started | | |

## Decisions

> Dated, append-only. `**2026-10-25** — Decided … because …`

## Verified facts

> Dated, append-only. Facts someone checked (with how), so the next agent does not re-check them.
> Correct a wrong one with a new dated entry.

## Open questions

> One per line, each with a recommended answer and who decides.

## Out of scope

> Things someone might expect here, and where they go instead.
