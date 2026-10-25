---
title: Review checklist
type: engineering
created: 2026-10-25
status: active
author: rishiparekh
tags: [review, process, agents]
related: [engineering-loop]
---

# Review checklist

Apply this before asking for review, and when reviewing. The `review-pr` skill reads this file.
Each line says why it is here. **When a reviewer catches the same thing twice, add a line in the
same PR.**

## The change

- The PR body links a Jira key, says what changed and why, and lists the commands run and their
  results. Anything not run is said so. *A reviewer cannot check what is not stated.*
- Every changed production file has a test, or the PR says why not. *Untested has to be a choice.*
- Each new test would fail if the behaviour it covers were reverted. *A test that passes either way
  proves nothing.*
- Documents the change makes untrue are updated in the same PR: README commands, `AGENTS.md`, the
  plan's status table. *Stale guidance misleads the next agent.*

## Data

- Every new or changed processed record keeps `license` and `attribution`. *Legal risk
  (project_scoping §8).*
- Nothing modifies or deletes `raw/` or `metadata/versions/`. *Every dataset version's picture-book
  `image_uri`s point into `raw/`.*
- Tests use the fake fetcher and the local `store`; nothing reaches GCS. *Tests must not change
  shared data (enforced by `tests/conftest.py`).*
- A new data source follows the four steps in `src/primer/data/README.md`, tests included.
- Train/test splits are by book, never by page. *Pages of one book in both sets overstate metrics
  (project_scoping §3.3, §8).*

## Children's safety and privacy

- No PII, child names, drawings or audio from real children in logs, test fixtures or committed
  files. *COPPA (project_scoping §8).*
- Story generation that is changed keeps its reading-level and target-word constraints, with a test
  that checks them. *The model ignoring constraints is a named failure (project_scoping §8).*

## Secrets and infrastructure

- No service-account key files, `.env` files, tokens or credentials in the diff.
- No new command that writes to the bucket by default; local runs use `--data-uri`.

## Finding categories (used by the `review-pr` skill)

Every finding goes in exactly one category. The category decides what the author must do; a
severity label (P0–P3) is only for ordering.

| Category | Meaning | Author must |
|---|---|---|
| **Blocking** | Without this fix the PR is wrong, incomplete, or says something untrue | fix it in this PR |
| **Fix** | High-confidence, valuable, and small enough to fix safely here | fix it in this PR, or argue it is wrong |
| **Tidy** | Small low-risk cleanup in a file the PR already touches | fix it, or say why not |
| **Follow-up** | Clearly worth doing, but would widen this PR too much | open a Jira issue and link it |

Test for **Blocking**: *does the PR's stated goal still hold if this is left as is?* If not, it is
blocking.

Every finding ends with one disposition from the author: **Fixed** (with the commit), **Rebutted**
(with code-level evidence that the finding is wrong), or **Tracked** (with the Jira key).

Not valid as a rebuttal: "it was already like that" when the fix is a couple of lines; "tests would
expand scope" for code that can be tested; "it's only P3"; disagreement without pointing at code.
