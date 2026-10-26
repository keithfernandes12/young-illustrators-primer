---
name: review-pr
description: Review a pull request (remote mode, given a PR number or URL) or your own branch before opening a PR (local mode, no argument), using the repo's review checklist and finding categories — Blocking, Fix, Tidy, Follow-up — with stable finding IDs, and re-review after changes (follow-up mode). Never approves. Use when asked to review a PR, review #N, re-review, self-review, or "check my branch before I open a PR".
---

# Review a pull request

Checklist and categories: `docs/engineering/review-checklist.md`. Read it every time.

## 1. Pin what you are reviewing

- **Remote:** `gh pr view <n> --json number,title,body,headRefOid,baseRefName,isDraft,files,author`.
  The head commit comes from GitHub, never from the local checkout. Note it.
- **Local:** the change is `git diff $(git merge-base origin/main HEAD)` plus staged, unstaged and
  untracked files. Say whether uncommitted work was included.
- **Follow-up:** if an earlier review comment from this skill exists (it contains
  `<!-- review-pr-manifest`), switch to follow-up mode (step 7).

## 2. Read the governing sources

The linked Jira issue's acceptance criteria; the plan slice in `docs/plans/`; every `AGENTS.md` for
the touched folders; existing tests for the touched code. Treat the plan and PR description as
claims to check, not as proof.

## 3. Review independently

Review from the diff and the sources, not from the author's summary. Where possible, give the diff to
a fresh subagent that did not write the code. Check:

- **Correctness and acceptance** — does the change do what the Jira issue and plan say?
- **Proof table** — every changed production file maps to a test, or to a stated reason. Behaviour
  with no proof is a finding. Would each new test fail if the behaviour were reverted?
- **Tests and CI** (`docs/engineering/ci.md`) — a `fix` without a regression test is a Fix finding;
  a non-hermetic test (network, bucket, uncontrolled time or randomness) is a Fix finding; anything
  that weakens CI (skip/xfail, deleted test, `# type: ignore`, `# noqa`, removed step, lowered
  threshold) without a stated reason is Blocking; `.github/` changes are checked against
  "Changing CI".
- **Rules in only one place** — a check enforced in the CLI or UI while the function below accepts
  anything is at least a Fix.
- **Every checklist line**, especially data rules (raw/, licence, GCS in tests, split by book),
  privacy, and secrets.
- **Docs** — is any document now untrue? Is the plan's status table updated?
- If the working tree is clean: `gh pr checkout <n>` then `ruff check .`, `mypy src` and `pytest`; otherwise say
  they were not run.

## 4. Categorise every finding

Exactly one category each — **Blocking / Fix / Tidy / Follow-up** (definitions in the checklist).
Test for Blocking: does the PR's stated goal still hold if this is left as is? Add a severity P0–P3
for ordering only.

Give each finding a stable ID: `F1, F2, …` plus a fingerprint `<category>:<path>:<approx-line>:<short-slug>`.
On a re-review, reuse an earlier finding's ID when the fingerprint matches.

## 5. Write the review

```
Reviewed <head sha> (<remote|local>, <draft|ready>)

Blocking: F1 path:line — what is wrong — what to do      (or "none")
Fix:      ...
Tidy:     ...
Follow-up: ...  (suggest the Jira issue to open)
Questions: ...
Done well: at least one specific thing
Not checked: what this review could not see (CI, Jira, running the pipeline)
Next: what the author should do
```

## 6. Post (remote mode only, and only when the person says so)

Show the review to the person first. When they agree, post it as a **comment** review, ending with
a hidden manifest:

```
<!-- review-pr-manifest {"head":"<sha>","findings":[{"id":"F1","fingerprint":"...","category":"blocking","path":"...","line":12}]} -->
```

Before posting, check the head has not moved (`gh pr view <n> --json headRefOid`). If a manifest
for this same head already exists, do not post again.

**Never approve.** Approving is the reviewer's own decision in the GitHub UI, and only once every
finding has a valid disposition.

## 7. Follow-up mode

Take the last manifest's head as the baseline; review `git diff <baseline>..<new head>` as new code
(the PR's own commits, not merges from `main`). For each earlier finding, read the author's reply —
`Fixed F1 (<commit>)`, `Tracked F1: KAN-…`, `Rebutted F1: <reason>` — and mark it **resolved**
(fixed and verified in the code, tracked with a real Jira key, or rebuttal accepted) or **still open**
(say why). Rebuttals must show the finding is factually wrong, not inconvenient; "already like that"
for a two-line fix, "tests would widen scope" for testable code, and "only P3" are not accepted.
Post a new review only if the head or a status changed.

## Never

Edit code, push, resolve the author's threads, approve, or merge. Local mode posts nothing.
