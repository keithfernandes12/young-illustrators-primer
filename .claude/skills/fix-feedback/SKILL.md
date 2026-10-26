---
name: fix-feedback
description: Work through review feedback on a pull request — collect every review thread and top-level comment, plan the fixes, make them, verify, and reply to each finding with a disposition (Fixed, Rebutted with evidence, or Tracked in Jira). Use when asked to address review comments, fix PR feedback, respond to requested changes, or handle findings from review-pr.
---

# Fix review feedback

## 1. Live state

- `gh pr view <n> --json number,headRefOid,headRefName,author,isDraft,reviewDecision,reviews`.
- Collect **all** feedback: review threads (`gh api repos/{owner}/{repo}/pulls/<n>/comments --paginate`),
  formal reviews, **and top-level comments** (`gh api repos/{owner}/{repo}/issues/<n>/comments --paginate`).
  Always paginate; the default page is 30 items and the rest are silently missing.
- If this is not the person's own branch, stop and give a read-only plan instead.

## 2. Plan

If changes were requested, convert to draft (`gh pr ready <n> --undo`) and post **one** comment,
updated in place, starting with `<!-- feedback-plan -->`:

| ID | From | What must become true | Files | Planned change | Proof | Decision |
|---|---|---|---|---|---|---|

Decision is `clear` or `needs-owner`. Mark `needs-owner` when product intent is unclear, sources
conflict, or several outcomes are reasonable; tag the person who decides and leave that row alone.
Do not pick the lowest-effort reading.

Treat each comment as evidence that some behaviour is wrong, not as a literal patch instruction.
Fix the behaviour where it lives (testing a helper is not a fix when the bug is at the call site).

## 3. Fix

In order: security, correctness, design, reliability, maintainability, style. Commit with the normal
format. Run `ruff check .`, `mypy src` and `pytest`. A failing CI check is handled with the
`fix-ci` skill, never by re-running the job or weakening the check.

## 4. Verify

Re-read the whole PR diff (not just your changes), or better, ask a fresh subagent to review the
final diff against the plan rows without telling it the expected answer. Do not call your own
re-read "independent". Record the head commit you verified; if it moves before you reply, verify
again.

## 5. Reply

Reply in each thread (or to each top-level comment) with exactly one:

- `Fixed F<n> (<commit>)` — what changed.
- `Rebutted F<n>: <reason>` — code-level evidence the finding is wrong. "It was already like that",
  "out of scope" for a two-line fix, or "only P3" are not reasons.
- `Tracked F<n>: KAN-<n>` — a real Jira issue (ask the person before creating it).

Check claims like "this PR doesn't touch X" against the diff before writing them. Update the plan
comment with commit and proof per row. Resolve a thread only after its fix is pushed and verified.

## 6. Report

Changes made, a disposition per finding, the verified head commit, CI state (`gh pr checks <n>`),
and the count still open: waiting on the owner, waiting on the reviewer. Do not call the PR ready
while anything is open; green CI and answered feedback are separate. Mark ready only if asked. After
three rounds that each surface new findings, stop and ask the person whether the approach is right.

## Never

Approve, merge, or dismiss a review.
