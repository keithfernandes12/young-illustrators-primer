---
name: pr-status
description: Read-only check of a pull request (or all open PRs) — live state, checks, reviews, open feedback, linked Jira issue — and the one next action. Use when given a bare PR number or URL, or asked what is going on with a PR, what is blocking it, whether it is ready, or which PRs need review.
---

# PR status

Read-only: never post, approve, resolve, push, label or merge.

## 1. Snapshot from GitHub (not the local checkout)

```bash
gh pr view <n> --json number,title,state,isDraft,headRefName,baseRefName,headRefOid,reviewDecision,reviews,labels,mergeable,body
gh pr checks <n>
gh api repos/{owner}/{repo}/pulls/<n>/comments --paginate --jq length
gh api repos/{owner}/{repo}/issues/<n>/comments --paginate
```

For "all PRs": `gh pr list --json number,title,isDraft,labels,reviewDecision,author`.

Note the Jira key, the plan link, any `review-pr-manifest` (and the head it reviewed), and whether
the `needs-reviewer` placeholder has been claimed.

## 2. Classify

One of: `local-branch`, `draft`, `needs-reviewer`, `in-review`, `changes-requested`,
`follow-up-needed` (new commits since the last review), `approved`, `blocked` (CI red, conflicts,
or base PR not merged), `merged`, `closed`.

Judge approval from the latest review by each reviewer, not from an empty `reviewDecision`. A check
that could not be read is unknown, not passing.

## 3. Output

| PR | State | Next action | Why |
|---|---|---|---|

Then blockers, and the one skill to run next (`fix-feedback`, `review-pr`, `create-pr`, or "a
teammate needs to claim review").
