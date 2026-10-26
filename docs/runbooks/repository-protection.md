---
title: Protect main with a ruleset and required checks
type: runbook
created: 2026-10-25
status: active
author: rishiparekh
jira: [KAN-19]
tags: [github, admin, ci]
related: [ai-coding-framework]
---

# Protect main with a ruleset and required checks

## When to use it

Once, after the AI coding framework reaches `main`, to turn the written rules in `AGENTS.md` into
rules GitHub enforces. Also when the ruleset needs changing later, or when it has locked the team
out and must be relaxed.

Slices S2 and S3 of `docs/plans/ai-coding-framework.md`.

## Before you start

**Every step here needs `admin` on the repository.** Nobody else can run them, agents included:
an agent is refused by `.claude/hooks/guard_bash.py` before it reaches the API, and a teammate with
`push` gets `403` from GitHub. Check with:

```bash
gh api repos/keithfernandes12/young-illustrators-primer --jq .permissions
```

`admin` must be `true`. At the time of writing only `@keithfernandes12` has it.

Two preconditions that are easy to miss:

- **The framework must be merged to `main` first.** GitHub reads `CODEOWNERS` and
  `.github/pull_request_template.md` from the **default branch**, and the ruleset below targets
  `~DEFAULT_BRANCH`. While the framework sits on a feature branch, those three are inert. Workflows
  are the exception: `ci.yml` and `pr-rules.yml` run from the pull request branch, so they already
  work before the merge.
- **Step 3 only works after `test` and `pr-rules` have each reported on at least one pull request.**
  A status check that GitHub has never seen cannot be made required.

The `needs-reviewer` and `agent-authored` labels already exist; they were created during the
framework install. Re-creating them is harmless but unnecessary.

## Steps

### 1. Confirm the starting state

```bash
gh api repos/keithfernandes12/young-illustrators-primer/rulesets --jq '.[] | {id, name, enforcement}'
```

Expect no output the first time. If `protect-main` is already listed, skip to step 3.

### 2. Create the ruleset (S2)

The payload is in the repository, so there is nothing to write by hand:

```bash
gh api -X POST repos/keithfernandes12/young-illustrators-primer/rulesets \
  --input .github/rulesets/protect-main.json
```

You should see the created ruleset as JSON, including its `id`. It applies to the default branch:

- one approving review required,
- stale approvals dismissed when new commits are pushed,
- the most recent push must itself be approved, so an author cannot approve their own last commit,
- no branch deletion,
- no force-push.

Verify:

```bash
gh api repos/keithfernandes12/young-illustrators-primer/rulesets --jq '.[] | {id, name, enforcement}'
```

Expect one row, `protect-main`, `active`.

**Expect this to change how you merge.** With one required approval and last-push approval, nobody
merges their own pull request unaided, the repository owner included. That is the intent, not a
misconfiguration.

### 3. Make the checks required (S3)

GitHub replaces the `rules` array wholesale on update, so the payload must repeat the rules from
step 2 and add the new one. Find the id:

```bash
gh api repos/keithfernandes12/young-illustrators-primer/rulesets \
  --jq '.[] | select(.name=="protect-main") | .id'
```

Write this to a file, for example `protect-main-with-checks.json`:

```json
{
  "name": "protect-main",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": true,
        "required_review_thread_resolution": false
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": false,
        "required_status_checks": [
          { "context": "test" },
          { "context": "pr-rules" }
        ]
      }
    }
  ]
}
```

Apply it:

```bash
gh api -X PUT repos/keithfernandes12/young-illustrators-primer/rulesets/<ID> \
  --input protect-main-with-checks.json
```

Set `strict_required_status_checks_policy` to `true` if a branch must also be up to date with
`main` before it can merge. That is stricter and means more rebasing on a busy repository; `false`
is the gentler default for a team of this size.

Verify:

```bash
gh api repos/keithfernandes12/young-illustrators-primer/rulesets/<ID> \
  --jq '[.rules[].type]'
```

Expect `deletion`, `non_fast_forward`, `pull_request` and `required_status_checks`.

### 4. Prove it works

On the next pull request, confirm that a red `test` or a commit subject without a Jira key blocks
the merge button rather than only reporting. Until step 3 is done, CI is advisory: it reports and
nothing stops a merge.

## If something goes wrong

- **The ruleset blocks work and you need it off now.** Do not delete it; disable it, so the
  configuration survives:

  ```bash
  gh api -X PUT repos/keithfernandes12/young-illustrators-primer/rulesets/<ID> \
    --input protect-main-with-checks.json --field enforcement=disabled
  ```

  Re-enable by setting `enforcement` back to `active`. On plans that offer it, `evaluate` runs the
  ruleset in report-only mode, which is a safer way to trial a change than `active`.

- **Step 3 rejects a check name.** The `context` must match the **job** name as GitHub reports it,
  which here is `test` (from `ci.yml`) and `pr-rules` (from `pr-rules.yml`) — not the workflow
  names. Confirm with `gh pr checks <n>` on a pull request that has run both.

- **Step 3 silently drops the earlier rules.** This happens when the payload omits them, because
  `PUT` replaces `rules` wholesale. Re-apply the full payload from step 3.

- **Nobody can merge anything.** With one required approval and `require_last_push_approval`, a
  solo author is stuck by design. Someone else approves, or an admin temporarily disables the
  ruleset as above. Adding a `bypass_actors` entry is the durable fix if the team decides admins
  should be able to override.

- **Undo entirely.** `gh api -X DELETE repos/keithfernandes12/young-illustrators-primer/rulesets/<ID>`
  removes the ruleset. Nothing else in this runbook changes repository content, so there is no
  other state to unwind. Deleting a ruleset cannot be undone, but re-running step 2 recreates it
  from the file in the repository.

## What is verified and what is not

- Step 2 uses `.github/rulesets/protect-main.json`, which ships in the repository. As of
  2026-10-25 it has **not yet been applied** — no ruleset exists on the repository.
- The step 3 payload is **not** in the repository and has not been run. It was written from the
  acceptance criteria of slice S3 in `docs/plans/ai-coding-framework.md`. Treat the first run as
  the test, and correct this runbook if GitHub rejects it.
- The `admin`/`push` split in "Before you start" was checked on 2026-10-25: `@rishiparekh` has
  `push` and `triage`, not `admin`.
