---
name: create-pr
description: Commit, push and open a correct draft pull request for the current branch — Jira key, template, labels, reviewer placeholder, plan link and provenance — after choosing the smallest preparation step (plain PR, extract staged changes, clean history, or split). Use when asked to create, open or raise a PR, or to "ship this branch for review".
---

# Create a pull request

## 0. Choose the smallest preparation step

- Branch is one concern and under ~400 changed lines → continue here.
- Only some staged changes belong in this PR → `split-pr` (extract mode).
- Several concerns, or over ~400 lines → `split-pr`.
- History is messy WIP commits → `commit-reorganize` first.

## 1. Preconditions

- Not on `main`. If you are, create `<type>/KAN-<n>-<slug>` first.
- The Jira key: from the branch name, the plan, or ask the person ("Which issue? key / skip"). A PR
  without a key is refused by the guard hook and CI. Never create a Jira issue without asking.
- `ruff check .`, `mypy src` and `pytest` pass. If not, fix or stop and report.
- A `fix` PR has a test that fails without the fix; write that in "How it was checked" (for
  example "`test_x` fails on `main` with …, passes here").
- After opening: if CI turns red on the latest commit, use `fix-ci`; do not mark ready while red.
- Self-review done (`review-pr`, local mode) for anything non-trivial.

## 2. Commit and push

- Commit remaining work with `type(scope): summary (KAN-<n>)`. Never `git add -A` blindly: check
  `git status` and leave out unrelated or generated files.
- `git push -u origin HEAD`.
- Size check: `git diff --stat origin/main...HEAD`.

## 3. Write the body from the template

```bash
BODY="$(mktemp)"            # a unique file, so parallel sessions do not overwrite each other
cp .github/pull_request_template.md "$BODY"
```

Fill every section:
- **What changed / Why** — behaviour, not a file list.
- **Jira** — `KAN-<n>`.
- **How it was checked** — commands and results (`pytest` → N passed). Name anything not run.
- **Where the reviewer's judgement is needed** — at most 3 lines, each `file:line — the question`
  (a behaviour choice, data contract, privacy boundary, error handling, a name that will spread).
  If the author already made the call visibly, it is not a question. Write "None — mechanical
  change" only when true.
- **Plan** — `docs/plans/<slug>.md` and slice, or "single-PR change".
- **Agent provenance** — replace `N/A` with the session line from the session-start hook.

## 4. Open it

```bash
gh pr create --draft --label agent-authored --label needs-reviewer \
  --title "type(scope): summary (KAN-<n>)" --body-file "$BODY"
rm "$BODY"
```

Use `--reviewer <login>` instead of `needs-reviewer` only if the person names a reviewer. The title
must follow the commit format (CI checks it).

## 5. Verify and report

`gh pr view --json url,isDraft,baseRefName,labels,body` — confirm draft, base `main`, labels, and
the body. Report the URL and what the reviewer should look at. Post a short comment on the Jira
issue linking the PR.

## Never

Mark ready unless asked; merge; push to `main`; use `--body`.
