---
name: write-doc
description: Create or update a document in docs/ (plan, ADR, brief, runbook, solution, engineering process, meeting notes) with the right folder, file name and frontmatter, then validate it. Use when asked to write a plan, record a decision, write an ADR, document a procedure or runbook, record how a problem was solved, write meeting notes, or update a plan's status.
---

# Write a document

Rules: `docs/engineering/documentation.md`. Each folder's `AGENTS.md` adds its own.

## 1. Pick the type

1. A meeting? → `meeting-notes/`.
2. Decided, and one choice others must follow? → `adrs/`.
3. Decided, and several PRs to build? → `plans/`.
4. Not decided yet? → `briefs/` (or use `shape-feature`).
5. A solved problem likely to recur? → `solutions/`.
6. A procedure for something that runs? → `runbooks/`. For how the team works? → `engineering/`.
7. A small, fully understood change → no document; the Jira issue and PR are enough.

**Before creating anything, search for an existing document** (`grep -ril "<topic>" docs/`). Update
an existing plan or brief rather than starting a parallel one.

## 2. Create it

- Copy the folder's `_template.md` to `docs/<folder>/<kebab-slug>.md`. Meeting notes:
  `YYYY-MM-DD-<slug>.md`. No dates, numbers or type prefixes in other names.
- Frontmatter: `title`, `type` (must match the folder), `created` (today, `YYYY-MM-DD`), `status`
  (first value for the type: `draft`, or `active` for runbooks/solutions/engineering), `author`
  (`gh api user --jq .login`). Optional: `updated`, `jira: [KAN-…]`, `tags`, `related` (bare slugs of
  other documents), `supersedes`, `superseded_by`. No other fields.
- Fill every section; delete the template's `>` guidance lines. Write plainly: what is true, what
  was decided and why, what is not known.

## 3. Updating

- Change `status` as the work moves; set `updated` to today when content changes.
- Plans: update the status table and append dated Decisions / Verified facts in the same PR as the
  code. Never delete or rewrite those two sections; add a dated correction instead.
- Superseding: set the old document's `status: superseded` and `superseded_by: <new-slug>`, add a
  note at its top saying where unfinished work went, and set `supersedes` on the new one.
- Accepted ADRs are not edited to change the decision; write a new ADR that supersedes it.

## 4. Validate

Run `python scripts/check_docs.py` and fix every problem it lists before committing. Commit with
`docs(<area>): <what> (KAN-<n>)`.
