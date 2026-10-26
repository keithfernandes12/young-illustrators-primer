---
title: How we write and keep documents
type: engineering
created: 2026-10-25
status: active
author: rishiparekh
tags: [docs, process, agents]
related: [engineering-loop, ai-coding-framework]
---

# How we write and keep documents

`docs/` is where the team — people and AI agents — records decisions, plans and procedures so the
next person or agent can continue without rediscovering them. Agents read these documents before
they start work (see the `engineering-loop` skill) and write to them as they go. A document that
nobody updates is worse than none, so every type below has one owner and a status.

The rules here are checked: `scripts/check_docs.py` runs inside `pytest`, so CI fails on a
document that breaks them.

## Which folder?

| Folder | `type` | Use it for | Status values (in order) |
|---|---|---|---|
| `docs/plans/` | `plan` | **How we will build something we have decided to build**, split into PR-sized slices; also the running record of what has been built | `draft` → `active` → `closed`, or `superseded` |
| `docs/adrs/` | `adr` | **A decision and why** — a choice of library, model, data split, architecture, that others must follow | `draft` → `accepted` → `deprecated` or `superseded` |
| `docs/briefs/` | `brief` | **Shaping before a decision** — a vague idea or Jira epic turned into options, questions and slices | `draft` → `decided` or `parked` |
| `docs/runbooks/` | `runbook` | **Operating something that runs** — freeze a data version, run a retrain, roll back a deploy | `active`, `deprecated` |
| `docs/solutions/` | `solution` | **A problem we solved that will come back** — symptom, cause, fix, how to recognise it | `active`, `deprecated` |
| `docs/engineering/` | `engineering` | **How we work** — processes and checklists that people and agents follow (this file, the review checklist) | `active`, `deprecated` |
| `docs/meeting-notes/` | `meeting-note` | **What a meeting decided**, with owners | `draft` → `final` |

Documents directly in `docs/` (`project_scoping.md`, `project_scoping_tldr.md`) are the course
submissions. They are the authority on scope and architecture and are not checked by the script.

### Deciding quickly

1. Was it a meeting? → `meeting-notes/`.
2. Has the decision been made?
   - Yes, and it is one choice others must follow → `adrs/`.
   - Yes, and it takes several PRs to build → `plans/` (link the ADR in `related` if there is one).
   - No → `briefs/`. When decided, write the ADR or plan and set the brief to `decided`.
3. Did something break, and might it again? → `solutions/`.
4. Is it a procedure for running something? → `runbooks/`. A procedure for how the *team works*
   (review, docs, git) → `engineering/`.
5. Small, fully understood change? → no document; the Jira issue and the PR are enough.

## Frontmatter

Every document in a type folder starts with this block:

```yaml
---
title: Amateur Drawings ingestion            # human-readable
type: plan                                   # must match the folder (table above)
created: 2026-10-25                          # YYYY-MM-DD, the day the file was first written
status: draft                                # one of the values for this type
author: keithfernandes12                     # GitHub username of whoever wrote it first
# optional
updated: 2026-10-27                          # YYYY-MM-DD, set by hand when the content changes
jira: [KAN-42, KAN-43]                       # Jira keys this document covers
tags: [data, ingestion]
related: [dataset-versioning, drawing-model] # bare slugs of other documents: no folder, no .md
supersedes: old-plan-slug                    # when this replaces another document
superseded_by: new-plan-slug                 # required when status is superseded
---
```

No other fields are allowed; an unknown field fails the check, which is what stops field names
drifting (`owner` vs `author`, `story` vs `jira`).

## File names

- Lower-case kebab-case slug that says what it is: `amateur-drawings-ingestion.md`.
- No dates, no numbers, no type prefixes (`plan-…`, `adr-…`, `001-…`). The folder says the type;
  git says the date.
- Exception: meeting notes start with the date: `2026-10-25-weekly-sync.md`.
- The slug is the document's id. `related`, `supersedes` and `superseded_by` use it, so renaming a
  file means updating the documents that point at it (the check will tell you which).

## Plans are the implementation record

A plan is written before the work starts and kept current while it happens, so that any later
agent can pick up the next slice, or build an enhancement on top, by reading one file. The
template is `docs/plans/_template.md`. The rules:

- **One plan per piece of work that needs more than one PR.** One PR's worth of work does not need
  a plan.
- **Slices are PRs.** Each slice says why it is separate, what it touches, how we will know it
  works (**Acceptance**, written so it can be shown false) and which tests prove it (**Testing**,
  naming the case and the change that would make it fail).
- **Update the plan in the same PR as the code.** When a slice lands, mark it done in the status
  table with the PR number. When the design changes, change the plan in that PR — never keep a
  second plan somewhere else.
- **Append, do not rewrite, the record.** "Decisions" and "Verified facts" are dated and only
  grow. If a fact turns out wrong, add a dated correction below it.
- **Superseding.** When another plan takes over, set `status: superseded`, set `superseded_by`,
  and add a note at the top saying where each unfinished slice went.
- **Closing.** When every slice is done or explicitly dropped, set `status: closed`.

## ADRs

One decision per ADR: context, the options considered, the decision, and its consequences. An
accepted ADR is not edited to change its decision; write a new ADR that `supersedes` it.

## Who updates what

- The author of a PR updates every document the PR makes untrue — README commands, AGENTS.md, a
  plan's status table. The review checklist asks about it.
- Agents follow the same rules and use the `write-doc` skill to create documents.
