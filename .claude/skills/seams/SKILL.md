---
name: seams
description: Review main as a whole, not a PR - where two slices solve the same problem differently, where one stage's output is not what the next stage reads, and which open questions and hand-offs nobody owns. Writes a dated brief in docs/briefs/ whose findings each propose a slice or a Jira key. Use after a batch of merges, when asked for a "seams review", "integration review", "review main", or "what has drifted".
---

# Seams: an integration review of `main`

Per-PR review sees one diff against the `main` it was branched from. It cannot see two slices that
each look right but solve the same problem two ways, or a stage whose output stopped matching what
the next one reads. This skill reviews `main` itself, at the joins. It is `cross-slice-coherence`
S6. Run it after each batch of merges.

Read-only on code. It writes one file, the brief, and creates nothing in Jira or GitHub without
the person's go-ahead.

## 1. Pin what you are reviewing

- `git fetch origin` and note `origin/main`'s short sha. Every finding is about that commit.
- Find the last seams brief (`docs/briefs/seams-*.md`, newest by date) and the sha it recorded. What
  landed since, `git log --stat <that sha>..origin/main`, is where new seams are most likely. On the
  first run, there is no earlier brief: review all of `main`.

## 2. Gather the evidence (commands, not memory)

- **Known exceptions:** the allowlists in `tests/test_conventions.py` (`START_RUN_ALLOWED`,
  `TAGS_UNCHECKED`, `GIT_ALLOWED`, `OWNER_ALLOWED`). An item listed there is *known*, with a reason
  and a key; report it only if its reason or key no longer holds.
- **Unowned questions:** `python scripts/audit_open_questions.py --json`.
- **Duplicates:** `python scripts/scan_duplicates.py` (cross-slice-coherence S5; exit 1 means
  findings that no S1 allowlist lists, which are most of what this review is for).
- **The joins:** each `src/primer/<area>/interface.py` and `src/primer/serving/book.py`, which
  composes them; `src/primer/evaluation/fakes.py` and `golden.py`, which run the gate on their own
  fakes.
- **What each plan says is in flight:** every `docs/plans/*.md` Status table, plus
  `gh pr list --state open --limit 100 --json number,title,headRefName,baseRefName,files` (the
  same command as `review-pr` step 2: a PR whose `baseRefName` is another open PR's branch is
  stacked on it, `Depends on`, not a sibling).

## 3. Ask three questions

1. **Same problem, two solutions.** Where do two areas or slices each have their own way of doing
   one thing (logging a run, reading git state, naming metrics, loading a config, a tag set)? Name
   each copy by `path:line`, say which one the others should use, and whether that one is
   area-specific (then the fix is generalising it, not copying it).
2. **A stage's output is not what the next stage reads.** Compare each `interface.py` output type
   with what `serving/book.py` and the next area read. Include the fakes the golden runner uses,
   which are a second, separate composition.
3. **Nobody owns it.** Each untracked open question from the audit, and any hand-off a plan or a
   docstring says is "recorded" somewhere: check that the record is really there.

## 4. Write the brief

`docs/briefs/seams-<YYYY-MM-DD>.md`, frontmatter `type: brief`, `status: draft`, `author`,
`created`, `jira: [<the plan's key>]`, `related: [cross-slice-coherence]`. Sections:

- **What was reviewed**: the `origin/main` sha, the previous brief's sha (or "first run"), and the
  commands run.
- **Findings**, numbered `N1, N2, …`. Each one states:
  - the evidence, as `path:line`;
  - whether it is *known* (allowlisted, with its key) or *new*;
  - a proposed disposition: a new slice (name the plan, with its slug: `cross-slice-coherence S9`)
    or a Jira issue to create (with a draft title).
- **Not checked**: what this run could not see (CI, MLflow, Jira, running the pipeline).

Name slices with their plan's slug everywhere in the brief.

## 5. Report and hand off

Give the person the findings and the proposed dispositions. Create a Jira issue or add a slice to a
plan **only when they say so**: both are outward-facing or change a plan someone else owns. The
brief is opened as a draft docs PR like any other document.

## Never

Edit code, create or move Jira issues without being asked, post to GitHub, or mark a finding fixed
that only a later PR can fix.
