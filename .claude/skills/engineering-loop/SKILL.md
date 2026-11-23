---
name: engineering-loop
description: Take a change from request to a reviewable draft PR the way this repo expects — classify the work, read the sources, write the contract in a docs/plans plan, build one slice per PR, self-review, and finish on what is actually true. Use for any change bigger than a one-file fix, anything spanning several modules, turning a notebook or prototype into src/ code, or continuing work described in docs/plans/. Triggers on "plan this", "implement KAN-…", "continue the plan", "next slice", "engineering loop".
---

# Engineering loop

The full process is `docs/engineering/engineering-loop.md`; this skill is how an agent runs it.
Keep it proportional: the goal is fewer repeated mistakes, not more ceremony.

## 1. Classify the work

Say which kind it is, because that sets the finish line: shaping (→ `shape-feature`), planning,
prototype/notebook, implementation, fixing review feedback (→ `fix-feedback`), debugging. If it is a
one-file fix with an obvious test, skip to step 5.

## 2. Build the source packet

Read and keep a short list of what you read:

- `AGENTS.md`, plus the nearest `AGENTS.md` for every folder you will touch.
- The Jira issue and its acceptance criteria (Jira tools; if unavailable, ask the person to run
  `/mcp`). Do not invent acceptance criteria — ask.
- `docs/plans/` — is there a plan for this area? If so, read all of it, including Decisions and
  Verified facts; you are continuing it, not starting over.
- Any ADR in `related`, and `docs/project_scoping.md` for the area.
- `docs/solutions/` for earlier problems in this area.
- The code you will change **and its tests**.
- Git state: current branch, base, uncommitted work. If the PR exists, its live state (`pr-status`).
- **Siblings**: open PRs (`gh pr list --state open`) and what landed on `main` since the plan was
  written. A slice is only "independent" if no open PR adds a helper it would use; otherwise that
  PR goes under *Depends on* (`docs/plans/cross-slice-coherence.md`).

Write down what you did not read and what you assumed.

## 3. Write the contract

- **Existing plan:** add or adjust the slice you are about to build, using the format in
  `docs/plans/_template.md`. Do not start a second plan.
- **New multi-PR work:** create the plan with the `write-doc` skill (`docs/plans/<slug>.md`,
  `status: draft`, then `active` when the first slice starts).
- **Single-PR change:** the contract goes in the PR body instead.

The contract names: target behaviour, not doing, patterns to follow, boundaries, test plan, and a
proof table (each file you expect to change → the test that covers it, or why none).

Stop and ask the person if the acceptance criteria are missing, sources disagree, or more than one
behaviour would be reasonable. Do not pick the smallest interpretation silently.

## 4. Build one slice

- Branch `<type>/KAN-<n>-<slug>` from `main` (or from the slice it depends on, at most 2 deep).
- Find existing tests first. For a risky refactor, write a test that pins current behaviour before
  changing it.
- Structure changes (moves, renames) in their own commits before behaviour changes.
- Run the narrowest test first (`pytest path::test_name`), then `ruff check . && mypy src && pytest`.
- Bug fix: write the test first and see it fail on the current code, then fix. Keep tests hermetic
  (no network, bucket, uncontrolled time or randomness) — `docs/engineering/ci.md`.
- Never get to green by skipping tests or adding `# type: ignore`/`# noqa`; never touch
  `.github/` in a feature PR.
- A test earns its place only if it would fail when the behaviour is reverted. "Not tested" is a
  decision recorded in the proof table, never an omission.
- Pipeline runs: `primer-data --data-uri .local-data … --batch-size 2`. Never the bucket.

## 5. Review yourself

Run `review-pr` in local mode. Every finding gets a disposition (Fixed / Rebutted with evidence /
Tracked in Jira). If the shape is wrong, reshape it rather than polishing it.

## 6. Reconcile with `main`, then update the record and open the PR

Slices are built in parallel and never read each other, so a slice can re-implement something that
landed while it was being built. Before opening the PR, and again before marking it ready:

- `git fetch origin`, then `git log --stat $(git merge-base HEAD origin/main)..origin/main`. For
  every module or helper that landed in the area of this slice, either use it or say in the PR body
  why not.
- `gh pr list --state open --json number,title,headRefName,files`: for any open PR that adds a helper
  this slice also adds, or touches the same convention (logging, git state, tags, a shared config),
  either depend on it (list it under *Depends on* and wait for it) or say why they stay separate.
- `python -m pytest tests/test_conventions.py` when it exists: it fails on the known kinds of drift.
- Write one line in the PR body, under **How it was checked**:
  `Reconciled with main at <sha>: reused <x> / not applicable / differs because <reason>.`

Then, in the same branch: mark the slice in the plan's status table, append any dated Decisions or
Verified facts, and fix any document the change made untrue. Then `create-pr`.

## 7. Finish on what is true

Before saying the work is ready: the branch is on current `main` and reconciled with it again
(step 6); CI is green on the latest commit
(`gh pr checks <n>`; if red, use `fix-ci`); the PR body matches the final diff; every review finding
has a disposition.
Green CI alone does not mean ready.

## 8. Keep the lesson

If a reviewer caught something that had been caught before, add a line to
`docs/engineering/review-checklist.md` or the relevant `AGENTS.md` in this PR.

## Never

Merge, push to `main`, write to the bucket, or create or move Jira issues without asking.
