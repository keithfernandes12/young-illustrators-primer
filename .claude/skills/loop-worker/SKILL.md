---
name: loop-worker
description: Run one unattended iteration of the engineering loop inside /loop, /goal, or a Ralph loop. Picks exactly one ready slice from docs/plans/*.md, builds it with the engineering-loop skill, proves it with verify, and opens a draft PR with create-pr. Use when asked to "work the plan", "take the next slice", "run overnight", or when a loop prompt names this skill. Never merges, never deploys to prod, never asks questions it cannot get answered: it records them and moves on.
---

# Loop worker

The repo's process is `engineering-loop`. This skill is the thin wrapper that makes it safe to
run with nobody watching: one slice per iteration, a hard stop when stuck, and a clear last line
for the loop to read.

## One iteration

1. **Orient.** Read `AGENTS.md`, the plan you are told to work (default
   `docs/plans/mvp-cloud-deploy.md`), its Status table, Decisions, Verified facts and Open
   questions. Run `gh pr list --label agent-authored --state open`. If 3 or more agent PRs are
   waiting for review, stop with `LOOP_WAIT review-queue-full` (humans are the bottleneck; more
   PRs make it worse).
2. **Pick** the first slice whose State is `not started`, whose `Depends on` slices are `merged`,
   that has a `Jira` key, and that is not marked `person-only`. If none, print `LOOP_EMPTY` and stop.
3. **Check the contract.** If its Acceptance cannot be shown false, or the sources disagree, do not
   guess: add the question under Open questions in the plan (on a `docs/KAN-<n>-<slug>` branch,
   as a draft PR) and print `LOOP_BLOCKED <slice> needs-decision`. This replaces "ask the person",
   because nobody is there to answer.
4. **Build** the slice with `engineering-loop` steps 2 to 6, on a branch named
   `<type>/KAN-<n>-<slug>` from current `main`. Narrowest test first, then
   `python .claude/skills/verify/scripts/verify.py --quick` while iterating.
5. **Prove.** Full `python .claude/skills/verify/scripts/verify.py` must print `VERIFY_RESULT: PASS`.
   Model slices also need `golden-gate` for their group; slices touching child data, logs,
   prompts or storage need `privacy-guard`. At most 3 fix attempts on the same failure; then
   record what you tried as a dated Verified fact, mark the slice `blocked` in the Status table,
   and print `LOOP_BLOCKED <slice> <reason>`.
6. **Ship** with `create-pr` (draft, `agent-authored`, `needs-reviewer`, Jira key, provenance
   line). Update the plan's Status row to `in review` with the PR number in the same branch.
7. **End** the turn with exactly one line:
   `LOOP_DONE <slice> <PR URL>` | `LOOP_BLOCKED <slice> <reason>` | `LOOP_EMPTY` | `LOOP_WAIT <reason>`.

## Things unattended agents must not do here

Everything `AGENTS.md` forbids, plus: do not move or create Jira issues (those are `ask` rules and
would stall the loop; comment on the PR instead), do not open a session from
`.claude/worktrees/` (permissions do not load there; see `AGENTS.md` -> Check your setup), do not
edit `.github/`, `configs/golden/`, or `.claude/skills/golden-gate/thresholds.json`, and do not
deploy anywhere except staging through `deploy-cloud-run`, and only when the slice says so.
