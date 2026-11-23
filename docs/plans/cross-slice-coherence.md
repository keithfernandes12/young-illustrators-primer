---
title: Keeping independently built slices coherent
type: plan
created: 2026-11-22
status: active
author: rishiparekh
jira: [KAN-38]
tags: [process, loops, review, mlflow]
related: [autonomous-loops, ai-coding-framework, mvp-cloud-deploy]
---

# Keeping independently built slices coherent

## Where things stand

One implementation agent builds independent slices, each as its own PR, and each PR goes to a
separate session for independent review. That keeps implementation from waiting on review rounds,
and it is the working model. This plan fixes its one known cost: slices that never read each other
re-implement the same shared conventions, and nobody owns the join. It adds a shared-conventions
list, a reconcile step before handoff, a merge-order rule, one cross-slice review question, and a few
mechanical scans. It is a process plan, so most slices change docs, skills and tests, not product code.

The case that exposed it: `mvp-cloud-deploy S6f` (PR #39, `sensitivity.py`) was built on `main` before
`mvp-cloud-deploy S6c3` (#44)
added `drawing/tracking.py`. It therefore wrote its own MLflow tags, params and git-state code, with
the owner hardcoded. Nothing in build or review compared it with #44 until a merge of `main` showed
the overlap. PR #46 (`refactor(drawing): log sensitivity runs through tracking.py`) fixed
that one file and merged; this plan is about the class of problem, not that file.

## Sources read

- `docs/plans/mvp-cloud-deploy.md`: the S6f slice text ("builds only on `main`", "reads nothing that
  S6c or the MLflow server change") and the Open question "MLflow logging is not in the repo".
- The PR 39 review history in-session: four independent reviews, each scoped to one PR against the
  `main` it had at the time.
- `origin/main` as of 2026-11-22: a grep of `start_run(` and of git-state helpers (Verified facts).
- PR #46's tree at `2f418ed`, re-checked at `3db5f56`: the S1 scans were run against it (three stale
  `drawing/sensitivity.py`
  entries) and its CLI test was checked against S0.
- Not read: `docs/engineering/review-checklist.md` and the `review-pr` skill, which S4 starts by
  reading. S3 read `loop-worker` and `engineering-loop` for its own PR. Assumed: the skills can be
  edited in this repo, and CI may run a new test file.

## Contract

- **Target behaviour** — a convention that exists in `main` is used by every later slice, or the
  slice says in its PR why not. A new copy of a shared helper, or a second way to start an MLflow run,
  fails a test instead of waiting for a person to notice.
- **Not doing** — no change to the implement-then-review split, no serial review gate, and no big
  refactor of the other areas' logging in the first slices. Existing offenders are listed and
  tracked, not fixed here.
- **Patterns to follow** — `drawing/tracking.py` (`run_tags`, `run_params`, `log_run`) for logging;
  `drawing/evaluate._git_state` (cwd is the package, with a dirty flag) for git state;
  `tests/drawing/` for tests that scan code.
- **Boundaries** — a shared helper lives in one module that model areas may import and that imports no
  model area. Where it lives is decided in S2.
- **Test plan** — convention tests read source files with `ast` or `git grep`-style scans, so they
  need no weights, network or MLflow. Each test names the code change that would make it fail.

## Slices

> One slice = one PR. S0 has merged (#46); S1 is the first to build.

### S0 · Move `sensitivity.py` onto `tracking.py`

*size: small · merged as PR #46*

`sensitivity.py` builds its MLflow tags with `tracking.run_tags` and its params with
`tracking.run_params`, takes `--owner`, and drops the hardcoded owner. Done in #46.

- **Why separate** — it is the concrete instance; it can merge before the tests exist.
- **Touches** — `src/primer/drawing/sensitivity.py`, `tests/drawing/test_sensitivity.py`; as merged,
  #46 also changed `drawing/tracking.py`, `tests/drawing/test_tracking.py` and
  `docs/plans/mvp-cloud-deploy.md`.
- **Acceptance** — a `--mlflow` run carries every tag `tracking.run_tags` writes, including `owner`
  from the flag; there is no literal owner in `sensitivity.py`; `git_dirty` reads `unknown` or
  `true`/`false`.
- **Testing** — the CLI test asserts the tag set equals `run_tags(...)` for the same result; putting
  the hardcoded owner back fails it.
- **Depends on** — #44 (merged). **Jira** — KAN-38.

### S1 · Convention tests for MLflow use and git state

*size: small*

`tests/test_conventions.py` scans `src/` and fails on drift. It has an explicit allowlist for the
call sites that exist today, each with a reason and a ticket, so it passes on `main` now and fails the
moment a new one appears.

1. Every `mlflow.start_run(` in `src/` is either in an allowlisted file or inside the shared logging
   helper. Today's call sites outside `drawing/tracking.py`: `speech/train.py`, `story/grid.py`,
   `illustration/consistency/experiment.py`.
2. A run's tag dict contains the seven required tags from `docs/engineering/model-development.md`
   (`stage`, `owner`, `data_version`, `git_sha`, `base_model`, `gate_group`, `where`). The tests check
   `tracking.run_tags` (in `tests/drawing/test_tracking.py`) and each allowlisted site's tags where
   they are built from a literal.
3. Git state comes from one function. Allowlist today's copies: `evaluation/golden.py:_git_sha`,
   `speech/evaluate.py:git_sha`, `story/grid.py:git_sha`, `illustration/consistency/experiment.py:git_sha`.
   A new function named `*git_sha*` or `*git_state*` anywhere else fails.
4. No literal owner string is passed as a tag value.

- **Why separate** — it is the guardrail; the migrations (S2) shrink its allowlist one area at a time.
- **Touches** — `tests/test_conventions.py`, no `src/` change.
- **Acceptance** — passes on `main`; adding a new `mlflow.start_run(` or `def _git_sha` in a file
  outside the allowlist fails with a message that names the shared helper to use; each allowlist entry
  carries a reason and a Jira key.
- **Testing** — for each of the four checks, a fixture file in `tmp_path` that violates it and is
  caught; removing the `ast` visit for `start_run` makes the first one fail.
- **Depends on** — nothing to build, but **merge order with S0 (#46) matters**. S1 allowed
  `drawing/sensitivity.py`'s `start_run`, its literal owner and its unchecked tag dict, and fails
  on stale entries. #46 merged first, so S1 deleted those three entries (2026-11-23) and starts with
  none for `drawing/sensitivity.py`. **Jira** —
  KAN-38 for drawing; KAN-128 for speech logging.

### S2 · One git-state helper, and the areas that can adopt it now

*size: medium*

Move `evaluate._git_state` to a shared module that no model area owns, keep the cwd-is-the-package
behaviour and the dirty flag, and switch `drawing/evaluate.py`, `drawing/sensitivity.py` and
`evaluation/golden.py` to it. The speech, story and illustration copies stay on the allowlist with
their area's ticket.

- **Why separate** — a shared helper must land before consumers; it touches several modules.
- **Touches** — the new helper module; `drawing/evaluate.py`, `drawing/sensitivity.py`,
  `evaluation/golden.py`; the S1 allowlist.
- **Acceptance** — one definition of git state for those three modules; the regression test
  `test_the_recorded_commit_is_the_packages_not_the_callers` still passes against the new location.
- **Testing** — the S1 git-state check fails if `golden.py`'s private copy comes back.
- **Depends on** — S1. **Jira** — KAN-38 (decide whether the helper needs its own key).

### S3 · The reconcile-with-main step, in the loop skills

*size: small*

Add one step to `engineering-loop` and `loop-worker`, before "mark ready": fetch `main`, list what
landed since the branch point (`git log --stat <merge-base>..origin/main`), and for every new module
or helper in the area of this slice, reuse it or say in the PR body why not. Also add the
merge-order rule (Decisions) to the plan template's Slices section: a slice that introduces a shared
helper lands before any slice that should use it, and the consumer lists it under *Depends on*.

- **Why separate** — a docs and skills change with no code.
- **Touches** — `.claude/skills/engineering-loop/`, `.claude/skills/loop-worker/`,
  `docs/plans/_template.md`, `docs/plans/AGENTS.md`.
- **Acceptance** — the step is in both skills; the PR template has a line "Reconciled with `main` at
  `<sha>`: reused / not applicable / differs because …".
- **Testing** — `scripts/check_docs.py` passes; none otherwise.
- **Depends on** — nothing to build; merges after S1, whose test file and this plan its text links.
  **Jira** — KAN-38.
- **Added (2026-11-23, at the KAN-38 owner's request)** — the slice-naming rule: outside its own
  plan, a slice is written with the plan's slug in front (`mvp-cloud-deploy S6f`,
  `cross-slice-coherence S1`), in `docs/plans/AGENTS.md`, the plan template and `create-pr`.
  `loop-worker`'s `LOOP_*` end line keeps a bare label, because loop prompts read that format.
- **Split (2026-11-23)** — the PR-template line goes in its own one-line PR, because `AGENTS.md`
  puts `.github/` changes in their own PR. Until it lands, `create-pr` asks for the line at the
  end of "How it was checked".

### S4 · A cross-slice question for the reviewer sessions

*size: small*

Add to `docs/engineering/review-checklist.md` and the `review-pr` skill: "Does this PR re-implement
something now in `main`, or add something an open sibling PR also adds?" The review prompt gets
`gh pr list --state open` output and `git log --stat <merge-base>..origin/main`, so the reviewer can
answer it. A finding in that category is a Fix, not a Tidy, because it is cheap now and costs a
migration later.

- **Why separate** — it changes how every review session is briefed.
- **Touches** — `docs/engineering/review-checklist.md`, `.claude/skills/review-pr/`.
- **Acceptance** — the checklist and the skill both carry the question and the two inputs.
- **Depends on** — S3 (same vocabulary). **Jira** — KAN-38.

### S5 · A duplicate-helper scan in `verify`

*size: small*

`scripts/scan_duplicates.py`, run by `verify`, reports private functions with the same name or the
same normalised body in more than one module, and the same string literal (for example `"yip-drawing"`,
a tag key) in more than three. It warns and prints; it does not fail, so it cannot block a slice.
Findings that stay open over two runs become Open questions in the owning plan.

- **Why separate** — heuristic, so it must not share a PR with the hard tests.
- **Touches** — `scripts/scan_duplicates.py`, the `verify` skill's script list.
- **Acceptance** — on today's `main` it lists the five git-state copies and the `start_run` sites;
  on a branch with a new duplicate it lists that one. It also covers what the S1 scans cannot see:
  inline `subprocess` calls to `git rev-parse` outside a function named for git state (today
  `src/primer/data/versioning.py:_git_commit`), `mlflow.set_tag("owner", …)` (singular),
  `dict(owner="x", …)` and `{**base, "owner": "x"}`, `MlflowClient().create_run`, a `start_run`
  reached by an alias (`from mlflow import start_run as sr`, `getattr(mlflow, "start_run")`,
  `s = mlflow.start_run`) or with `tags` as the sixth positional argument or in `**opts`, and
  `scripts/`, which S1 does not scan.
- **Testing** — a fixture tree with two identical private helpers is reported; a single one is not.
- **Depends on** — nothing. **Jira** — KAN-38.

### S6 · A periodic integration review over `main`

*size: small*

A reviewer prompt, saved as a skill (`seams`), that reviews `main` rather than a PR: where two slices
solve the same problem differently, where one stage's output is not what the next stage reads, and
which Open questions and hand-offs have no slice or Jira key. You run it after each batch of merges.
Its output is a list of findings, each either a new slice in this plan or a Jira key.

- **Why separate** — a different kind of review, with no diff.
- **Touches** — `.claude/skills/seams/SKILL.md`.
- **Acceptance** — run once on today's `main`, it reports at least the five git-state copies and the
  four logging styles; the report is saved under `docs/` as a dated note or a brief.
- **Depends on** — S1 (so the first report separates known from new). **Jira** — KAN-38.

### S7 · A seam smoke test

*size: medium*

One end-to-end test with fakes that runs each stage's contract in order (drawing recognition, story,
illustration) and asserts that each stage's output type is what the next reads. It catches name and
schema mismatches that per-slice tests cannot.

- **Why separate** — it spans areas and belongs to the serving slice's owner as well.
- **Acceptance** — passes on `main`; renaming a field in one stage's output type fails it.
- **Depends on** — the stage contracts in `src/primer/interface.py`; confirm they are stable.
  **Jira** — to create.

### S8 · An open-questions audit

*size: small*

A script that reads every plan's Open questions and hand-offs and lists those with neither a slice
nor a Jira key in the text. S5 and S6 findings use it as the place a gap becomes work.

- **Acceptance** — on today's plans it lists the questions with no key; a question given a key
  disappears from the list.
- **Depends on** — nothing. **Jira** — to create.

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S0 | merged | [#46](https://github.com/keithfernandes12/young-illustrators-primer/pull/46) | confirmed against #46 at `2f418ed`, re-checked at `3db5f56`: tags built by `run_tags` (the CLI test asserts equality), owner from `--owner`, no literal owner, `git_dirty` `unknown`/`true`/`false`. Merged 2026-11-23 as `2d3f077` |
| S1 | in review | [#47](https://github.com/keithfernandes12/young-illustrators-primer/pull/47) | `tests/test_conventions.py`: four scans, count-based allowlists with reasons and keys, stale entries fail |
| S2 | not started | | needs S1 |
| S3 | in review | [#48](https://github.com/keithfernandes12/young-illustrators-primer/pull/48) | reconcile step in `engineering-loop`, `loop-worker`, `create-pr` and the process doc; merge-order rule in the template and `docs/plans/AGENTS.md`. Merges after #47 (it links this plan and `tests/test_conventions.py`). Also the slice-naming rule (plan slug in front, outside the plan). The PR-template line is split out: `.github/` changes go in their own PR |
| S4 | not started | | needs S3 |
| S5 | not started | | |
| S6 | not started | | needs S1 |
| S7 | not started | | needs stable stage contracts |
| S8 | not started | | |

## Decisions

- **2026-11-22** — Keep implementation and review decoupled, and add reconciliation at the seams
  instead of review gates, because waiting on review rounds is the cost the owner is avoiding.
- **2026-11-22** — Mechanical checks come before process steps in the build order (S1 before S3), because
  a test fails without anyone remembering to run it.
- **2026-11-22** — Merge order follows dependencies, not readiness: a slice that adds a shared helper
  merges before the slices that should use it, and the consumers list it under *Depends on*. A consumer
  that is ready first waits, or ships with the gap recorded as an allowlist entry and a ticket.
- **2026-11-22** — Existing divergent call sites are allowlisted with a reason and a key, not migrated
  in S1, so the guardrail lands without a cross-area refactor.

- **2026-11-23** — S1's allowlists count, and an entry that no longer matches fails. `start_run`
  is allowlisted by (file, count) and git helpers by (file, function name), so a second
  `start_run` in an already-listed file is still caught. The shared logger and the shared git
  helper are named constants (`SHARED_LOGGER`, `SHARED_GIT`), not allowlist entries, so S2 moves
  the helper by changing one constant. A stale entry fails so the list shrinks with each
  migration. No expiry dates yet: that is an Open question for the repository owner, and the
  entry shape (`Allowed`) leaves room for one.
- **2026-11-23** — S1's tag check reads literals only. A tag dict that is neither a literal at the
  call site nor a name bound to one in the same or an enclosing scope (keys added with
  `tags["k"] = v` before the call count) is listed in `TAGS_UNCHECKED` with a reason. That the
  shared logger writes the seven tags is asserted once, by `tests/drawing/test_tracking.py`, not
  repeated in `test_conventions.py`. The owner scan counts a dict display that has a literal
  `"owner"` and at least three other required tag keys, or a name bound to one that later gets
  `name["owner"] = "<str>"`, so a config dict is not a tag dict. `set_tags` is read only on
  `mlflow` and `self._mlflow` and a literal there is held to all seven tags; a dict changed by
  `update`, `setdefault` or `|=` is unreadable. Today the unchecked list covers only
  `story/grid.py` (tags arrive as a parameter); `drawing/sensitivity.py` was listed until #46 merged.

## Verified facts

- **2026-11-22** — On `origin/main`, `mlflow.start_run(` appears in `drawing/tracking.py`,
  `drawing/sensitivity.py`, `speech/train.py` (two sites), `story/grid.py` (two sites) and
  `illustration/consistency/experiment.py`. Checked with `git grep` over `src` and `scripts`.
  (`drawing/sensitivity.py`'s was removed by #46.)
- **2026-11-22** — On `origin/main`, a private or public git-sha or git-state function is defined in five
  places: `drawing/evaluate.py` (`_git_state`), `evaluation/golden.py` (`_git_sha`), `speech/evaluate.py`,
  `story/grid.py`, and `illustration/consistency/experiment.py`. Checked the same way.
- **2026-11-22** — `drawing/tracking.py` defines `run_tags` (the seven required tags plus `kind`,
  `git_dirty` and `gate_eligible`; `owner` is required and must be non-empty), `run_params` and
  `log_run`.

- **2026-11-23** — Re-checked on `origin/main` at `a38c7ab` (after #39 and #44 merged), with
  `git grep` and then S1's `ast` scans. The 2026-11-22 facts still hold, with three details the
  scans have to handle:
  - **Call sites.** There are seven `.start_run(` calls. `story/grid.py` makes both of its calls
    through an attribute (`self._mlflow.start_run`), so the scan matches any `.start_run`, not only
    `mlflow.start_run`. `speech/train.py:350` re-opens a run by `run_id` and sets no tags.
  - **Where tags are set.** `illustration/consistency/experiment.py` and `story/grid.py` set tags
    with `set_tags(...)` inside the run.
  - **Tag dicts and owners.** Every literal tag dict (drawing, speech, illustration) has all seven
    required tags. The only literal owner was `drawing/sensitivity.py:439`, removed by #46.
- **2026-11-23** — `src/primer/data/versioning.py:40` (`_git_commit`) is a sixth place that runs
  `git rev-parse HEAD`, and S1's scan does not see it because its name is not `*git_sha*` or
  `*git_state*`. It is deliberately out of scope: it records the commit of a dataset version
  (`"git_commit"`, a full sha), not the provenance of a model run. S5 lists it.
- **2026-11-23** — S1's tests were re-run against #46's head `3db5f56` (merged into a scratch tree
  with this branch's `tests/test_conventions.py`): exactly three fail, the `drawing/sensitivity.py`
  `start_run`, `TAGS_UNCHECKED` and `OWNER_ALLOWED` entries. This corrects the S0 sha above: the
  earlier check was at `2f418ed`, and #46 has since added tests, a plan note and a stricter
  occlusion read in `tracking.py`.
- **2026-11-23** — #46 merged to `main` as `2d3f077`. S1 merged `main` and deleted the three
  `drawing/sensitivity.py` entries (`start_run`, `TAGS_UNCHECKED`, `OWNER_ALLOWED`); the scans then
  pass on the merged tree with no `drawing/sensitivity.py` allowance.
- **2026-11-23** — The Jira keys for the allowlist come from each file's commit history:
  - illustration consistency: KAN-103;
  - golden runner: KAN-89;
  - story grid: KAN-39;
  - speech git state: KAN-97, and speech logging: KAN-128.

## Open questions

- **Where does the shared git-state helper live?** Candidates: a new `src/primer/_git.py`, or
  `primer/evaluation/`. *Recommended:* a leaf module with no model-area imports, chosen in S2. *Decides:*
  the repository owner.
- **Should the other areas' logging move onto one shared helper, not only the drawing one?**
  `tracking.py` is drawing-specific (`STAGE = "drawing"`). *Recommended:* a separate slice after S1,
  led by whoever owns speech logging (KAN-128), generalising the helper rather than copying it.
  *Decides:* the owners of speech, story and illustration.
- **Should S1's allowlist entries expire?** *Recommended:* yes, a date in each entry, so a forgotten
  exception shows up. *Decides:* the repository owner.
- **Is `interface.py` stable enough for S7 now?** *Recommended:* check before starting; if not, S7
  waits. *Decides:* the owner of serving.

## Out of scope

- Changing who reviews or how many rounds a review runs. The loop for that is in `autonomous-loops`.
- Rewriting the speech, story or illustration logging in this plan's first slices.
- Any model or data change. This plan touches process, docs, skills and tests.
