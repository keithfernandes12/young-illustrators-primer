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
- The 2026-11-24 prep also read `docs/engineering/review-checklist.md` (the categories table), the
  `review-pr` skill (steps 2 and 7), `.claude/skills/verify/scripts/verify.py`,
  `src/primer/evaluation/golden.py` and `tests/models/test_area_interfaces.py`.
- Not read: S3 read `loop-worker` and `engineering-loop` for its own PR. Assumed: the skills can be
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

> One slice = one PR. S0, S1 and S3 have merged (#46, #47, #48, #49). Build order after the
> 2026-11-24 prep: S5 and S8 (no dependency), then S6 (uses S8), S4, and S2 once its home is
> decided. S7 is mostly covered already; narrow it or close it.

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
- **Prep (2026-11-24)** — ready to build once the Open question on its home is answered.
  - **The helper takes the caller's package folder.** `_git_state` runs git with `cwd` set to its
    own folder and scopes the dirty check to it (`git status --porcelain -- .`). Moved unchanged
    into a shared module, it would report whether *that module's* folder is dirty, not the
    caller's package. The shared function is therefore
    `git_state(package: Path) -> tuple[str, bool | None]`, and each caller passes
    `Path(__file__).parent`.
  - **`golden.py`'s copy is different.** It runs at the repository root and has no dirty flag.
    Switching it means recording `git_dirty` in the golden metrics file. That is additive:
    `check_gate.py` reads only the keys `thresholds.json` names and `fake_models`, and nothing
    reads `git_sha` or `git_dirty` today, so no reader needs updating. Its dirty flag is scoped to
    the folder it is given: `golden.py` passes `Path(__file__).parent`, so it reports whether
    `src/primer/evaluation` is dirty, where its copy today runs at the repository root. *Recommended:* switch it in S2, add
    `git_dirty`, and tell the golden-gate owner (KAN-89).
  - **Patch points.** Tests patch `evaluate._git_state` and `sensitivity._git_state`
    (`tests/drawing/test_stability.py`, `tests/drawing/test_sensitivity.py`). Do not keep a thin
    `def _git_state` in each caller: S1's scan flags every function named for git state except
    `SHARED_GIT`, so the wrapper would turn S2's own check red. Call the helper through its module
    (`_git.git_state(...)`) and move the patches to `primer._git.git_state` in the same PR, which
    covers both callers; or bind the old name by assignment (`_git_state = functools.partial(...)`),
    which a `FunctionDef` scan does not see.
  - **`data/versioning.py:_git_commit` stays out.** It records a dataset version's full commit,
    not a run's provenance (Verified facts, 2026-11-23).
  - **Proof table:**

    | File | Covered by |
    |---|---|
    | the new helper module | `tests/drawing/test_top5.py::test_the_recorded_commit_is_the_packages_not_the_callers` and `::test_git_dirty_follows_git_status` (which asserts the `cwd` is the module's folder), both moved to point at it and asserting the folder passed in; plus a test that two packages get their own dirty flag |
    | `drawing/evaluate.py`, `drawing/sensitivity.py` | their existing CLI tests, whose patches still apply |
    | `evaluation/golden.py` | `tests/evaluation/` plus a check that the metrics file carries `git_dirty` |
    | `tests/test_conventions.py` | `SHARED_GIT` points at the new home; the `golden.py` entry is deleted, so the stale check proves the copy is gone |

### S3 · The reconcile-with-main step, in the loop skills

*size: small*

Add one step to `engineering-loop` and `loop-worker`, before "mark ready": fetch `main`, list what
landed since the branch point (`git log --stat <merge-base>..origin/main`), and for every new module
or helper in the area of this slice, reuse it or say in the PR body why not. Also add the
merge-order rule (Decisions) to the plan template's Slices section: a slice that introduces a shared
helper lands before any slice that should use it, and the consumer lists it under *Depends on*.

- **Why separate** — a docs and skills change with no code.
- **Touches** — `.claude/skills/engineering-loop/`, `.claude/skills/loop-worker/`,
  `.claude/skills/create-pr/`, `docs/plans/_template.md`, `docs/plans/AGENTS.md`,
  `docs/engineering/engineering-loop.md`, `docs/engineering/onboarding.md` (step count), the root
  `AGENTS.md` summary, and `.claude/loops/goal-drawing-stage.md` and `ralph-overnight.md`.
- **Acceptance** — the step is in both skills; the PR template has a line "Reconciled with `main` at
  `<sha>`: reused / not applicable / differs because …".
- **Testing** — `scripts/check_docs.py` passes; none otherwise.
- **Depends on** — nothing to build; merges after S1, whose test file and this plan its text links.
  **Jira** — KAN-38.
- **Added (2026-11-23, at the KAN-38 owner's request)** — the slice-naming rule: outside its own
  plan, a slice is written with the plan's slug in front (`mvp-cloud-deploy S6f`,
  `cross-slice-coherence S1`), in `docs/plans/AGENTS.md`, the plan template and `create-pr`.
  `loop-worker`'s `LOOP_*` end line keeps a bare label, because loop prompts read that format.
- **Added (2026-11-24, from the independent review)** — the unattended worker only waits: a
  slice whose helper is still in an open PR is not ready, and when that is the only reason nothing
  is ready the worker prints `LOOP_WAIT <slice> waiting-on-#<pr>`, not `LOOP_EMPTY`, which the
  Ralph prompt reads as "plan finished". Both loop prompts say so. See the 2026-11-24 Decision.
- **Split (2026-11-23)** — the PR-template line goes in its own one-line PR, because `AGENTS.md`
  puts `.github/` changes in their own PR. That PR is #49 (opened 2026-11-24). Until it merges,
  `create-pr` asks for the line at the end of "How it was checked".

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
- **Prep (2026-11-24)** — ready; S3 merged (#48, #49). Docs and skill only.
  - **The checklist** (`docs/engineering/review-checklist.md`, *The change*) gets one line, worded
    as S3 words it: "Does this PR re-implement something now in `main`, or add something an open
    sibling PR also adds?" The *Finding categories* table already says what a **Fix** is; the line
    names this case as one.
  - **The skill** (`review-pr`, step 2) gets the two inputs as commands the reviewer runs:
    `gh pr list --state open --limit 100 --json number,title,headRefName,baseRefName,files` and
    `git log --stat $(git merge-base origin/main <O>^)..origin/main`, where `<O>` is
    `git rev-list --no-merges origin/main..<head> | tail -1`, the oldest commit that is not on
    `main`. `git merge-base <head> origin/main` is wrong here: once the branch has merged `main`
    it is `main`'s tip and the range is empty, which hides what the merge brought in. The
    `<O>` form also works for a stacked PR, whose first own commit's parent is already a lower
    PR's commit. The list includes the PR itself and its own base chain; those are `Depends on`,
    not siblings, and the reviewer skips them (a stack reads as "stack, expected"). Step 7
    (follow-up mode) reviews only the PR's own commits and skips merges of `main`, which is exactly
    where a new helper arrives, so it asks the cross-slice question again after any merge of `main`,
    with the same range. A PR's own change is `git diff origin/<baseRefName>...<head>`, not
    `origin/main...<head>`, which for a stacked PR is the whole stack.
  - **Naming.** Findings name slices with their plan's slug (S3's rule).
  - **Acceptance** — both files carry the question; the skill carries both commands and the
    follow-up rule; `scripts/check_docs.py` passes.

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
- **Prep (2026-11-24)** — ready to build; no dependency.
  - **`verify` has no warn-only path today.** Every check either passes or goes into `failed`
    (`.claude/skills/verify/scripts/verify.py`, `main`). S5 adds one: an advisory check whose
    output is printed under `WARN  duplicates` and never appended to `failed`, so `VERIFY_RESULT`
    cannot change. A test pins that: the scan reporting findings still leaves verify at PASS.
  - **The scan reads `src/` and `scripts/` with `ast`,** like S1, and reports three kinds:
    - private functions with the same name, or the same normalised body (the `ast.dump` of the
      body with docstrings removed), in two or more modules;
    - a string literal repeated in more than three modules (for example `"yip-drawing"` or a tag
      key);
    - the S1 blind spots listed above, and the two call shapes the acceptance names: a function
      named `git_(sha|state|commit)` (private or public) or a `subprocess` call whose arguments
      include `rev-parse`, and every `.start_run` call site.
    A site is *known* when S1 already accounts for it, so the output separates known from new:
    `GIT_ALLOWED` and `SHARED_GIT` for git state, `START_RUN_ALLOWED` and `SHARED_LOGGER` for
    `start_run`. `data/versioning.py:_git_commit` is on none of S1's lists, because S1 skips it on
    purpose (`tests/test_conventions.py`, module docstring), so S5 keeps its own one-entry
    allowlist for it, with the reason.
  - **Proof table:**

    | File | Covered by |
    |---|---|
    | `scripts/scan_duplicates.py` | `tests/test_scan_duplicates.py`: fixture trees for each kind (two identical private helpers reported, one not; a literal in four modules reported, in three not; each S1 blind spot; a git-named function and a `start_run` site, each marked known when S1 lists it and new when not) |
    | `verify.py` | a test that an advisory finding prints `WARN` and leaves the result `PASS` |

  - **Acceptance, measured on `main` at `c331364`:** the six git-state copies (S1's five plus
    `data/versioning.py:_git_commit`) and the five `start_run` sites outside `tracking.py` (speech 2, story 2, illustration 1) appear,
    marked known.

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
- **Prep (2026-11-24)** — ready to build; best after S8, whose script it calls.
  - **Inputs the skill gathers itself:** `tests/test_conventions.py` (known exceptions), S5's
    scan when it exists, S8's audit when it exists, every plan's Status table, and the stage
    contracts (`src/primer/<area>/interface.py`) with `serving/book.py`, which joins them.
  - **Output** — a dated brief, `docs/briefs/seams-<date>.md` (`type: brief`, `status: draft`).
    Each finding is either a new slice in this plan or a Jira key. Nothing is filed without the
    person's go-ahead, because creating Jira issues is an `ask` action.
  - **A first finding, found while preparing it.** `src/primer/evaluation/golden.py` says its
    fakes' mismatch with the area Protocols "is recorded in docs/plans/autonomous-loops.md", but
    that plan does not mention it. It is recorded stage by stage and not together: for drawing in
    `mvp-cloud-deploy` (`top5(true_label)` is handed the answer), for story in
    `story-generation-llm` (S5 adds an adapter). No plan records the seam as a whole. The first
    report should carry it.

### S7 · A seam smoke test

*size: medium*

One end-to-end test with fakes that runs each stage's contract in order (drawing recognition, story,
illustration) and asserts that each stage's output type is what the next reads. It catches name and
schema mismatches that per-slice tests cannot.

- **Why separate** — it spans areas and belongs to the serving slice's owner as well.
- **Acceptance** — passes on `main`; renaming a field in one stage's output type fails it.
- **Depends on** — the stage contracts, which are per area (`src/primer/<area>/interface.py`;
  there is no `src/primer/interface.py`). **Jira** — to create.
- **Prep (2026-11-24): mostly built already.** `tests/models/test_area_interfaces.py::test_serving_builds_a_complete_fake_book`
  runs drawing → story → readability → safety → illustration → speech through
  `serving.book.build_book` with each area's Fake. Renaming `StoryRequest.description` makes it
  fail (checked, then reverted), so the acceptance above already holds on `main`.
  - **The seam it misses** is the golden runner's own fakes (`evaluation/fakes.py`). Their
    signatures differ from the area Protocols (`drawing.top5(label)` is handed the answer), so a
    real area cannot be plugged into `golden.py`, and nothing tests the two against each other.
  - *Recommended:* narrow S7 to that seam: one test that each `FakeComponents` member can be
    replaced by the area's Fake through an adapter, or a recorded decision to retire
    `evaluation/fakes.py`. Either way it is a KAN-89 (golden runner) question first.
    *Decides:* the golden-gate owner.

### S8 · An open-questions audit

*size: small*

A script that reads every plan's Open questions and hand-offs and lists those with neither a slice
nor a Jira key in the text. S5 and S6 findings use it as the place a gap becomes work.

- **Acceptance** — on today's plans it lists the questions with no key; a question given a key
  disappears from the list.
- **Depends on** — nothing. **Jira** — to create.
- **Prep (2026-11-24)** — ready to build; no dependency.
  - **What counts as tracked.** A bullet under `## Open questions` is tracked if it contains a
    `KAN-<n>` key or a slice label. The probe below used that rule; a stricter rule (a
    `*Tracked:*` marker, as `mvp-cloud-deploy` already uses) would list more.
    *Recommended:* start with key-or-slice, so the first list is short and true.
  - **Measured on `main` at `c331364`** with a scratch probe:
    - 37 open questions across the plans;
    - 13 without a Jira key;
    - **5 with neither a key nor a slice**: 4 in `mvp-cloud-deploy` (the `drawing` extra pin, L4
      quota region, course credit for Claude on Vertex, and when a SAM mask is too small to use) and 1 in `autonomous-loops`
      (subscription vs Vertex billing).
    - **Correction (2026-11-24, from the independent review of #51):** the probe read only `- `
      bullets. Three plans (`ai-coding-framework`, `ci-hardening`, `story-generation-llm`) write
      their questions as numbered lists. With those counted and struck-through (`~~`) questions
      skipped, the script reports **13 of 47** on `main` at `de4a121`. The 5 above are all still
      in the 13.
  - **Hand-offs.** `mvp-cloud-deploy`'s "What S6c hands to S6d" sit under Open questions, so the
    same scan covers them.
  - **Proof table:**

    | File | Covered by |
    |---|---|
    | `scripts/audit_open_questions.py` | `tests/test_audit_open_questions.py`: a fixture plan with one keyed, one slice-linked and one bare question lists only the bare one; adding a key removes it |

## Status

| Slice | State | PR | Notes |
|---|---|---|---|
| S0 | merged | [#46](https://github.com/keithfernandes12/young-illustrators-primer/pull/46) | confirmed against #46 at `2f418ed`, re-checked at `3db5f56`: tags built by `run_tags` (the CLI test asserts equality), owner from `--owner`, no literal owner, `git_dirty` `unknown`/`true`/`false`. Merged 2026-11-23 as `2d3f077` |
| S1 | merged | [#47](https://github.com/keithfernandes12/young-illustrators-primer/pull/47) | merged 2026-11-23 as `c331364`. `tests/test_conventions.py`: four scans, count-based allowlists with reasons and keys, stale entries fail |
| S2 | not started | | prepped 2026-11-24; S1 merged; waits on the Open question on the helper's home |
| S3 | merged | [#48](https://github.com/keithfernandes12/young-illustrators-primer/pull/48), [#49](https://github.com/keithfernandes12/young-illustrators-primer/pull/49) | reconcile step in `engineering-loop`, `loop-worker`, `create-pr` and the process doc; merge-order rule in the plan template and `docs/plans/AGENTS.md`. Needed #47 first (it links this plan and `tests/test_conventions.py`); #47 merged as `c331364`; #48 merged as `93d441e`. Also the slice-naming rule (plan slug in front, outside the plan). #48 merged as the skills and docs half; #49 adds the PR-template line (`.github/`), which goes in its own PR |
| S4 | not started | | prepped 2026-11-24; S3 merged (#48, #49), so ready |
| S5 | not started | | prepped 2026-11-24; ready |
| S6 | not started | | prepped 2026-11-24; ready, best after S8 |
| S7 | not started | | prepped 2026-11-24: mostly built already (`test_serving_builds_a_complete_fake_book`); narrow to the golden-fakes seam, or close |
| S8 | in review | [#51](https://github.com/keithfernandes12/young-illustrators-primer/pull/51) | `scripts/audit_open_questions.py`: lists open questions with neither a Jira key nor a slice; 13 of 47 on `main` (numbered lists counted, struck-through questions skipped). Built under KAN-38, the plan's key; a separate key was not created |

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
- **2026-11-24** — The `loop-worker` implements only the "wait" outcome of the 2026-11-22 merge-order
  Decision, not "ship with the gap recorded", because nobody is there to approve an allowlist entry.
  `LOOP_EMPTY` therefore means no slice is ready and none is waiting on an open PR; a wait ends in
  `LOOP_WAIT`. A `LOOP_WAIT` iteration uses one of the Ralph prompt's `--max-iterations`; that is
  accepted while a slice waits, and is an Open question for the owner if it proves wasteful.

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

- **2026-11-24** — Prep for S2 and S4–S8, read on `main` at `c331364`:
  - **Stage contracts are per area.** `src/primer/interface.py` does not exist; the contracts are
    in `src/primer/<area>/interface.py`, and `serving/book.py` composes them.
  - **The book test catches renames.** Renaming `StoryRequest.description` fails
    `test_serving_builds_a_complete_fake_book` (checked, then reverted).
  - **`verify.py` has no warn-only check type.**
  - **`_git_state` is package-scoped,** in both its `cwd` and its dirty check.
  - **`golden.py` points at a record that isn't there.** It says the fakes-versus-Protocols gap is
    recorded in `autonomous-loops.md`, and it is not.
  - **The open-questions probe:** 37 questions, 13 without a key, 5 with neither a key nor a
    slice, counting `- ` bullets only. The script, which also reads numbered lists, finds 13 of 47.

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
- **Narrow S7 to the golden-fakes seam, or close it?** The prep found that
  `src/primer/interface.py` does not exist and that the book test already covers renames
  (Verified facts, 2026-11-24). *Recommended:* narrow it, as the S7 prep says. *Decides:* the
  golden-gate owner (KAN-89).

## Out of scope

- Changing who reviews or how many rounds a review runs. The loop for that is in `autonomous-loops`.
- Rewriting the speech, story or illustration logging in this plan's first slices.
- Any model or data change. This plan touches process, docs, skills and tests.
