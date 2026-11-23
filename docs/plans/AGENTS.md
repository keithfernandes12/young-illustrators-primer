# docs/plans

A plan says how we will build something already decided, as PR-sized slices, and is the running
record of what was built. Later agents continue or extend the work from here, so keep it true.

- Start from `_template.md`. Frontmatter rules: `docs/engineering/documentation.md`.
- `type: plan`; `status: draft` → `active` (work started) → `closed` (all slices done or dropped),
  or `superseded` (set `superseded_by` and say at the top where each unfinished slice went).
- Read the whole plan before starting a slice. Work on one slice per PR.
- Slice labels are per plan, so S1 means different things in different plans. Inside a plan a bare
  label is fine. Anywhere else (a PR title or body, a commit, a Jira comment, another plan, a
  memory note) write the plan's slug in front: `mvp-cloud-deploy S6f`, `cross-slice-coherence S1`.
- Merge order follows dependencies, not readiness: a slice that adds a shared helper lands before
  the slices that should use it, and they list it under *Depends on*. Before handing a slice off,
  reconcile it with `main` (`docs/engineering/engineering-loop.md` step 6).
- In the same PR as the code: mark the slice in the status table with the PR number, and change the
  plan wherever the design changed. Never keep a second plan elsewhere.
- "Decisions" and "Verified facts" are dated and append-only; correct a wrong entry with a new dated
  entry below it.
- One PR's worth of work needs no plan; the PR body carries its contract.
