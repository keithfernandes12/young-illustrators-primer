/loop Run one loop-worker iteration on docs/plans/mvp-cloud-deploy.md, restricted to slices S6a-S6e
(the drawing recognition stage, Jira KAN-38 and KAN-87). Only slices with a Jira key, not marked
person-only, whose dependencies are merged. If no slice is ready, output LOOP_EMPTY and stop the
loop.

**This prompt deliberately overrides `loop-worker`'s review-queue rule, which stops at 3 open
agent-authored PRs of any kind.** Count only the ones for this stage: stop with
`LOOP_WAIT review-queue-full` when **3 or more open agent-authored PRs carry KAN-38 or KAN-87**, or
when more than **5** are open across the whole repository. The reason for the change is that the
original count is repo-wide, so another member's open PR spends this stage's budget and a loop here
stops before doing anything — which is what happened the first time this prompt was used. The
reason for keeping a cap at all is unchanged and still binding: humans review, and more open PRs
make that worse, not better. Raising this further is a person's decision, not a loop's.

`AGENTS.md` is the rule set and `loop-worker` reads it at the start of every iteration; it is not
repeated here. What follows is only what is specific to this stage.

**Model choice is constrained by the local GPU, not by the benchmark.** The candidates and their
licences are in `docs/briefs/drawing-recognition-model-candidates.md`. Prefer the checkpoint that
fine-tunes comfortably in 8 GB of VRAM — the smaller tower, bf16, gradient checkpointing — and keep
anything that will not fit as inference-only. A candidate rejected on size is still a finding:
record the VRAM it actually needed as a dated Verified fact, measured rather than estimated, so
nobody re-tries it.

**Log every comparison to MLflow.** Two candidates are only comparable later if both runs carry
their params, metrics and the data version. A number that exists only in this transcript did not
happen, and this loop's transcript is not somewhere the next iteration can read.

**Run anything that touches the GPU from Git Bash.** PowerShell does not reliably see it; the
container path is in `docs/engineering/model-containers-on-windows.md`.

**Subagents, and what they are for.** One slice per iteration is deliberate — it is what keeps a PR
reviewable and what stops the review queue filling. So do not use subagents to work several slices
at once. Use them *inside* a slice, for work that is independent and whose result you need in this
turn: reading a model card or a paper, sweeping the codebase for a pattern, drafting the test cases
for a branch you are about to write, or a second read of what you just built. Brief each one with
the files and the question, never with your conclusion. If two slices are genuinely ready and
independent, that is the next iteration's job, not this one's.

**Training is capped and local.** `docs/adrs/bounded-local-training-for-agents.md`: inside the
area's `max_steps` and 30 minutes of wall clock, on this machine, never as a cloud job. S6d builds
that cap as part of itself, because no area has one yet. A capped run is evidence about wiring,
not about model quality, and the Verified fact must say which it is.

**S6b is ours now, and it is KAN-87's shared render function — build it once.** This stage took
ownership of the rendering, so `render_strokes` is yours to write, but the reason KAN-87 exists is
unchanged: **one** function, used by both training and inference, so the two cannot drift. Never
add a second renderer because the first is inconvenient to import, and never inline a quick
`PIL.ImageDraw` loop in a data loader or a test fixture. Same strokes and settings must always give
the same pixels; the settings live in config, not in a call site.

Record each attempt's hypothesis and result as a dated Verified fact in the plan. Stop after 30
turns.
