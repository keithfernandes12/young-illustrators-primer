---
title: Agents may train locally within a cap; cloud launches and registration stay with people
type: adr
created: 2026-11-14
status: accepted
author: rishiparekh
jira: [KAN-38]
tags: [agents, models, training, gcp, mlflow]
supersedes: [where-model-training-happens]
related: [autonomous-loops, mvp-cloud-deploy]
---

# Agents may train locally within a cap; cloud launches and registration stay with people

## Context

`where-model-training-happens` drew its line around the **actor**: agent sessions never train, "not
locally, not on Vertex". Six weeks of building the loop kit, and the first attempt to point a loop
at a model stage, showed that the line was drawn in the wrong place.

The ADR gave two reasons. Neither one is about local compute:

1. **Unattended spend.** "An agent session left alone can start a job that costs real credits, and
   nobody is watching the loss curve. Credits attach to a billing account and do not come back."
2. **Reproducibility.** "Six members training on six laptops... produces six numbers that cannot be
   compared with each other or with what the gate measured."

Reason 1 is entirely about paid cloud compute. A training run on the machine the session is already
running on bills nobody. Reason 2 was never a restriction on *training* at all — the original ADR
says so itself: "A laptop run can motivate a registration; it cannot be one." It binds what enters
the registry, which clause 3 below keeps exactly as it was.

What the blanket rule cost us is concrete. A loop pointed at a model stage builds `data.py`,
`model.py`, `train.py`, `evaluate.py` and the tests, and then stops — because the next thing that
would move `drawing_top5_acc` is a training run, and the agent may not do the one step that tells it
whether any of what it just wrote actually works. It cannot tell a wiring bug from a bad
hyperparameter without running a few hundred steps. So it hands over a stage that has never trained,
and the person who takes it over debugs the agent's code by hand, which is the work the loop existed
to absorb.

Meanwhile the cloud half is no longer held up by instructions at all. Since the loop kit landed,
`.claude/settings.json` denies `gcloud ai custom-jobs create`, `gcloud ai custom-jobs cancel` and
`gcloud run jobs execute`, and `guard_cloud.py` refuses those operations on their own tokens —
through compound commands, subshells, command substitutions, nested shells with clustered short
options, and interpreters whose inline code names a cloud tool. The expensive thing is now stopped
by a mechanism. The free thing is stopped by a sentence.

## Options considered

**Leave it as it is.** No work, and no new ways to be wrong. Keeps a rule whose stated justification
does not cover the case it blocks, and keeps every model loop stopping one step before the step that
would tell it whether its own code works.

**Drop the actor distinction and let agents train anywhere.** Simplest to state. Throws away the
reason that is still good: an unattended session with cloud credentials and a bad config can spend
real money overnight with nobody reading the loss curve.

**Bound the permission by cost and by time, not by actor (chosen).** Agents may train where it is
free and interruptible, and may not where it is billed. Costs us a cap that every area's `train.py`
has to honour, and one more distinction to explain.

## Decision

**An agent session may run bounded local training. It never launches a cloud training job. A member
trains freely on their own machine. A registered run must still be reproducible on Vertex AI.**

Four clauses:

1. **Agents never launch a cloud training job.** Vertex AI custom jobs, hyperparameter tuning jobs
   and Cloud Run jobs are a person's action, whatever the project and whoever is billed. This is
   unchanged, and it is the clause the `deny` rules and `guard_cloud.py` enforce.
2. **An agent may train locally, within a cap.** On the machine the session is running on, bounded
   two ways: a step cap the area's config carries (`max_steps`), and a **30-minute wall clock**. An
   area whose `train.py` cannot honour both must gain that before an agent runs it — a run an agent
   cannot stop is the thing this clause is trying not to create. The result is working notes: record
   it as a dated Verified fact in the plan, with the hypothesis and what it changed.
3. **A member may train on their own machine.** No cap, no approval, no ceremony. Unchanged.
4. **A registered run must be reproducible.** Unchanged from the superseded ADR: an MLflow
   `candidate`, `production` or `rollback` alias requires a run from the area's container, on Vertex
   AI, against the frozen data version, logged to the shared MLflow with the tags
   `model-development.md` lists. A laptop run — a member's or an agent's — can motivate a
   registration; it cannot be one.

The cap is the same for everyone. A per-member switch is one more thing to configure wrong, and the
local case carries no shared cost, so there is nothing for a per-member setting to protect.

## Consequences

**Easier.** A model loop can now close its own feedback loop: write the training step, run a few
hundred of them, see the loss move or not, and fix its own wiring before handing over. The handover
is a stage that has trained once, not a stage that has only ever been type-checked.

**Harder.** Every area's `train.py` needs a real `max_steps` and must be interruptible before an
agent may run it. A capped run proves wiring, not quality — nobody should read a 30-minute laptop
number as evidence about the model, and the Verified fact should say which it is.

**Unchanged, and worth saying plainly.** Nothing here lets an agent spend money. Clause 1 is the same
sentence it always was, and it is the one with a mechanism behind it.

**Others must now:**

- `AGENTS.md` states the four clauses and links here instead of to the superseded ADR.
- `train-vertex` keeps "an agent launches nothing in the cloud" and stops citing the superseded ADR
  to forbid local runs.
- `.claude/loops/goal-golden-gate.md` lets the loop train locally inside the cap and hand over only
  when the next step needs a GPU job.
- `guard_cloud.py`'s refusal says *cloud* training, since local training is no longer refused and
  the message is what an agent reads when it is stopped.
- A slice that cannot be proven inside the cap stays marked `person-only` in its plan.
