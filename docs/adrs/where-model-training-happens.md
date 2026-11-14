---
title: Agents never train; members train locally; registered runs are reproducible on Vertex AI
type: adr
created: 2026-11-13
updated: 2026-11-14
status: superseded
author: rishiparekh
jira: [KAN-89]
tags: [agents, models, training, gcp, mlflow]
superseded_by: [bounded-local-training-for-agents]
related: [autonomous-loops, mvp-cloud-deploy]
---

# Agents never train; members train locally; registered runs are reproducible on Vertex AI

> **Superseded on 2026-11-14 by `bounded-local-training-for-agents`.** Clause 1 below drew the line
> around the actor — "not locally, not on Vertex" — but both reasons this ADR gives are about paid
> cloud compute and about registration, neither of which covers a free, interruptible run on the
> machine the session is already on. The successor keeps clause 1 for cloud jobs, keeps clause 3
> untouched, and lets an agent train locally inside a step and wall-clock cap. The text below is
> kept as written, as the record of what was decided on 2026-11-13.

## Context

Six people are about to build six model areas in parallel, and the rules about where training may
happen did not agree with each other.

`AGENTS.md` said **"No local model training. Cap local pipeline runs with `--batch-size`; real runs
go to Vertex AI."** Read literally that forbids a member from training on their own machine, which
is not what anyone intends: members have laptops with GPUs and a two-minute experiment there is
worth more than a Vertex job. The `train-vertex` skill cited that line to restrict local work to a
CPU smoke test, which made the restriction look like a rule about *people* rather than about agents.

Meanwhile the thing that actually matters was written down nowhere: that a run which gets registered
must be one somebody else can reproduce. Six members training on six laptops, each with their own
CUDA version and their own idea of which data version they pinned, produces six numbers that cannot
be compared with each other or with what the gate measured.

The distinction that was missing is **who is running it**, not **where it runs**:

- An agent session left alone can start a job that costs real credits, and nobody is watching the
  loss curve. Credits attach to a billing account and do not come back.
- A member at their desk watching a run is exercising judgement the agent does not have.
- A run whose result goes in the registry is a claim about the model, and a claim has to be
  checkable.

## Options considered

**Forbid all local training.** Simple to state and simple to enforce. Costs every member the fast
feedback loop that makes model work tractable, and pushes trivial experiments onto paid GPUs. It is
also what the old wording appeared to say and what nobody actually did.

**Allow anyone, including agents, to train anywhere, and rely on the gate.** Maximum speed. The gate
catches a bad model but not a run nobody can reproduce, and not an unattended agent spending a
member's credits overnight on a misconfigured job.

**Separate the actor from the location (chosen).** Agents never train. Members train where they
like. Reproducibility is required only of runs that are registered, which is where the claim is
made. Slightly more to explain, and it needs the container and the frozen data version to be real
rather than aspirational.

## Decision

**Agent sessions never train. A member may train on their own machine. Any run that is registered
must be reproducible from the area's container on Vertex AI, with the same frozen data version,
logged to the shared MLflow.**

Three clauses, each doing one job:

1. **Agent sessions never train.** Not locally, not on Vertex. An agent may run the CPU smoke test
   in `train-vertex` — an import-and-one-step check that takes seconds — and may read, launch
   nothing, and report. Launching a training job is a person's action.
2. **A member may train on their own machine.** No cap, no approval, no ceremony. This is the fast
   loop, and its results are the member's own working notes.
3. **A registered run must be reproducible.** To put a model in the registry — an MLflow
   `candidate`, `production` or `rollback` alias — the run must come from the area's container, on
   Vertex AI, against the frozen data version the team pinned, logged to the shared MLflow with the
   tags `model-development.md` lists. A laptop run can motivate a registration; it cannot be one.

## Consequences

**Easier.** A member can experiment at their desk without asking anyone. Any registered number can
be re-run by another member from the container and the data version alone. Agent sessions have an
unambiguous answer — launching training is not theirs to do — so `train-vertex` can stop citing a
rule about local training to explain a restriction on agents.

**Harder.** Every area needs a working `Dockerfile` and a real pinned data version before its first
registration, rather than at some later tidy-up. A member whose laptop result looks good still has
to spend the Vertex run to register it, which costs credits and time.

**Others must now:**

- `AGENTS.md` states this rule rather than "No local model training", and links here.
- `train-vertex` cites this ADR for why an agent may only smoke-test, and states the three
  conditions a run must meet before `register.py` is allowed to run.
- Nothing in `.claude/` may launch a training job. `guard_cloud.py` already refuses the IAM and
  traffic verbs; it does not police training, because an agent that follows this ADR never gets
  there. If an agent is ever seen launching a job, the guard is where to add the refusal.
- The budget section of `docs/engineering/model-development.md` assumes members spend their own
  credits on their own compute project, which this ADR makes explicit.
