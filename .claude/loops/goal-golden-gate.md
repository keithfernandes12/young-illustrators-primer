/goal `python .claude/skills/golden-gate/scripts/check_gate.py --metrics .local-data/eval/golden_metrics.json --only <group>`
prints GATE_RESULT: PASS for group <group> (story | readability | safety | drawing | illustration),
or a training run is required, whichever comes first.

**You never start a job in the cloud.** `docs/adrs/bounded-local-training-for-agents.md`: Vertex
and Cloud Run jobs are a person's action. When the gate can only be moved by a cloud run, stop and
hand over: prepare the config with train-vertex, state the hypothesis and the one thing it changes,
print the exact command, and say that a person runs it. Monitoring a job someone else started, and
re-running the gate on its metrics, is yours.

**You may train locally, within the cap.** Same ADR: on this machine, inside the area's `max_steps`
and 30 minutes of wall clock. Use it to find out whether your own code works — whether the loss
moves, whether the data loader feeds the model what you think — before handing anything over. A
capped run is evidence about wiring, never about model quality, and the Verified fact you write
must say which it is. If the area's `train.py` cannot honour both bounds, add that first or mark
the slice `person-only`.

Everything short of a cloud launch is in scope: prompts, decoding constraints, thresholds you are
allowed to touch, data selection, and evaluation itself. Record each attempt's hypothesis and
result as a dated Verified fact in the plan. Never edit thresholds.json or configs/golden/.
Stop after 40 turns.
