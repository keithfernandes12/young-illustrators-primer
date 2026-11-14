/goal `python .claude/skills/golden-gate/scripts/check_gate.py --metrics .local-data/eval/golden_metrics.json --only <group>`
prints GATE_RESULT: PASS for group <group> (story | readability | safety | drawing | illustration),
or a training run is required, whichever comes first.

**You do not train.** `docs/adrs/where-model-training-happens.md` is explicit: agent sessions
never launch a training job. When the gate can only be moved by training, stop and hand over:
prepare the config with train-vertex, state the hypothesis and the one thing it changes, print
the exact command, and say that a person runs it. Monitoring a job someone else started, and
re-running the gate on its metrics, is yours.

Everything short of a launch is in scope: prompts, decoding constraints, thresholds you are
allowed to touch, data selection, and evaluation itself. Record each attempt's hypothesis and
result as a dated Verified fact in the plan. Never edit thresholds.json or configs/golden/.
Stop after 40 turns.
