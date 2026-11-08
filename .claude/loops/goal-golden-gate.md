/goal `python .claude/skills/golden-gate/scripts/check_gate.py --metrics .local-data/eval/golden_metrics.json --only <group>`
prints GATE_RESULT: PASS for group <group> (story | readability | safety | drawing | illustration),
or 4 Vertex AI jobs have been launched in this goal, whichever comes first.
Use train-vertex for every training run, change one thing per run, and record each run's hypothesis and
result as a dated Verified fact in the plan. Never edit thresholds.json or configs/golden/.
Stop after 40 turns.
