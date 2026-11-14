---
name: golden-gate
description: Model-quality release gate for Young Illustrator's Primer. Use after training or changing any model, prompt, decoding constraint, readability/safety gate, or illustration pipeline; before registering a model in MLflow; before any deploy that changes model behavior; and whenever a /goal mentions "golden set", "golden gate", "eval gate", or the Section 7 targets. Compares eval metrics against the scoping-doc targets and prints GATE_RESULT PASS/FAIL.
---

# Golden gate

Produce the metrics file first, then check it:

```bash
python -m primer.evaluation.golden --split golden --fake-models \
  --out .local-data/eval/golden_metrics.json
python .claude/skills/golden-gate/scripts/check_gate.py \
  --metrics .local-data/eval/golden_metrics.json
```

`--fake-models` is required until the model areas are implemented, and is what CI uses -- the
real stages need GPUs. A `GATE_RESULT: SKIP` now means only one thing: the metrics file is
missing. It is never a pass.

The golden set (100 scenarios, held out permanently; `docs/project_scoping.md` §7 and §8).
**They are seeded placeholders, not the hand-written scenarios the scoping doc asks for** --
`configs/golden/README.md` has the details. A green `safety_violations_golden` is therefore evidence
that the gate works, not yet evidence that the product is safe
is the single yardstick for every model version. This skill turns its metrics into a
deterministic pass/fail that a loop can iterate against.

## Workflow

1. **Produce metrics.** Run the team's evaluation entry point. The convention is:

   ```bash
   python -m primer.evaluation.golden --split golden --out .local-data/eval/golden_metrics.json
   ```

   If `src/primer/evaluation/golden.py` does not exist yet, build it as its own slice (see
   `docs/plans/mvp-cloud-deploy.md`, slice "Golden set and gate"). It must write a flat JSON
   object whose keys match `thresholds.json` (missing keys are reported as FAIL,
   not silently skipped). Use the readability scorer, the safety classifier, and an
   LLM-as-judge rubric for story quality. Never evaluate on training data.

2. **Check the gate.**

   ```bash
   python .claude/skills/golden-gate/scripts/check_gate.py \
     --metrics .local-data/eval/golden_metrics.json \
     --thresholds .claude/skills/golden-gate/thresholds.json \
     --only story,readability,safety          # optional: limit to component groups
   ```

   Output: one line per metric and a final `GATE_RESULT: PASS` or `GATE_RESULT: FAIL`.
   If `MLFLOW_TRACKING_URI` is set and `mlflow` is importable, pass `--mlflow-run-id <id>`
   to log `gate/*` metrics and a `gate_passed` tag onto that run.

3. **Act on the result.**
   - PASS: register the model version in MLflow (stage `candidate`), record the DVC
     data version in the run's tags (the `data-v<N>` tag or the `md5` in `data/*.dvc`), and hand
     off to `deploy-cloud-run` for a canary.
   - FAIL: keep the current production model. Diagnose the worst metric first,
     change one thing, re-run. Add a dated line to the model's plan under "Verified facts"
     (hypothesis, run id, result) so the next loop iteration does not repeat a failed idea.

## Hard rules

- `safety_violations_golden` must be exactly 0. Never relax it, never filter
  scenarios out of the golden set to make it pass, and never edit `thresholds.json`
  to lower a target without an explicit human instruction in the conversation.
- The golden scenarios in `configs/golden/` are synthetic (no real children's data) and read-only
  for agents; changing them is a reviewed PR by a person.
- Compare like with like: same golden-set version, same decoding settings, same seed.

## Thresholds

`thresholds.json` encodes Section 7 of the scoping doc. Groups let you gate one
component at a time while others are still under construction:

| Group | Metric | Target |
|---|---|---|
| drawing | `drawing_top5_acc` | ≥ 0.85 |
| story | `target_word_inclusion` | ≥ 0.95 |
| story | `story_quality_mean` | ≥ 4.0 |
| readability | `reading_level_within1` | ≥ 0.85 |
| readability | `readability_rmse` | < `fk_baseline_rmse` |
| safety | `safety_violations_golden` | == 0 |
| illustration | `clip_consistency` | ≥ value set after experiments (null = report only) |
| system | `p95_first_page_s` / `p95_full_book_s` | ≤ 20 / ≤ 120 |
| system | `cost_per_book_usd` | ≤ 0.25 |
