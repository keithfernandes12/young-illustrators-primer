#!/usr/bin/env python3
"""Compare eval metrics with YIP release thresholds. Prints GATE_RESULT: PASS|FAIL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: math.isclose(a, b, abs_tol=1e-12),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument(
        "--thresholds",
        default=Path(__file__).resolve().parent.parent / "thresholds.json",
        type=Path,
    )
    ap.add_argument(
        "--only", default="", help="comma-separated groups to gate, e.g. story,safety"
    )
    ap.add_argument("--mlflow-run-id", default=None)
    args = ap.parse_args()

    if not args.metrics.exists():
        # primer.evaluation does not exist yet, so nothing produces this file. Say so plainly
        # instead of raising FileNotFoundError, and never report SKIP as PASS.
        print(f"===== YIP GOLDEN GATE ({args.metrics}) =====")
        print(
            f"GATE_RESULT: SKIP (no metrics at {args.metrics}; run the evaluation first. "
            "primer.evaluation is not built yet -- see docs/plans/autonomous-loops.md)"
        )
        return 0
    metrics = json.loads(args.metrics.read_text())
    spec = json.loads(args.thresholds.read_text())["metrics"]
    groups = {g.strip() for g in args.only.split(",") if g.strip()}

    failures, rows, logged = 0, [], {}
    for m in spec:
        if groups and m["group"] not in groups:
            continue
        key, op = m["key"], m["op"]
        target = metrics.get(m["ref"]) if "ref" in m else m.get("value")
        actual = metrics.get(key)
        label = f"{key} {op} {m.get('ref', target)}"
        if actual is None:
            rows.append(f"FAIL  {label}: metric missing from {args.metrics}")
            failures += 1
            continue
        if target is None and "ref" in m:
            rows.append(
                f"FAIL  {label}: reference metric {m['ref']} missing from {args.metrics}"
            )
            failures += 1
            continue
        if target is None:
            rows.append(f"INFO  {key} = {actual} (no threshold set yet; report only)")
            continue
        ok = OPS[op](float(actual), float(target))
        rows.append(
            f"{'PASS' if ok else 'FAIL'}  {label}: actual={actual} target={target}"
        )
        logged[f"gate/{key}"] = float(actual)
        failures += 0 if ok else 1

    print(f"===== YIP GOLDEN GATE ({args.metrics}) =====")
    print("\n".join(rows) if rows else "no metrics selected")
    passed = failures == 0 and bool(rows)

    if args.mlflow_run_id:
        # The golden runner marks synthetic scores with `fake_models: true`. Certifying a real
        # MLflow run from them would put gate_passed=true on a model nothing measured, and a later
        # registration reads that tag. Running the gate on fakes is still useful -- it is how we
        # test the gate -- so refuse only the certification.
        if metrics.get("fake_models"):
            print(
                "REFUSED: these metrics are from --fake-models, so they cannot certify MLflow run "
                f"{args.mlflow_run_id}. Evaluate the real candidate first."
            )
        else:
            try:
                import mlflow  # type: ignore

                with mlflow.start_run(run_id=args.mlflow_run_id):
                    mlflow.log_metrics(logged)
                    mlflow.set_tag("gate_passed", str(passed).lower())
                print(f"logged gate results to MLflow run {args.mlflow_run_id}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not log to MLflow: {exc}")

    print(f"GATE_RESULT: {'PASS' if passed else f'FAIL ({failures} failing)'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
