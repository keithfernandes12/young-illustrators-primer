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
        # A missing metrics file is not a pass and not a clean exit: nothing was measured.
        # It exits non-zero so `&&` chains and CI stop here rather than continuing.
        print(f"===== YIP GOLDEN GATE ({args.metrics}) =====")
        print(
            f"GATE_RESULT: SKIP (no metrics at {args.metrics}). Produce them first:\n"
            "  python -m primer.evaluation.golden --split golden --fake-models --out "
            f"{args.metrics}"
        )
        return 1
    metrics = json.loads(args.metrics.read_text())
    spec = json.loads(args.thresholds.read_text())["metrics"]
    groups = {g.strip() for g in args.only.split(",") if g.strip()}

    failures, rows, logged = 0, [], {}
    compared = 0  # rows where a real threshold was applied
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
        compared += 1
        ok = OPS[op](float(actual), float(target))
        rows.append(
            f"{'PASS' if ok else 'FAIL'}  {label}: actual={actual} target={target}"
        )
        logged[f"gate/{key}"] = float(actual)
        failures += 0 if ok else 1

    print(f"===== YIP GOLDEN GATE ({args.metrics}) =====")
    print("\n".join(rows) if rows else "no metrics selected")

    # A PASS has to mean something was measured against a threshold. Rows that only report a
    # value (a null threshold, such as clip_consistency) are not evidence, so a selection made
    # only of those -- `--only illustration` today -- is INCONCLUSIVE rather than PASS.
    passed = failures == 0 and compared > 0
    if failures == 0 and compared == 0:
        print(
            "GATE_RESULT: INCONCLUSIVE (nothing in this selection has a threshold yet; "
            "set one in thresholds.json or gate a different group)"
        )
        return 1

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

    # Synthetic scores prove the gate works, not that a model does. A bare PASS here is how a loop
    # goal ends, or a model gets registered, with nothing measured -- and no loop prompt matches
    # FAKE-PASS, which is the point.
    if metrics.get("fake_models"):
        print(
            f"GATE_RESULT: FAKE-{'PASS' if passed else 'FAIL'} "
            "(--fake-models metrics: the gate ran correctly, but no real model was scored. "
            "This is not a release signal.)"
        )
        return 0 if passed else 1

    print(f"GATE_RESULT: {'PASS' if passed else f'FAIL ({failures} failing)'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
