"""Evaluate the readability model. CLI: --model --split --out metrics.json.

Writes the shape in ``primer.evaluation.contract``: ``overall`` carries this stage's gated keys,
``baseline`` is what the model must beat and is never omitted, ``slices`` carries the per-slice
breakdown. ``primer.evaluation.golden`` merges these into the file the release gate reads.
"""

from __future__ import annotations

import argparse
import sys

from primer.evaluation import contract

#: The gated keys this stage owns, from the shared contract.
OWNED_METRICS = tuple(k for k, owner in contract.METRIC_OWNERS.items() if owner == "readability")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.parse_args(argv)
    raise NotImplementedError("readability/evaluate.py is a stub")


if __name__ == "__main__":
    sys.exit(main())
