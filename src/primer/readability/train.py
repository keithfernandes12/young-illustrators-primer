"""Train the readability model. CLI: --config --data-version --out.

Logs exactly one MLflow run, with the tags `model-development.md` lists. An agent session never
runs this: see docs/adrs/where-model-training-happens.md.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true", help="import-and-one-step check on CPU")
    parser.parse_args(argv)
    raise NotImplementedError("readability/train.py is a stub")


if __name__ == "__main__":
    sys.exit(main())
