"""Train the safety model. CLI: --config --data-version --out.

Logs exactly one MLflow run, with the tags `model-development.md` lists. An agent session may
run this locally inside the cap -- `max_steps` and 30 minutes -- to check the wiring, and never
launches it as a cloud job: see docs/adrs/bounded-local-training-for-agents.md.
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
    raise NotImplementedError("safety/train.py is a stub")


if __name__ == "__main__":
    sys.exit(main())
