"""Register a safety candidate: MLflow Model Registry plus the Artifact Registry push.

Only a run that passed the gate may be registered, and only one that is reproducible from this
area's container on Vertex AI against the frozen data version
(docs/adrs/where-model-training-happens.md). Images go to the SHARED project's registry; the job
that produced them ran in the member's own compute project.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alias", default="candidate", choices=["candidate", "production", "rollback"])
    parser.parse_args(argv)
    raise NotImplementedError("safety/register.py is a stub")


if __name__ == "__main__":
    sys.exit(main())
