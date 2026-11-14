"""Register a speech candidate in the MLflow Model Registry. CLI: --run-id [--alias --name --max-wer].

What gets registered is the frozen *configuration* -- ``recognizer.json``: checkpoint, revision and
decoding settings -- not a copy of the weights, which stay on Hugging Face at the pinned revision. A
registry version therefore means "this exact recogniser", and rolling back means pointing the alias
at an earlier version.

Only a run that ``train.py`` selected and that meets the WER target may be registered. Moving the
``production`` or ``rollback`` alias changes what children hear and is a person's decision: it needs
``--confirm``. The container push to Artifact Registry is the ``deploy-cloud-run`` skill's job.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping

from primer.speech.evaluate import STAGE

MODEL_NAME = "primer-speech"
MAX_WER = 0.25


def refusal(
    tags: Mapping[str, str], metrics: Mapping[str, float], revision: str, max_wer: float = MAX_WER
) -> str | None:
    """Why this run may not be registered, or None if it may."""
    if tags.get("stage") != STAGE:
        return f"run is stage {tags.get('stage')!r}, not {STAGE!r}"
    if tags.get("selected") != "true":
        return "run was not selected by train.py"
    if "asr_wer" not in metrics:
        return "run has no asr_wer"
    if metrics["asr_wer"] > max_wer:
        return f"asr_wer {metrics['asr_wer']} is above the {max_wer} target"
    if revision == "main":
        return "checkpoint revision is 'main', which moves; pin a commit sha in the config and re-run"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alias", default="candidate", choices=["candidate", "production", "rollback"])
    parser.add_argument("--name", default=MODEL_NAME)
    parser.add_argument("--max-wer", type=float, default=MAX_WER)
    parser.add_argument("--confirm", action="store_true", help="required to move production or rollback")
    args = parser.parse_args(argv)
    if args.alias != "candidate" and not args.confirm:
        parser.error(f"moving {args.alias!r} is a person's decision: re-run with --confirm")

    import json

    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    run = client.get_run(args.run_id)
    config_path = client.download_artifacts(args.run_id, "recognizer.json")
    with open(config_path, encoding="utf-8") as handle:
        revision = json.load(handle).get("revision", "main")
    reason = refusal(run.data.tags, run.data.metrics, revision, args.max_wer)
    if reason:
        print(f"not registered: {reason}", file=sys.stderr)
        return 1
    version = mlflow.register_model(f"runs:/{args.run_id}/recognizer.json", args.name)
    client.set_registered_model_alias(args.name, args.alias, version.version)
    print(f"registered {args.name} version {version.version} as {args.alias!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
