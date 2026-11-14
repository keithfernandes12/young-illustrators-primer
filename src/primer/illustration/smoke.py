"""Generate one page locally with the real weights, to prove the pipeline runs end to end.

    pip install -e ".[illustration]"
    HF_HUB_OFFLINE=1 python -m primer.illustration.smoke \\
        --drawing-ref file:///absolute/path/to/sketch.png --subject house \\
        --page-text "a cozy little house on a fluffy cloud" --out .local-data/illustration-smoke

Synthetic sketches only. Generated images are not safety-checked until KAN-94 adds that check in
serving, so this must not be pointed at a real child's drawing. The drawing is read through
``drawing_ref`` rather than passed as bytes, so the run exercises the reference contract too.
Prints one JSON object: the image reference, the device, and load and generation seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

from primer.illustration.model import DiffusersPipeline, IllustrationConfig, IllustratorModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--drawing-ref", required=True, help="file:// or gs:// URI of a synthetic sketch")
    parser.add_argument("--page-text", default="a cozy little house on a fluffy cloud, a smiling sun")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--style", default="soft watercolor")
    parser.add_argument("--out", default=".local-data/illustration-smoke")
    args = parser.parse_args(argv)

    config = IllustrationConfig.from_yaml()
    pipeline = DiffusersPipeline(config)
    model = IllustratorModel(args.out, config, pipeline=pipeline)

    started = time.perf_counter()
    pipeline.load()
    loaded = time.perf_counter()
    page = model.illustrate(0, args.page_text, args.style, args.subject, drawing_ref=args.drawing_ref)
    finished = time.perf_counter()

    print(
        json.dumps(
            {
                "illustration": asdict(page),
                "device": pipeline.device,
                "load_seconds": round(loaded - started, 1),
                "generate_seconds": round(finished - loaded, 1),
                "steps": config.steps,
                "seed": config.seed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
