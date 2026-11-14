"""Analyse one drawing locally with the real weights, to prove the baseline runs end to end.

    pip install -e ".[drawing]"
    HF_HUB_OFFLINE=1 python -m primer.drawing.smoke --drawing /absolute/path/to/synthetic.png

Synthetic drawings only: this must not be pointed at a real child's drawing. The analysis holds an
embedding derived from whatever it is given, and this prints to a terminal and a transcript.

The drawing is read from a path rather than generated here **on purpose**. Turning strokes into
pixels is KAN-87's `render_strokes` (plan slice S6b), and it exists precisely so that there is one
renderer rather than one per call site; a quick `ImageDraw` loop in this file would be the second
one. Until S6b lands, pass any PNG -- the point of this run is that weights load, the vocabulary is
scored and a ranked character sheet comes back, not what the picture is of.

Prints one JSON object: the winning label, the ranked top-5, the device, load and analyse seconds,
and peak VRAM when it ran on a GPU. Run it from Git Bash, not PowerShell, which does not reliably
see the GPU (AGENTS.md; docs/engineering/model-containers-on-windows.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from primer.drawing.model import DrawingConfig, DrawingRecognizerModel, SiglipEncoder


def peak_vram_gb(device: str | None) -> float | None:
    """Peak CUDA memory this process reserved, or None when the run did not use a GPU.

    Reported because the choice of checkpoint is constrained by an 8 GB laptop, and AGENTS.md asks
    for the figure a candidate actually needed rather than an estimate of it.

    Keyed on the device the encoder actually used, not on whether the machine has a GPU: with
    `--device cpu` on a CUDA box the latter reports `0.0` for a run that never touched the card,
    which is a measurement of nothing wearing the name of the one number that must be measured.
    """
    if device != "cuda":
        return None
    import torch

    return round(torch.cuda.max_memory_reserved() / 1024**3, 2)


def reset_vram_peak() -> None:
    """Zero the CUDA peak counter, so the figure reported is this run's rather than the process's."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--drawing", required=True, help="path to a synthetic drawing, PNG or JPEG")
    parser.add_argument("--device", default=None, help="cuda or cpu; autodetected when omitted")
    args = parser.parse_args(argv)

    config = DrawingConfig.from_yaml()
    encoder = SiglipEncoder(config, device=args.device)
    model = DrawingRecognizerModel(config, encoder=encoder)

    drawing = Path(args.drawing).read_bytes()
    reset_vram_peak()
    started = time.perf_counter()
    encoder.load()
    loaded = time.perf_counter()
    analysis = model.analyse(drawing)
    first = time.perf_counter()
    # Twice, and both reported. The first analyse encodes all 345 prompts, which is most of its
    # cost and happens once per process; the second is what a drawing actually costs, and is the
    # number KAN-71's latency budget is about. Reporting only the first -- as this did -- puts a
    # one-time cost under a name every reader takes as per-drawing.
    model.analyse(drawing)
    finished = time.perf_counter()

    character = analysis.main
    if character is None:
        # Not an `assert`: `python -O` drops those, and the next line would raise an
        # `AttributeError` on None instead of saying what went wrong.
        raise RuntimeError("the baseline always returns one character, and this analysis has none")
    print(
        json.dumps(
            {
                "label": character.label,
                "confidence": character.confidence,
                "top_labels": analysis.top_labels(5),
                "candidates": [[c.label, c.confidence] for c in character.candidates],
                "embedding_dimensions": len(character.embedding),
                "labels_scored": len(config.labels()),
                "encoder": config.encoder,
                "encoder_revision": config.encoder_revision,
                "device": encoder.device,
                "load_seconds": round(loaded - started, 1),
                "first_analyse_seconds": round(first - loaded, 1),
                "warm_analyse_seconds": round(finished - first, 2),
                "peak_vram_gb": peak_vram_gb(encoder.device),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
