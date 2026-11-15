"""Sensitivity analysis for the speech stage: how far the metrics move when the input changes.

ASR is generative, so this is ablation rather than SHAP. The input side is perturbed here (added
noise, speed); the decoding side (beam size, prompting with the page) is swept as candidates in
``train.py``. Both end up in MLflow, and the conclusion goes in ``CARD.md``.

Noise is white Gaussian at a target signal-to-noise ratio: a classroom or a TV in the next room is at
roughly 10-20 dB. Speed perturbation resamples, so pitch moves with tempo -- the standard Kaldi-style
augmentation -- and stands in for slow, halting reading and for rushed reading.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from primer.speech.evaluate import ReadAloudClip, evaluate
from primer.speech.interface import SpeechRecognizer


def add_noise(samples: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """`samples` plus white noise at `snr_db` dB below the signal's power. Silence stays silent."""
    audio = np.asarray(samples, dtype=np.float32)
    power = float(np.mean(audio**2)) if audio.size else 0.0
    if power == 0.0:
        return audio.copy()
    noise = np.random.default_rng(seed).normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), audio.shape)
    return (audio + noise).astype(np.float32)


def change_speed(samples: np.ndarray, factor: float) -> np.ndarray:
    """Play `samples` `factor` times faster (0.9 is 10% slower). Duration scales by 1 / `factor`."""
    from scipy.signal import resample

    audio = np.asarray(samples, dtype=np.float32)
    if factor == 1.0 or audio.size == 0:
        return audio.copy()
    return resample(audio, max(1, round(audio.size / factor))).astype(np.float32)


@dataclass(frozen=True)
class Perturbation:
    name: str
    apply: Callable[[np.ndarray, int], np.ndarray]  # (samples, seed) -> samples


PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("clean", lambda x, seed: np.asarray(x, dtype=np.float32)),
    Perturbation("noise_20db", lambda x, seed: add_noise(x, 20, seed)),
    Perturbation("noise_10db", lambda x, seed: add_noise(x, 10, seed)),
    Perturbation("noise_5db", lambda x, seed: add_noise(x, 5, seed)),
    Perturbation("speed_0.9", lambda x, seed: change_speed(x, 0.9)),
    Perturbation("speed_1.1", lambda x, seed: change_speed(x, 1.1)),
)


def perturbed(
    clips: Iterable[ReadAloudClip], perturbation: Perturbation, seed: int = 0
) -> list[ReadAloudClip]:
    """Copies of `clips` with the audio perturbed. Each clip gets its own noise seed."""
    return [
        replace(clip, samples=perturbation.apply(np.asarray(clip.samples), seed + i))
        for i, clip in enumerate(clips)
    ]


def run(
    recognizer: SpeechRecognizer,
    clips: Sequence[ReadAloudClip],
    perturbations: Sequence[Perturbation] = PERTURBATIONS,
    seed: int = 0,
    prompt_with_page: bool = False,
) -> list[dict[str, Any]]:
    """One row per perturbation: its metrics and the change in WER from the clean audio."""
    rows: list[dict[str, Any]] = []
    clean_wer: float | None = None
    for perturbation in perturbations:
        overall, _ = evaluate(
            recognizer, perturbed(clips, perturbation, seed), prompt_with_page=prompt_with_page
        )
        if perturbation.name == "clean":
            clean_wer = overall["asr_wer"]
        delta = None if clean_wer is None else round(overall["asr_wer"] - clean_wer, 4)
        rows.append({"perturbation": perturbation.name, "wer_change": delta, **overall})
    return rows


def table(rows: Sequence[dict[str, Any]]) -> str:
    """The rows as a Markdown table, for the run artifact and ``CARD.md``."""
    keys = ("asr_wer", "wer_change", "asr_misread_precision", "asr_misread_recall")
    lines = ["| perturbation | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for row in rows:
        cells = ["n/a" if row.get(k) is None else f"{row[k]:.3f}" for k in keys]
        lines.append(f"| {row['perturbation']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
