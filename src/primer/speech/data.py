"""Evaluation clips for the speech stage.

**Speechocean762 is a fallback, not the target.** MyST (children reading aloud) is access-by-request
and has no pipeline yet (KAN-101). Speechocean762 (``mispeech/speechocean762`` on Hugging Face,
CC BY 4.0) is 5,000 read English sentences from 250 speakers, about half of them children, with age
and gender per speaker. Its limits, which ``CARD.md`` repeats: every speaker's first language is
Mandarin, so it cannot measure accent; and the sentences are short prompts, not story pages.

It is read straight from Hugging Face rather than through ``primer-data`` because there is no
ingestion for it yet. That is a known gap against the repo's data rules, recorded in the README.

**Labelled misreads without children's audio.** A clip where the speaker read the sentence correctly
becomes a labelled misread by changing the *page*: if the audio says "the cat sat" and the page now
says "the hat sat", the reader misread "hat" at position 1. Inserting a word the audio never says
makes a labelled skip. Every label is known exactly, and no child is recorded to get it.

Splitting is by speaker, never by clip. Speechocean762's own train / test split is already by speaker.
Audio is decoded in memory and never written anywhere.
"""

from __future__ import annotations

import io
import random
from collections.abc import Iterable, Sequence
from dataclasses import replace
from difflib import SequenceMatcher
from math import gcd
from typing import Any

import numpy as np

from primer.speech.alignment import NEAR_MISS, normalize
from primer.speech.evaluate import ReadAloudClip
from primer.speech.interface import SAMPLE_RATE
from primer.speech.slices import age_band

SPEECHOCEAN = "mispeech/speechocean762"
SPEECHOCEAN_LICENSE = "CC BY 4.0"
SPEECHOCEAN_ATTRIBUTION = "Zhang et al., speechocean762 (Interspeech 2021), Speechocean and Xiaomi"


def resample(audio: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Polyphase resampling to `target` Hz."""
    if rate == target:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    factor = gcd(rate, target)
    return resample_poly(audio, target // factor, rate // factor).astype(np.float32)


def decode_audio(data: bytes) -> np.ndarray:
    """Mono float32 samples at 16 kHz from an encoded file (WAV or FLAC), in memory."""
    import soundfile

    audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), rate)


def _speechocean_clip(row: dict[str, Any], split: str, index: int) -> ReadAloudClip:
    audio = row["audio"]
    data = audio.get("bytes")
    if data is None:
        with open(audio["path"], "rb") as handle:
            data = handle.read()
    text = str(row["text"])
    slices = {"age_band": age_band(int(row["age"]))}
    if row.get("gender"):
        slices["gender"] = str(row["gender"]).lower()
    return ReadAloudClip(
        clip_id=f"so762-{split}-{index:05d}",
        speaker_id=f"so762-{row['speaker']}",
        page_text=text,
        reference=text,
        misread_positions=frozenset(),
        samples=decode_audio(data),
        slices=slices,
    )


def load_speechocean(
    split: str = "test", revision: str = "main", max_clips: int | None = None
) -> list[ReadAloudClip]:
    """Speechocean762 clips the speaker read completely, so the sentence is a fair reference.

    Downloads from Hugging Face on first use (a person's machine, not CI). `max_clips` takes the first
    clips in the split's order, which is deterministic for a pinned `revision`.
    """
    from datasets import Audio, load_dataset

    rows = load_dataset(SPEECHOCEAN, split=split, revision=revision)
    rows = rows.cast_column("audio", Audio(decode=False))  # decode ourselves: no torchcodec or ffmpeg
    missing = {"audio", "text", "speaker", "age", "completeness"} - set(rows.column_names)
    if missing:
        raise ValueError(f"{SPEECHOCEAN}@{revision} has no {sorted(missing)}: the schema changed")
    clips: list[ReadAloudClip] = []
    for index, row in enumerate(rows):
        if float(row["completeness"]) < 10:  # a word was left out, so the text is not what was said
            continue
        clips.append(_speechocean_clip(row, split, index))
        if max_clips is not None and len(clips) >= max_clips:
            break
    return clips


def _vocabulary(clips: Iterable[ReadAloudClip]) -> list[str]:
    return sorted({w for clip in clips for w in normalize(clip.page_text)})


def _distinct_word(avoid: Sequence[str], vocabulary: Sequence[str], rng: random.Random) -> str:
    """A word unlike every word in `avoid`, so ``align``'s leniency does not forgive it.

    `avoid` holds the word being replaced and its neighbours: a word equal to a neighbour would let
    the alignment pair it with that neighbour and flag a different position than the label says.
    """
    for _ in range(100):
        word = rng.choice(vocabulary)
        if all(SequenceMatcher(None, other, word).ratio() < NEAR_MISS for other in avoid):
            return word
    raise ValueError(f"no word in the vocabulary is distinct from {list(avoid)}")


def with_synthetic_misreads(
    clips: Sequence[ReadAloudClip], rate: float = 0.7, seed: int = 0, skip_share: float = 0.25
) -> list[ReadAloudClip]:
    """Give a `rate` share of clips one labelled misread (or, `skip_share` of those, one skip).

    The audio and `reference` are untouched: only the page changes. Deterministic for a given `seed`
    and clip ids. The rest of the clips stay clean, which is what makes precision measurable.
    """
    vocabulary = _vocabulary(clips)
    if len(vocabulary) < 2:
        raise ValueError("need at least two distinct words to make a misread")
    out: list[ReadAloudClip] = []
    for clip in clips:
        rng = random.Random(f"{seed}:{clip.clip_id}")
        words = normalize(clip.page_text)
        if not words or rng.random() >= rate:
            out.append(clip)
            continue
        position = rng.randrange(len(words))
        if rng.random() < skip_share:
            words.insert(
                position, _distinct_word(words[max(position - 1, 0) : position + 1], vocabulary, rng)
            )
        else:
            words[position] = _distinct_word(words[max(position - 1, 0) : position + 2], vocabulary, rng)
        out.append(replace(clip, page_text=" ".join(words), misread_positions=frozenset({position})))
    return out


def load_clips(data: dict[str, Any]) -> list[ReadAloudClip]:
    """The evaluation clips the ``data`` block of a candidates config describes."""
    if data.get("dataset") != "speechocean762":
        raise ValueError(f"no loader for dataset {data.get('dataset')!r}; MyST is KAN-101")
    clips = load_speechocean(data.get("split", "test"), data.get("revision", "main"), data.get("max_clips"))
    return with_synthetic_misreads(clips, data.get("misread_rate", 0.7), data.get("seed", 0))


def data_version(data: dict[str, Any]) -> str:
    """The data version string recorded with every run, e.g. ``hf:mispeech/speechocean762@main/test``."""
    return f"hf:{data.get('hf_id', SPEECHOCEAN)}@{data.get('revision', 'main')}/{data.get('split', 'test')}"
