"""The real speech implementations. Weights load lazily, never at import time.

Importing this module must stay cheap: the serving layer imports every area, and a test that
imports it must not pull a checkpoint off disk or out of GCS. ``torch`` and ``transformers`` are
imported inside ``_load`` and come from the ``speech`` extra (``pip install -e ".[speech]"``).

Two recogniser families, because they fail differently on a child reading aloud:

- **Whisper** is a sequence-to-sequence model with a strong language model inside it. It is the most
  accurate on fluent speech and the most likely to "hear" the word that should have been there.
- **wav2vec2 (CTC)** emits characters frame by frame with no language model. It is less accurate on
  fluent speech, but it writes down what it hears, which is what misread detection needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from primer.speech.interface import SAMPLE_RATE, Transcript, Word

Family = Literal["whisper", "wav2vec2"]


@dataclass(frozen=True)
class RecognizerConfig:
    """One frozen recogniser configuration: what ``train.py`` compares and ``register.py`` records.

    `revision` pins the Hugging Face checkpoint; "main" is not reproducible and is only for a first
    look. `beam_size` 1 is greedy decoding (temperature 0). `prompt_with_page` passes the page text
    as Whisper's prompt; wav2vec2 has no prompt and ignores it.

    The two guards are against Whisper's runaway loops, which do happen on young children: a 4 s clip
    of "Lilly likes biscuit" decoded to 443 repetitions of "ni" until the token limit. They are off by
    default so the published model is measured as published. `max_tokens_per_second` caps the output
    at that many tokens per second of audio (plus ``_TOKEN_MARGIN``); `no_repeat_ngram_size` forbids
    any run of that many tokens from appearing twice.
    """

    name: str
    family: Family
    checkpoint: str
    revision: str = "main"
    beam_size: int = 1
    prompt_with_page: bool = False
    max_tokens_per_second: float | None = None
    no_repeat_ngram_size: int = 0
    device: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecognizerConfig:
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown recogniser settings: {sorted(unknown)}")
        if data.get("family") not in ("whisper", "wav2vec2"):
            raise ValueError(f"{data.get('name')}: family must be whisper or wav2vec2")
        return cls(**data)


#: Tokens allowed on top of the per-second cap, so a very short clip can still be transcribed.
_TOKEN_MARGIN = 8
#: Whisper's decoder holds 448 positions; a few go to the start-of-transcript tokens.
_MAX_NEW_TOKENS = 440


def generation_kwargs(config: RecognizerConfig, duration_s: float) -> dict[str, Any]:
    """The ``generate`` settings for one clip of `duration_s` seconds, prompt aside."""
    kwargs: dict[str, Any] = {"num_beams": config.beam_size, "do_sample": False}
    if not config.checkpoint.endswith(".en"):
        kwargs |= {"language": "en", "task": "transcribe"}  # English-only checkpoints refuse these
    if config.max_tokens_per_second:
        cap = math.ceil(duration_s * config.max_tokens_per_second) + _TOKEN_MARGIN
        kwargs["max_new_tokens"] = min(cap, _MAX_NEW_TOKENS)
    if config.no_repeat_ngram_size:
        kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size
    return kwargs


def _device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mono_16k(samples: ArrayLike, sample_rate: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("expected mono samples")
    if sample_rate != SAMPLE_RATE:
        from primer.speech.data import resample

        audio = resample(audio, sample_rate)
    return audio


def _words(text: str) -> Transcript:
    """A transcript from plain text. These recognisers produce no word timings or confidences."""
    return Transcript(words=tuple(Word(w, 0.0, 0.0, math.nan) for w in text.split()))


class WhisperRecognizer:
    """Whisper through ``transformers``. Satisfies ``interface.SpeechRecognizer``."""

    def __init__(self, config: RecognizerConfig) -> None:
        self.config = config
        self._loaded: tuple[Any, Any, str] | None = None  # (processor, model, device), on first use

    def _load(self) -> tuple[Any, Any, str]:
        if self._loaded is None:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            device = _device(self.config.device)
            processor = WhisperProcessor.from_pretrained(
                self.config.checkpoint, revision=self.config.revision
            )
            # Typed Any: transformers 5's stubs type the bound `.to` as taking a model, not a device.
            model: Any = WhisperForConditionalGeneration.from_pretrained(
                self.config.checkpoint, revision=self.config.revision
            )
            model = model.to(device=device, dtype=torch.float32)  # some checkpoints ship in float16
            model.eval()
            self._loaded = (processor, model, device)
        return self._loaded

    def transcribe(
        self, samples: ArrayLike, sample_rate: int = SAMPLE_RATE, prompt: str | None = None
    ) -> Transcript:
        import torch

        processor, model, device = self._load()
        audio = _mono_16k(samples, sample_rate)
        features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
        kwargs = generation_kwargs(self.config, len(audio) / SAMPLE_RATE)
        use_prompt = self.config.prompt_with_page and prompt
        if use_prompt:
            kwargs["prompt_ids"] = torch.tensor(processor.get_prompt_ids(prompt)).to(device)
        with torch.no_grad():
            ids = model.generate(features, **kwargs)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if use_prompt and prompt and text.startswith(prompt.strip()):
            # Some transformers versions decode the prompt back into the output.
            text = text[len(prompt.strip()) :].strip()
        return _words(text)


class Wav2Vec2Recognizer:
    """A CTC model through ``transformers``, greedy decoding, no language model. Ignores `prompt`."""

    def __init__(self, config: RecognizerConfig) -> None:
        self.config = config
        self._loaded: tuple[Any, Any, str] | None = None

    def _load(self) -> tuple[Any, Any, str]:
        if self._loaded is None:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            device = _device(self.config.device)
            processor = Wav2Vec2Processor.from_pretrained(
                self.config.checkpoint, revision=self.config.revision
            )
            model: Any = Wav2Vec2ForCTC.from_pretrained(self.config.checkpoint, revision=self.config.revision)
            model = model.to(device=device, dtype=torch.float32)  # some checkpoints ship in float16
            model.eval()
            self._loaded = (processor, model, device)
        return self._loaded

    def transcribe(
        self, samples: ArrayLike, sample_rate: int = SAMPLE_RATE, prompt: str | None = None
    ) -> Transcript:
        import torch

        processor, model, device = self._load()
        audio = _mono_16k(samples, sample_rate)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(inputs).logits
        text = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
        return _words(text.lower())


def build(config: RecognizerConfig) -> WhisperRecognizer | Wav2Vec2Recognizer:
    """The recogniser for `config`. Nothing is downloaded until the first ``transcribe``."""
    return WhisperRecognizer(config) if config.family == "whisper" else Wav2Vec2Recognizer(config)


class SpeechSynthesizerModel:
    """The production speech implementation. Will satisfy ``interface.SpeechSynthesizer``."""

    def __init__(self, model_uri: str | None = None) -> None:
        self._model_uri = model_uri
        self._loaded: object | None = None  # populated on first use, not here

    def _load(self) -> object:
        raise NotImplementedError("speech/model.py: narration (TTS) belongs to its own owner")
