"""The speech contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Audio:
    """Narration for one page. `audio_ref` is a URI or cache key, never audio bytes.

    Audio is discarded by default (`docs/adrs/`, the privacy commitments in `privacy-guard`), so
    nothing here holds a waveform and nothing writes one without a parent opt-in.
    """

    page_index: int
    audio_ref: str
    duration_s: float


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Read the story aloud."""

    def speak(self, page_index: int, text: str) -> Audio:
        """Narration for one page."""
        ...


class FakeSpeechSynthesizer:
    """Deterministic refs and a duration derived from length. Synthesises nothing."""

    def speak(self, page_index: int, text: str) -> Audio:
        return Audio(
            page_index=page_index,
            audio_ref=f"fake://speech/page-{page_index}",
            duration_s=round(len(text.split()) / 2.5, 2),
        )


# ---------------------------------------------------------------- read-aloud ASR (stage 8)

#: Whisper's native rate; every recogniser takes mono samples at this rate.
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Word:
    """One heard word. Times are seconds from the start of the utterance."""

    text: str
    start_s: float
    end_s: float
    confidence: float


@dataclass(frozen=True)
class Transcript:
    """What the recogniser heard. Text and timings only: the audio is never kept."""

    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


#: How one expected word fared. `inserted` is a heard word the page does not have; it is reported
#: but never counted against the child.
WordStatus = Literal["correct", "misread", "skipped", "inserted"]


@dataclass(frozen=True)
class WordResult:
    """One position in the aligned page: the word on the page, the word heard, and the verdict."""

    expected: str | None
    heard: str | None
    status: WordStatus


@dataclass(frozen=True)
class ReadingResult:
    """A page's read-aloud outcome, the shape the learner profile consumes."""

    page_index: int
    words: tuple[WordResult, ...]

    @property
    def misread_words(self) -> tuple[str, ...]:
        """Page words the child misread or skipped, in page order."""
        return tuple(w.expected for w in self.words if w.status in ("misread", "skipped") and w.expected)


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Listen to a child read. Audio is processed in memory and discarded."""

    def transcribe(self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE) -> Transcript:
        """Transcribe one utterance of mono samples."""
        ...


class FakeSpeechRecognizer:
    """Hears `heard` whatever the audio is, with even word timings. Loads nothing."""

    def __init__(self, heard: str = "", word_s: float = 0.4, confidence: float = 1.0) -> None:
        self.heard = heard
        self.word_s = word_s
        self.confidence = confidence

    def transcribe(self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE) -> Transcript:
        words = tuple(
            Word(text, round(i * self.word_s, 2), round((i + 1) * self.word_s, 2), self.confidence)
            for i, text in enumerate(self.heard.split())
        )
        return Transcript(words=words)
