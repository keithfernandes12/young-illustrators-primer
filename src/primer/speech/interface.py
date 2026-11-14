"""The speech contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
