"""The story contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class StoryRequest:
    """What the parent and child asked for, plus what the drawing turned out to be.

    ``subject`` is the label the drawing recogniser returned. Without it the writer has no idea
    what the child drew, and the book would be about the prompt alone -- which is not the product.
    Serving fills it in between recognition and writing; it is None only when no drawing was given.
    """

    prompt: str
    age: int
    target_words: tuple[str, ...] = ()
    subject: str | None = None
    drawing_ref: str | None = None


@dataclass(frozen=True)
class Story:
    """A story as pages of text."""

    pages: tuple[str, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return " ".join(self.pages)


@runtime_checkable
class StoryWriter(Protocol):
    """Write an age-appropriate story. Gated on `target_word_inclusion` and `story_quality_mean`."""

    def write(self, request: StoryRequest) -> Story:
        """A story that contains every target word."""
        ...


class FakeStoryWriter:
    """Writes one page per target word, so target_word_inclusion is 1.0 by construction."""

    def __init__(self, pages: int = 3) -> None:
        self.pages = pages

    def write(self, request: StoryRequest) -> Story:
        body = [f"The {word} was there." for word in request.target_words]
        while len(body) < self.pages:
            body.append(f"A quiet page for a child of {request.age}.")
        return Story(pages=tuple(body))
