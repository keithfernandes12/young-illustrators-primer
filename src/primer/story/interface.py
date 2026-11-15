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
    #: The recogniser's description of the whole drawing. The writer needs this to write about what
    #: the child actually drew rather than about the prompt alone.
    description: str | None = None
    #: The name the child gave their character ("Zip"). Optional and additive (KAN-92): serving does
    #: not fill it yet. It can be a child's own name, so it is never logged (COPPA).
    character_name: str | None = None
    #: Where the story happens ("the garden"). Optional; the prompt treats it as a hint.
    setting: str | None = None
    #: The age mode: "learn_to_read" (3-5) or "grow_vocabulary" (5-8). None means derive from `age`.
    #: Only learn_to_read is built so far.
    age_mode: str | None = None
    #: Target Flesch-Kincaid grade, on the scale `primer.story.levels.fk_grade` returns.
    reading_level: float | None = None


@dataclass(frozen=True)
class Story:
    """A story as pages of text, plus the title the writer gave it."""

    pages: tuple[str, ...] = field(default_factory=tuple)
    title: str = ""

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
