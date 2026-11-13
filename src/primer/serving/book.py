"""Compose the six model areas into a book.

Serving imports every area; no area imports serving (`AGENTS.md`, and
`docs/engineering/model-development.md` section 2). Everything here takes the Protocols as
arguments, so the whole path can be exercised with the Fakes -- no weights, no GCS, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

from primer.drawing.interface import DrawingRecognizer, Prediction
from primer.illustration.interface import Illustration, Illustrator
from primer.readability.interface import ReadabilityScorer
from primer.safety.interface import SafetyClassifier, SafetyVerdict
from primer.speech.interface import Audio, SpeechSynthesizer
from primer.story.interface import Story, StoryRequest, StoryWriter


@dataclass(frozen=True)
class Book:
    """One finished book, plus the numbers the gate cares about."""

    subject: Prediction
    story: Story
    illustrations: tuple[Illustration, ...]
    narration: tuple[Audio, ...]
    reading_level: float
    consistency: float
    safety: SafetyVerdict

    @property
    def pages(self) -> int:
        return len(self.story.pages)

    @property
    def complete(self) -> bool:
        """Every page has a picture and a voice, and nothing was flagged."""
        return (
            self.pages > 0
            and len(self.illustrations) == self.pages
            and len(self.narration) == self.pages
            and not self.safety.flagged
        )


def build_book(
    *,
    request: StoryRequest,
    drawing: bytes,
    recognizer: DrawingRecognizer,
    writer: StoryWriter,
    scorer: ReadabilityScorer,
    classifier: SafetyClassifier,
    illustrator: Illustrator,
    synthesizer: SpeechSynthesizer,
    style: str = "crayon",
) -> Book:
    """Run the whole path: recognise, write, check, illustrate, narrate, score.

    Safety is checked on the generated story, not only on the request: the request being harmless
    does not make the output harmless. A flagged book is still returned, with `complete` False, so
    the caller decides what to do rather than getting an exception.
    """
    predictions = recognizer.top_k(drawing)
    if not predictions:
        raise ValueError("the recognizer returned no predictions")
    subject = predictions[0]

    story = writer.write(request)
    verdict = classifier.check(story.text)

    illustrations = tuple(
        illustrator.illustrate(index, page, style) for index, page in enumerate(story.pages)
    )
    narration = tuple(synthesizer.speak(index, page) for index, page in enumerate(story.pages))

    return Book(
        subject=subject,
        story=story,
        illustrations=illustrations,
        narration=narration,
        reading_level=scorer.grade(story.text),
        consistency=illustrator.consistency(list(illustrations)),
        safety=verdict,
    )
