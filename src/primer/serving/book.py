"""Compose the six model areas into a book.

Serving imports every area; no area imports serving (`AGENTS.md`, and
`docs/engineering/model-development.md` section 2). Everything here takes the Protocols as
arguments, so the whole path can be exercised with the Fakes -- no weights, no GCS, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from primer.drawing.interface import Character, DrawingRecognizer
from primer.illustration.interface import Illustration, Illustrator
from primer.readability.interface import ReadabilityScorer
from primer.safety.interface import SafetyClassifier, SafetyVerdict
from primer.speech.interface import Audio, SpeechSynthesizer
from primer.story.interface import Story, StoryRequest, StoryWriter


@dataclass(frozen=True)
class Book:
    """One finished book, plus the numbers the gate cares about."""

    subject: Character
    story: Story
    illustrations: tuple[Illustration, ...]
    narration: tuple[Audio, ...]
    reading_level: float
    consistency: float
    safety: SafetyVerdict
    #: What the recogniser said the whole drawing shows.
    description: str = ""

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
    drawing_ref: str,
    style: str = "crayon",
) -> Book:
    """Run the whole path: recognise, check, write, check, illustrate, narrate, score.

    The recogniser runs first and its two outputs go to different places: the description to
    the writer, so the story is about what the child drew; the character embeddings to the
    illustrator, so page four looks like page one. The sketch itself goes to the illustrator
    too, for structure (KAN-42: ControlNet on the drawing, IP-Adapter on the character).

    Safety is checked **twice**, because the two checks catch different things. The request is
    classified before anything is generated -- the contract is to refuse unsafe requests, and an
    unsafe prompt that a writer happens to render harmlessly is still a request we must not serve.
    The story is classified after, because a harmless request does not guarantee harmless output.

    A refusal returns a Book rather than raising, so the caller decides what to do, but nothing is
    generated past the refusal: no story for a refused request, and no illustrations or narration
    for a refused story. Generating pictures and audio for text we have just judged unsafe costs
    money and creates artefacts of the thing we refused.
    """
    if not drawing_ref:
        # The illustrator needs a durable handle on the drawing, not only the bytes in this
        # process: a retry, a queue hop or an attribution all outlive the request.
        raise ValueError("drawing_ref is required whenever a drawing is given")

    analysis = recognizer.analyse(drawing)
    subject = analysis.main
    if subject is None:
        raise ValueError("the recognizer returned no characters")

    request_verdict = classifier.check(request.prompt)
    if request_verdict.flagged:
        return Book(
            subject=subject,
            story=Story(),
            illustrations=(),
            narration=(),
            reading_level=0.0,
            consistency=0.0,
            safety=request_verdict,
            description=analysis.description,
        )

    # The writer and the illustrator both need to know what the child actually drew.
    grounded = replace(
        request,
        subject=subject.label,
        drawing_ref=drawing_ref,
        description=analysis.description,
    )
    story = writer.write(grounded)
    story_verdict = classifier.check(story.text)
    if story_verdict.flagged:
        return Book(
            subject=subject,
            story=story,
            illustrations=(),
            narration=(),
            reading_level=scorer.grade(story.text),
            consistency=0.0,
            safety=story_verdict,
            description=analysis.description,
        )

    illustrations = tuple(
        illustrator.illustrate(
            index,
            page,
            style,
            subject.label,
            drawing_ref,
            sketch=drawing,
            characters=analysis.characters,
        )
        for index, page in enumerate(story.pages)
    )
    narration = tuple(synthesizer.speak(index, page) for index, page in enumerate(story.pages))

    return Book(
        subject=subject,
        story=story,
        illustrations=illustrations,
        narration=narration,
        reading_level=scorer.grade(story.text),
        consistency=illustrator.consistency(list(illustrations), analysis.characters),
        safety=story_verdict,
        description=analysis.description,
    )
