"""Every model area offers a Protocol and a Fake, and serving composes all six into a book.

This is the skeleton five other members import on day one: without it, nobody can write against an
area somebody else has not built. Nothing here loads weights, touches GCS or reaches the network --
the Fakes are pure computation, which is also what lets the golden runner work before any model does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from primer.drawing.interface import DrawingRecognizer, FakeDrawingRecognizer, Prediction
from primer.illustration.interface import FakeIllustrator, Illustration, Illustrator
from primer.readability.interface import FakeReadabilityScorer, ReadabilityScorer
from primer.safety.interface import FakeSafetyClassifier, SafetyClassifier
from primer.serving.book import build_book
from primer.speech.interface import FakeSpeechSynthesizer, SpeechSynthesizer
from primer.story.interface import FakeStoryWriter, Story, StoryRequest, StoryWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "primer"

AREAS = ("drawing", "story", "readability", "safety", "illustration", "speech")

#: (Protocol, Fake) for each area. The six the blueprint asks for, readability included.
CONTRACTS = [
    (DrawingRecognizer, FakeDrawingRecognizer()),
    (StoryWriter, FakeStoryWriter()),
    (ReadabilityScorer, FakeReadabilityScorer()),
    (SafetyClassifier, FakeSafetyClassifier()),
    (Illustrator, FakeIllustrator()),
    (SpeechSynthesizer, FakeSpeechSynthesizer()),
]


@pytest.mark.parametrize("protocol,fake", CONTRACTS, ids=[p.__name__ for p, _ in CONTRACTS])
def test_each_fake_satisfies_its_protocol(protocol, fake):
    assert isinstance(fake, protocol), f"{type(fake).__name__} does not satisfy {protocol.__name__}"


@pytest.mark.parametrize("area", AREAS)
def test_each_area_has_the_section_2_layout(area):
    """The file layout other members code against, from model-development.md section 2."""
    base = SRC / area
    for name in (
        "interface.py",
        "data.py",
        "model.py",
        "train.py",
        "evaluate.py",
        "slices.py",
        "sensitivity.py",
        "register.py",
        "Dockerfile",
        "CARD.md",
    ):
        assert (base / name).exists(), f"src/primer/{area}/{name} is missing"
    assert list((base / "configs").glob("*.yaml")), f"src/primer/{area}/configs has no yaml"


@pytest.mark.parametrize("area", AREAS)
def test_no_model_area_imports_serving(area):
    """serving may import any area; an area never imports serving. A cycle here is architectural."""
    offenders = []
    for path in (SRC / area).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.startswith("primer.serving"):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")
    assert not offenders, "\n".join(offenders)


def test_each_areas_evaluate_claims_only_its_own_metrics():
    """evaluate.py takes its gated keys from the shared contract, so no two areas claim one key."""
    from primer.evaluation import contract

    seen: dict[str, str] = {}
    for key, owner in contract.METRIC_OWNERS.items():
        assert key not in seen, f"{key} is owned twice"
        seen[key] = owner
    assert set(seen.values()) >= set(AREAS) - {"speech"}, "every gated area must own a metric"


# ----------------------------------------------------------------- the whole path, on fakes


@pytest.fixture
def request_for_a_cat() -> StoryRequest:
    return StoryRequest(prompt="a cat in the garden", age=5, target_words=("cat", "nap"))


def test_serving_builds_a_complete_fake_book(request_for_a_cat):
    book = build_book(
        request=request_for_a_cat,
        drawing=b"not-a-real-png",
        recognizer=FakeDrawingRecognizer(label="cat"),
        writer=FakeStoryWriter(),
        scorer=FakeReadabilityScorer(grade_value=1.2),
        classifier=FakeSafetyClassifier(),
        illustrator=FakeIllustrator(consistency_score=0.9),
        synthesizer=FakeSpeechSynthesizer(),
    )

    assert book.complete
    assert book.subject == Prediction(label="cat", confidence=1.0)
    assert book.pages == len(book.story.pages) > 0
    assert len(book.illustrations) == book.pages
    assert len(book.narration) == book.pages
    assert book.reading_level == 1.2
    assert book.consistency == 0.9
    assert not book.safety.flagged
    # Every target word survives into the story, which is what target_word_inclusion measures.
    for word in request_for_a_cat.target_words:
        assert word in book.story.text


def test_a_flagged_story_is_returned_not_raised(request_for_a_cat):
    """The caller decides what to do with an unsafe book; serving does not throw it away."""
    book = build_book(
        request=request_for_a_cat,
        drawing=b"x",
        recognizer=FakeDrawingRecognizer(),
        writer=FakeStoryWriter(),
        scorer=FakeReadabilityScorer(),
        classifier=FakeSafetyClassifier(unsafe_markers=("cat",)),
        illustrator=FakeIllustrator(),
        synthesizer=FakeSpeechSynthesizer(),
    )
    assert book.safety.flagged
    assert not book.complete
    assert book.pages > 0


def test_safety_is_checked_on_the_story_not_the_request():
    """A harmless request does not make the output harmless."""
    book = build_book(
        request=StoryRequest(prompt="a perfectly nice garden", age=5, target_words=("marker",)),
        drawing=b"x",
        recognizer=FakeDrawingRecognizer(),
        writer=FakeStoryWriter(),
        scorer=FakeReadabilityScorer(),
        classifier=FakeSafetyClassifier(unsafe_markers=("marker",)),
        illustrator=FakeIllustrator(),
        synthesizer=FakeSpeechSynthesizer(),
    )
    assert book.safety.flagged, "the marker is only in the story, not in the prompt"


def test_no_predictions_is_an_error(request_for_a_cat):
    class Empty:
        def top_k(self, image: bytes, k: int = 5) -> list[Prediction]:
            return []

    with pytest.raises(ValueError, match="no predictions"):
        build_book(
            request=request_for_a_cat,
            drawing=b"x",
            recognizer=Empty(),
            writer=FakeStoryWriter(),
            scorer=FakeReadabilityScorer(),
            classifier=FakeSafetyClassifier(),
            illustrator=FakeIllustrator(),
            synthesizer=FakeSpeechSynthesizer(),
        )


def test_fakes_hold_no_payloads():
    """Illustrations and audio carry references, never bytes: a book must not pin child data."""
    illustration = FakeIllustrator().illustrate(0, "a page", "crayon")
    audio = FakeSpeechSynthesizer().speak(0, "a page")
    assert isinstance(illustration, Illustration)
    assert illustration.image_ref.startswith("fake://")
    assert audio.audio_ref.startswith("fake://")
    assert not any(isinstance(getattr(audio, f), bytes) for f in ("audio_ref", "duration_s"))


def test_story_text_joins_pages():
    story = Story(pages=("One.", "Two."))
    assert story.text == "One. Two."
