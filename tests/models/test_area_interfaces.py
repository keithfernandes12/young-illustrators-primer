"""Every model area offers a Protocol and a Fake, and serving composes all six into a book.

This is the skeleton five other members import on day one: without it, nobody can write against an
area somebody else has not built. Nothing here loads weights, touches GCS or reaches the network --
the Fakes are pure computation, which is also what lets the golden runner work before any model does.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

from primer.drawing.interface import (
    Character,
    DrawingAnalysis,
    DrawingRecognizer,
    FakeDrawingRecognizer,
)
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


# ------------------------------------------------- findings from the Copilot review of PR #11


@pytest.mark.parametrize("area", AREAS)
def test_the_image_installs_the_package(area):
    """`pip install -e .` ran before src/ was copied, and `|| true` hid the failure.

    pyproject.toml names README.md as the readme, so the install needed that too. The result was an
    image that built cleanly with no `primer` package inside it.
    """
    text = (SRC / area / "Dockerfile").read_text(encoding="utf-8")
    # Comment lines explain why `|| true` is absent, so only the instructions are searched.
    code = [line for line in text.splitlines() if not line.strip().startswith("#")]
    install = next(i for i, line in enumerate(code) if "pip install" in line)
    copied = " ".join(code[:install])
    assert "pyproject.toml" in copied, "pyproject must be copied before the install"
    assert "README.md" in copied, "pyproject names README.md; the install reads it"
    assert "COPY src" in copied, "the package must be present before an editable install"
    assert "|| true" not in "\n".join(code), "a failed install must stop the build"


def test_the_vertex_template_passes_flags_train_actually_accepts():
    """The template used --output, omitted --data-version, and named a config that did not exist.

    A copied job died in argument parsing. This runs the template's own arguments through the real
    parser: reaching NotImplementedError means the flags were accepted; SystemExit means they were
    not.
    """
    import yaml

    from primer.story import train

    template = REPO_ROOT / ".claude" / "skills" / "train-vertex" / "job_template.yaml"
    spec = yaml.safe_load(template.read_text(encoding="utf-8"))
    args = _find_args(spec)
    assert args, "no args found in the job template"

    config = next(value for flag, value in itertools.pairwise(args) if flag == "--config")
    assert (REPO_ROOT / config).exists(), f"the template names a config that does not exist: {config}"

    with pytest.raises(NotImplementedError):
        train.main(list(args))


def _find_args(node: object) -> list[str] | None:
    """The first `args` list anywhere in the job spec."""
    if isinstance(node, dict):
        if isinstance(node.get("args"), list):
            return [str(item) for item in node["args"]]
        for value in node.values():
            found = _find_args(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_args(item)
            if found:
                return found
    return None


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
        drawing_ref="gs://yip-drawings/test/cat.png",
        recognizer=FakeDrawingRecognizer(label="cat"),
        writer=FakeStoryWriter(),
        scorer=FakeReadabilityScorer(grade_value=1.2),
        classifier=FakeSafetyClassifier(),
        illustrator=FakeIllustrator(consistency_score=0.9),
        synthesizer=FakeSpeechSynthesizer(),
    )

    assert book.complete
    assert book.subject.label == "cat"
    assert book.subject.has_embedding, "the illustrator needs an identity to condition on"
    assert "cat" in book.description
    assert book.pages == len(book.story.pages) > 0
    assert len(book.illustrations) == book.pages
    assert len(book.narration) == book.pages
    assert book.reading_level == 1.2
    assert book.consistency == 0.9
    assert not book.safety.flagged
    # Every target word survives into the story, which is what target_word_inclusion measures.
    for word in request_for_a_cat.target_words:
        assert word in book.story.text


def build(request: StoryRequest, **overrides):
    """build_book on fakes, with sensible defaults so each test states only what it varies."""
    parts = {
        "drawing": b"x",
        "drawing_ref": "gs://yip-drawings/test/cat.png",
        "recognizer": FakeDrawingRecognizer(label="cat"),
        "writer": FakeStoryWriter(),
        "scorer": FakeReadabilityScorer(),
        "classifier": FakeSafetyClassifier(),
        "illustrator": FakeIllustrator(),
        "synthesizer": FakeSpeechSynthesizer(),
    }
    parts.update(overrides)
    return build_book(request=request, **parts)


def test_an_unsafe_request_is_refused_before_anything_is_generated():
    """The contract is to refuse unsafe requests, not only unsafe output.

    An unsafe prompt that the writer happens to render harmlessly is still a request we must not
    serve, and before this the request was never classified at all.
    """
    book = build(
        StoryRequest(prompt="a request with a bad-marker in it", age=5, target_words=("cat",)),
        classifier=FakeSafetyClassifier(unsafe_markers=("bad-marker",)),
    )
    assert book.safety.flagged
    assert not book.complete
    assert book.pages == 0, "no story may be written for a refused request"
    assert book.illustrations == () and book.narration == ()


def test_a_flagged_story_is_returned_but_not_illustrated(request_for_a_cat):
    """The caller decides what to do with an unsafe book; serving does not throw it away.

    It also does not keep generating: pictures and audio for text we have just judged unsafe cost
    money and create artefacts of the thing we refused.
    """
    book = build(
        StoryRequest(prompt="a perfectly nice garden", age=5, target_words=("marker",)),
        classifier=FakeSafetyClassifier(unsafe_markers=("marker",)),
    )
    assert book.safety.flagged, "the marker is only in the story, not in the prompt"
    assert not book.complete
    assert book.pages > 0, "the story is returned so the caller can see what was refused"
    assert book.illustrations == (), "no illustrations for a refused story"
    assert book.narration == (), "no narration for a refused story"


def test_the_recognised_subject_reaches_the_writer_and_the_illustrator(request_for_a_cat):
    """The drawing is the point of the product; a book that ignores it is the wrong book."""
    seen: dict[str, object] = {}

    class RecordingWriter(FakeStoryWriter):
        def write(self, request: StoryRequest) -> Story:
            seen["subject"] = request.subject
            seen["drawing_ref"] = request.drawing_ref
            return super().write(request)

    book = build_book(
        request=request_for_a_cat,
        drawing=b"x",
        drawing_ref="gs://fixtures/cat.png",
        recognizer=FakeDrawingRecognizer(label="cat"),
        writer=RecordingWriter(),
        scorer=FakeReadabilityScorer(),
        classifier=FakeSafetyClassifier(),
        illustrator=FakeIllustrator(),
        synthesizer=FakeSpeechSynthesizer(),
    )

    assert seen["subject"] == "cat", "the writer was told what the child drew"
    assert seen["drawing_ref"] == "gs://fixtures/cat.png"
    assert all(illustration.subject == "cat" for illustration in book.illustrations)
    assert book.consistency > 0, "every page conditioned on the same subject"


def test_consistency_collapses_when_pages_disagree():
    """The fake scores 0 when pages were conditioned differently, so a threading bug is visible."""
    illustrator = FakeIllustrator()
    cat = Character(label="cat", confidence=1.0, embedding=(0.1, 0.2))
    dog = Character(label="dog", confidence=1.0, embedding=(0.9, 0.8))
    mixed = [
        illustrator.illustrate(0, "page one", "crayon", "cat", characters=(cat,)),
        illustrator.illustrate(1, "page two", "crayon", "dog", characters=(dog,)),
    ]
    assert illustrator.consistency(mixed) == 0.0
    assert illustrator.consistency(mixed[:1], (cat,)) > 0.0


def test_the_sketch_reaches_the_illustrator(request_for_a_cat):
    """KAN-42 conditions ControlNet on the child's drawing, so the bytes have to arrive.

    Requested by the illustration owner on PR #11. This is the structure half of the handoff; the
    identity half is the character embeddings, covered below.
    """
    received: dict[int, bytes | None] = {}

    class RecordingIllustrator(FakeIllustrator):
        def illustrate(self, page_index, page_text, style, subject=None, drawing_ref=None,
                       sketch=None, characters=()):
            received[page_index] = sketch
            return super().illustrate(
                page_index, page_text, style, subject, drawing_ref, sketch, characters
            )

    sketch = b"\x89PNG\r\n\x1a\n fake scribble"
    book = build(request_for_a_cat, illustrator=RecordingIllustrator(), drawing=sketch)

    assert received, "the illustrator was never called"
    assert all(value == sketch for value in received.values()), "every page gets the sketch"
    assert all(illustration.conditioned_on_sketch for illustration in book.illustrations)


def test_the_drawing_ref_reaches_every_illustrate_call(request_for_a_cat):
    """Serving must not drop the durable handle on the way to illustration.

    The existing subject test asserted `drawing_ref` reached the *writer* only, and
    `FakeIllustrator` discarded it, so replacing the argument with None in `book.py` left every
    test green. This fails under exactly that mutation.
    """
    received: list[str | None] = []

    class RecordingIllustrator(FakeIllustrator):
        def illustrate(self, page_index, page_text, style, subject=None, drawing_ref=None,
                       sketch=None, characters=()):
            received.append(drawing_ref)
            return super().illustrate(
                page_index, page_text, style, subject, drawing_ref, sketch, characters
            )

    ref = "gs://yip-drawings/child-42/cat.png"
    book = build(request_for_a_cat, illustrator=RecordingIllustrator(), drawing_ref=ref)

    assert len(received) == book.pages > 0, "every page calls the illustrator"
    assert received == [ref] * book.pages, "every page gets the same reference build_book was given"


def test_a_drawing_without_a_reference_is_refused(request_for_a_cat):
    """An implementation may assume drawing_ref is present, so serving has to guarantee it."""
    with pytest.raises(ValueError, match="drawing_ref is required"):
        build(request_for_a_cat, drawing_ref="")


def test_the_character_embedding_reaches_the_illustrator(request_for_a_cat):
    """The identity half: IP-Adapter conditions on the recogniser's character embedding."""
    received: list[tuple[Character, ...]] = []

    class RecordingIllustrator(FakeIllustrator):
        def illustrate(self, page_index, page_text, style, subject=None, drawing_ref=None,
                       sketch=None, characters=()):
            received.append(characters)
            return super().illustrate(
                page_index, page_text, style, subject, drawing_ref, sketch, characters
            )

    book = build(request_for_a_cat, illustrator=RecordingIllustrator())

    assert received, "the illustrator was never called"
    for characters in received:
        assert characters, "every page is told which characters it must look like"
        assert characters[0].label == "cat"
        assert characters[0].has_embedding
    # Every page carried the same identity, so the fake's consistency score is non-zero.
    assert book.consistency > 0


def test_the_description_reaches_the_writer():
    """The recogniser's other output: the story is about what the child drew, not just the prompt."""
    seen: dict[str, object] = {}

    class RecordingWriter(FakeStoryWriter):
        def write(self, request: StoryRequest) -> Story:
            seen["description"] = request.description
            return super().write(request)

    build(
        StoryRequest(prompt="a quiet afternoon", age=5, target_words=("cat",)),
        writer=RecordingWriter(),
        recognizer=FakeDrawingRecognizer(label="cat", also=("dog",)),
    )
    assert seen["description"] and "cat" in str(seen["description"])


def test_several_characters_survive_recognition():
    """A drawing with a child, a dog and a house is three characters, not one winner."""
    analysis = FakeDrawingRecognizer(label="child", also=("dog", "house")).analyse(b"x")
    assert [character.label for character in analysis.characters] == ["child", "dog", "house"]
    assert analysis.main is not None and analysis.main.label == "child"
    assert analysis.top_labels(5) == ["child", "dog", "house"]
    # Distinct identities, so the illustrator can tell them apart.
    assert len({character.embedding for character in analysis.characters}) == 3


def test_consistency_is_zero_without_an_identity_to_hold_to():
    """No embedding means nothing was conditioned, so the pages cannot be judged consistent."""
    illustrator = FakeIllustrator()
    pages = [
        illustrator.illustrate(0, "page one", "crayon", "cat"),
        illustrator.illustrate(1, "page two", "crayon", "cat"),
    ]
    assert illustrator.consistency(pages) == 0.0


def test_no_characters_is_an_error(request_for_a_cat):
    class Empty:
        def analyse(self, image: bytes) -> DrawingAnalysis:
            return DrawingAnalysis()

    with pytest.raises(ValueError, match="no characters"):
        build_book(
            request=request_for_a_cat,
            drawing=b"x",
            drawing_ref="gs://yip-drawings/test/cat.png",
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
