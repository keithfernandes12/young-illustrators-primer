"""S6e's character sheet: the pure decisions, and the recogniser end to end with fake models.

No weights, no torch, no Pillow, no bucket: models, decoder and PNG encoder are fakes, so CI runs
all of it. The real models are measured against real drawings in the plan's Verified facts.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from drawing_fakes import TEST_LABELS, RecordingEncoder

from primer.drawing import sheet
from primer.drawing.interface import KEYPOINT_NAMES, BodyType, DrawingRecognizer
from primer.drawing.model import DrawingRecognizerModel
from primer.drawing.sheet import (
    CharacterSheetRecognizer,
    MemoryArtifactStore,
    body_type_of,
    characters_in,
    cutouts,
    keypoints_in,
)

CONFIG = sheet.default_sheet_config()


# ------------------------------------------------------------------------------- characters_in


def test_a_part_inside_a_figure_is_not_a_second_character():
    """Florence-2 boxes faces and shoes beside the person; those fold into the figure."""
    found = characters_in(
        [("human face", (40, 10, 60, 30)), ("person", (20, 0, 80, 100)), ("footwear", (30, 90, 45, 100))],
        (100, 100),
    )

    assert found == [("person", (20.0, 0.0, 80.0, 100.0))]


def test_two_figures_side_by_side_are_two_characters_largest_first():
    found = characters_in([("person", (0, 0, 40, 90)), ("dog", (50, 50, 100, 100))], (100, 100))

    assert [label for label, _ in found] == ["person", "dog"]


def test_a_speck_is_dropped():
    found = characters_in([("person", (0, 0, 50, 100)), ("bird", (90, 0, 95, 5))], (100, 100))

    assert [label for label, _ in found] == ["person"]


def test_no_detection_leaves_the_whole_drawing_as_one_character_of_unknown_body():
    assert characters_in([], (64, 32)) == [(None, (0.0, 0.0, 64.0, 32.0))]


def test_boxes_are_clipped_to_the_drawing():
    assert characters_in([("person", (-5, -5, 120, 50))], (100, 100)) == [("person", (0.0, 0.0, 100.0, 50.0))]


def test_at_most_max_characters_are_kept():
    many = [("person", (i * 10, 0, i * 10 + 9, 100)) for i in range(10)]

    assert len(characters_in(many, (100, 100), replace(CONFIG, max_characters=3))) == 3


# -------------------------------------------------------------------------------- body_type_of


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("person", BodyType.HUMANOID),
        ("Fancy Dress", BodyType.HUMANOID),
        ("dog", BodyType.CREATURE),
        ("house", BodyType.OBJECT),
        ("cattle", BodyType.OBJECT),  # whole words: `cat` must not match inside it
        (None, BodyType.UNKNOWN),
    ],
)
def test_the_detector_word_picks_the_reposing_branch(label, expected):
    assert body_type_of(label) is expected


def test_a_word_cannot_be_both_humanoid_and_creature():
    with pytest.raises(ValueError, match="both humanoid and creature"):
        replace(CONFIG, creature_words=(*CONFIG.creature_words, "person"))


def test_the_shipped_config_loads_and_pins_every_model():
    loaded = sheet.SheetConfig.from_yaml()

    assert loaded == CONFIG
    for name in ("florence_revision", "sam_revision", "vitpose_revision"):
        assert len(getattr(loaded, name)) == 40, f"{name} is a full commit sha"
    assert {loaded.florence_license, loaded.sam_license, loaded.vitpose_license} <= {"mit", "apache-2.0"}


# -------------------------------------------------------------------------------- keypoints_in


def test_keypoints_are_normalised_to_the_crop_positional_and_never_dropped():
    points = [(10 + i, 20 + 2 * i, 0.9 if i % 2 else 0.1) for i in range(17)]

    joints = keypoints_in(points, (10, 20, 30, 60), threshold=0.3)

    assert [j.name for j in joints] == list(KEYPOINT_NAMES)
    assert joints[0].x == 0.0 and joints[0].y == 0.0 and not joints[0].visible
    assert joints[1].x == pytest.approx(1 / 20) and joints[1].y == pytest.approx(2 / 40) and joints[1].visible


def test_a_joint_outside_the_crop_is_clipped_to_its_edge():
    points = [(0.0, 0.0, 1.0)] * 16 + [(500.0, -3.0, 1.0)]

    assert keypoints_in(points, (10, 10, 20, 20), threshold=0.3)[-1].x == 1.0


def test_a_skeleton_with_the_wrong_number_of_joints_is_refused():
    with pytest.raises(ValueError, match="17 COCO joints"):
        keypoints_in([(0, 0, 1)] * 16, (0, 0, 10, 10), threshold=0.3)


# ------------------------------------------------------------------------------------ cutouts


def test_the_cutout_keeps_the_childs_lines_and_turns_the_rest_to_paper():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:5, 3:5] = True

    crop, inside = cutouts(image, mask, (2, 2, 6, 6))

    assert crop.shape == (4, 4, 3) and inside.shape == (4, 4)
    assert (crop[1:3, 1:3] == 0).all(), "the character's own pixels"
    assert (crop[0] == 255).all(), "paper outside the mask"


def test_a_mask_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="does not match"):
        cutouts(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((5, 5), dtype=bool), (0, 0, 5, 5))


# ---------------------------------------------------------------------------- artifact store


def test_store_refs_are_content_hashes_that_say_nothing_about_the_child():
    store = MemoryArtifactStore()

    ref = store.put(b"crop bytes", "crop")

    assert ref.startswith("mem://crop/") and len(ref.rsplit("/", 1)[1]) == 64
    assert store.put(b"crop bytes", "crop") == ref, "the same crop is one entry"
    with pytest.raises(ValueError, match="plain word"):
        store.put(b"x", "../escape")


# ------------------------------------------------------------------------- the whole recogniser


class FakeModels:
    """Two characters: a person on the left, a dog on the right, with a face inside the person."""

    def __init__(self, detections=None):
        self.detections = (
            detections
            if detections is not None
            else [("person", (0, 0, 20, 40)), ("human face", (5, 2, 15, 12)), ("dog", (24, 20, 40, 40))]
        )
        self.posed: list[tuple] = []

    def caption(self, image):
        return " a drawing of a person and a dog "

    def detect(self, image):
        return list(self.detections)

    def segment(self, image, box):
        mask = np.zeros(image.shape[:2], dtype=bool)
        x0, y0, x1, y1 = (int(v) for v in box)
        mask[y0:y1, x0:x1] = True
        return mask

    def pose(self, image, box):
        self.posed.append(box)
        return [(box[0] + 1, box[1] + 1, 0.9)] * 17


def recogniser(config, models=None, store=None):
    scores = tuple(1.0 - i / 10 for i in range(len(TEST_LABELS)))
    encoder = RecordingEncoder(scores)
    built = CharacterSheetRecognizer(
        recognizer=DrawingRecognizerModel(config, encoder=encoder),
        models=models or FakeModels(),
        store=store or MemoryArtifactStore(),
        decoder=lambda data: np.full((40, 40, 3), 255, dtype=np.uint8),
        encoder=lambda pixels: pixels.tobytes() + str(pixels.shape).encode(),
    )
    return built, encoder


def test_a_two_character_drawing_returns_two_complete_sheets(config, drawing):
    """S6e's acceptance, on fake models: one sheet per character, every field `analyse` owns."""
    built, encoder = recogniser(config)

    analysis = built.analyse(drawing)

    assert isinstance(built, DrawingRecognizer)
    assert analysis.description == "a drawing of a person and a dog"
    assert len(analysis.characters) == 2, "the face folded into the person"
    person, dog = analysis.characters
    assert person.body_type is BodyType.HUMANOID and dog.body_type is BodyType.CREATURE
    for character in analysis.characters:
        assert character.candidates and character.embedding
        assert character.crop_ref.startswith("mem://crop/") and character.mask_ref.startswith("mem://mask/")
        assert not character.reference_image_ref and not character.approved, "the approve step sets those"
    assert len(person.keypoints) == 17 and dog.keypoints == (), "joints for humanoids only"
    assert encoder.boxes == [(0, 0, 20, 40), (24, 20, 16, 20)], "each scored on its own box, as x/y/w/h"


def test_only_humanoids_are_posed(config, drawing):
    models = FakeModels()
    built, _ = recogniser(config, models)

    built.analyse(drawing)

    assert models.posed == [(0.0, 0.0, 20.0, 40.0)]


def test_a_drawing_with_nothing_detected_is_one_sheet_of_unknown_body(config, drawing):
    built, encoder = recogniser(config, FakeModels(detections=[]))

    analysis = built.analyse(drawing)

    assert len(analysis.characters) == 1
    assert analysis.characters[0].body_type is BodyType.UNKNOWN
    assert analysis.characters[0].crop_ref, "the whole drawing is still cut out"
    assert encoder.boxes == [(0.0, 0.0, 40.0, 40.0)]


def test_two_characters_get_two_different_crops(config, drawing):
    store = MemoryArtifactStore()
    built, _ = recogniser(config, store=store)

    person, dog = built.analyse(drawing).characters

    assert person.crop_ref != dog.crop_ref
    assert {person.crop_ref, person.mask_ref, dog.crop_ref, dog.mask_ref} <= set(store.items)


def test_no_bytes_is_refused(config):
    built, _ = recogniser(config)

    with pytest.raises(ValueError, match="bytes"):
        built.analyse(b"")


def test_every_character_sheet_field_reaches_the_illustrator_from_the_real_recogniser(config, drawing):
    """S6e's acceptance, through `build_book`: the real recogniser class on fake weights.

    `reference_image_ref` and `approved` are set by the stylise-and-approve step, which does not
    exist yet (KAN-64), so a stand-in approves each sheet -- exactly the step the Fake fakes.
    Everything else on the sheet comes from `CharacterSheetRecognizer.analyse` unaltered.
    """
    from primer.drawing.interface import DrawingAnalysis
    from primer.illustration.interface import FakeIllustrator
    from primer.readability.interface import FakeReadabilityScorer
    from primer.safety.interface import FakeSafetyClassifier
    from primer.serving.book import build_book
    from primer.speech.interface import FakeSpeechSynthesizer
    from primer.story.interface import FakeStoryWriter, StoryRequest

    real, _ = recogniser(config)

    class ApprovedByAParent:
        def analyse(self, image):
            found = real.analyse(image)
            approved = tuple(c.approve(f"fake://reference/{i}") for i, c in enumerate(found.characters))
            return DrawingAnalysis(description=found.description, characters=approved)

    received = []

    class Recording(FakeIllustrator):
        def illustrate(
            self, page_index, page_text, style, subject=None, drawing_ref=None, sketch=None, characters=()
        ):
            received.append(characters)
            return super().illustrate(page_index, page_text, style, subject, drawing_ref, sketch, characters)

    build_book(
        request=StoryRequest(prompt="a person and a dog", age=5, target_words=("dog",)),
        drawing=drawing,
        drawing_ref="gs://yip-drawings/test/drawing.png",
        recognizer=ApprovedByAParent(),
        writer=FakeStoryWriter(),
        scorer=FakeReadabilityScorer(),
        classifier=FakeSafetyClassifier(),
        illustrator=Recording(),
        synthesizer=FakeSpeechSynthesizer(),
    )

    assert received, "the illustrator was never called"
    for characters in received:
        humanoid = next(c for c in characters if c.body_type is BodyType.HUMANOID)
        assert humanoid.crop_ref and humanoid.mask_ref
        assert len(humanoid.keypoints) == len(KEYPOINT_NAMES)
        assert humanoid.has_approved_reference and humanoid.candidates
        assert len(characters) == 2, "both characters reach the illustrator"
