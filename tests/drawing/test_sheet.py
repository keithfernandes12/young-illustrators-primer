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


def test_no_bytes_is_refused_by_the_sheet_before_any_model_runs(config):
    """The recogniser underneath refuses empty bytes too, with the same words; this must be the sheet."""

    class NoModelMayRun(FakeModels):
        def detect(self, image):
            raise AssertionError("a model ran on no bytes")

    built, _ = recogniser(config, NoModelMayRun())

    with pytest.raises(ValueError, match="analyse needs the drawing's bytes"):
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


# ------------------------------------------------------------ order: most confident first, not largest


class ConfidenceByBox(RecordingEncoder):
    """A smaller box scores higher, so confidence order is the reverse of area order."""

    def encode(self, image, prompts, *, box=None):
        encoding = super().encode(image, prompts, box=box)
        small = box is not None and box[2] * box[3] < 500
        scores = (0.9 if small else 0.4, *encoding.confidences[1:])
        return type(encoding)(confidences=scores, embedding=encoding.embedding)


def test_characters_come_back_most_confident_first_not_largest_first(config, drawing):
    """`main` is the story's subject: a big, doubtful background object must not outrank a sure one."""
    encoder = ConfidenceByBox([0.0] * len(TEST_LABELS))
    built = CharacterSheetRecognizer(
        recognizer=DrawingRecognizerModel(config, encoder=encoder),
        models=FakeModels(),
        store=MemoryArtifactStore(),
        decoder=lambda data: np.full((40, 40, 3), 255, dtype=np.uint8),
        encoder=lambda pixels: pixels.tobytes() + str(pixels.shape).encode(),
    )

    analysis = built.analyse(drawing)

    assert encoder.boxes == [(0, 0, 20, 40), (24, 20, 16, 20)], "detected largest first: the person, then the dog"
    assert [c.body_type for c in analysis.characters] == [BodyType.CREATURE, BodyType.HUMANOID]
    assert analysis.main.body_type is BodyType.CREATURE, "the dog is the confident one, and the subject"
    assert analysis.characters[0].confidence > analysis.characters[1].confidence


def test_equal_confidences_keep_the_largest_first_order(config, drawing):
    built, _ = recogniser(config)

    person, dog = built.analyse(drawing).characters

    assert person.confidence == dog.confidence
    assert person.body_type is BodyType.HUMANOID and dog.body_type is BodyType.CREATURE


# ----------------------------------------------------------------------- grouping, in the details


def test_overlapping_figures_that_are_not_contained_stay_two_characters():
    """The real failure mode: two figures that overlap a little are not one figure and a part."""
    found = characters_in([("person", (0, 0, 60, 60)), ("person", (40, 40, 100, 100))], (100, 100))

    assert len(found) == 2


def test_containment_folds_at_exactly_the_configured_share_and_not_below_it():
    """`containment` is 'or more inside': 0.80 folds, 0.79 does not."""
    outer = ("person", (0, 0, 100, 100))

    folded = characters_in([outer, ("face", (20, 0, 120, 100))], (200, 200))
    kept = characters_in([outer, ("face", (21, 0, 121, 100))], (200, 200))

    assert CONFIG.containment == 0.8
    assert len(folded) == 1 and len(kept) == 2


def test_a_box_wholly_outside_the_drawing_is_dropped_even_when_min_area_is_zero():
    found = characters_in(
        [("person", (0, 0, 50, 50)), ("dog", (150, 150, 200, 200))], (100, 100), replace(CONFIG, min_area=0.0)
    )

    assert [label for label, _ in found] == ["person"]


def test_a_humanoid_word_wins_over_a_creature_word_in_the_same_label():
    assert body_type_of("cat person") is BodyType.HUMANOID
    assert body_type_of("person cat") is BodyType.HUMANOID


# ----------------------------------------------------------------- keypoints against the saved crop


def test_keypoints_are_normalised_against_the_rounded_crop_not_the_raw_box():
    """A fractional detection is cropped at (10, 20)-(51, 81): joints must land on that crop's pixels."""
    box = (10.4, 20.6, 50.2, 80.9)
    assert sheet.crop_bounds(box) == (10, 20, 51, 81)
    points = [(30.0, 40.0, 0.9)] * 15 + [(10.0, 20.0, 0.9), (51.0, 81.0, 0.9)]

    joints = keypoints_in(points, box, threshold=0.3)

    assert joints[0].x == pytest.approx(20 / 41) and joints[0].y == pytest.approx(20 / 61)
    assert (joints[15].x, joints[15].y) == (0.0, 0.0), "the crop's own top-left corner"
    assert (joints[16].x, joints[16].y) == (1.0, 1.0), "and its bottom-right"


def test_a_keypoint_and_the_saved_crop_use_the_same_pixel_bounds():
    box = (3.7, 4.2, 9.1, 8.6)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    crop, _ = cutouts(image, np.ones((20, 20), dtype=bool), box)
    x0, y0, x1, y1 = sheet.crop_bounds(box)

    assert crop.shape[:2] == (y1 - y0, x1 - x0)


def test_a_joint_exactly_at_the_threshold_is_visible_and_just_below_is_not():
    points = [(5.0, 5.0, 0.3)] * 16 + [(5.0, 5.0, 0.299999)]

    joints = keypoints_in(points, (0, 0, 10, 10), threshold=0.3)

    assert joints[0].visible and not joints[16].visible


def test_a_joint_above_the_crop_is_clipped_to_its_top_edge():
    points = [(5.0, 5.0, 1.0)] * 16 + [(5.0, -3.0, 1.0)]

    assert keypoints_in(points, (0, 0, 10, 10), threshold=0.3)[16].y == 0.0
    assert keypoints_in([(5.0, 500.0, 1.0)] * 17, (0, 0, 10, 10), threshold=0.3)[0].y == 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_joint_that_is_not_finite_is_kept_but_not_visible(bad):
    points = [(5.0, 5.0, 0.9)] * 16 + [(bad, 5.0, 0.9)]

    last = keypoints_in(points, (0, 0, 10, 10), threshold=0.3)[16]

    assert not last.visible and (last.x, last.y) == (0.0, 0.0)
    assert not keypoints_in([(5.0, 5.0, bad)] * 17, (0, 0, 10, 10), threshold=0.3)[0].visible


@pytest.mark.parametrize("box", [(10, 10, 10, 20), (10, 10, 20, 10), (20, 20, 10, 10)])
def test_a_box_with_no_area_cannot_carry_keypoints(box):
    with pytest.raises(ValueError, match="no area"):
        keypoints_in([(0, 0, 1)] * 17, box, threshold=0.3)


# -------------------------------------------------------------------------------- cutouts, in the details


def test_the_cutout_rounds_the_box_outward_and_leaves_the_drawing_untouched():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    before = image.copy()
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 3] = True

    crop, _ = cutouts(image, mask, (2.4, 2.6, 5.2, 5.9))

    assert crop.shape == (4, 4, 3), "floor of the top-left, ceil of the bottom-right: x 2..6, y 2..6"
    assert np.array_equal(image, before), "whitening the crop must not whiten the drawing the next character is scored on"
    assert (crop[0] == 255).all()


def test_a_box_outside_the_drawing_cannot_be_cut_out():
    with pytest.raises(ValueError, match="outside the drawing"):
        cutouts(np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10), dtype=bool), (50, 50, 60, 60))


def test_an_empty_mask_falls_back_to_the_unmasked_box_instead_of_erasing_the_childs_lines():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    crop, inside = cutouts(image, np.zeros((10, 10), dtype=bool), (2, 2, 6, 6))

    assert (crop == 0).all(), "the child's lines are kept"
    assert inside.shape == (4, 4) and inside.all(), "and the mask says no cut-out was made"


def test_the_stored_crop_and_mask_are_the_crop_and_the_mask(config, drawing):
    """Each ref must point at its own bytes: a mask stored under the crop's name is a different file."""
    stored = []

    def encode(pixels):
        stored.append(pixels)
        return pixels.tobytes() + str(pixels.shape).encode()

    store = MemoryArtifactStore()
    built, _ = recogniser(config, store=store)
    built._encode = encode

    person, _ = built.analyse(drawing).characters

    crop, mask = stored[0], stored[1]
    assert crop.dtype == np.uint8 and mask.dtype == np.bool_
    assert store.items[person.crop_ref] == crop.tobytes() + str(crop.shape).encode()
    assert store.items[person.mask_ref] == mask.tobytes() + str(mask.shape).encode()
    assert person.crop_ref != person.mask_ref


# --------------------------------------------------------------------------------- the config


@pytest.mark.parametrize("name", ["keypoint_threshold", "containment", "min_area"])
@pytest.mark.parametrize("value", [-0.1, 1.5, True, "0.5"])
def test_a_share_outside_zero_to_one_is_refused(name, value):
    with pytest.raises(ValueError, match="share in 0-1"):
        replace(CONFIG, **{name: value})


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_max_characters_must_be_a_whole_number_of_at_least_one(value):
    with pytest.raises(ValueError, match="whole number >= 1"):
        replace(CONFIG, max_characters=value)


def test_a_config_file_with_an_unknown_or_missing_key_is_refused(tmp_path):
    import yaml

    shipped = yaml.safe_load(sheet.CONFIG_PATH.read_text(encoding="utf-8"))
    extra = tmp_path / "extra.yaml"
    extra.write_text(yaml.safe_dump({**shipped, "surprise": 1}), encoding="utf-8")
    short = tmp_path / "short.yaml"
    short.write_text(yaml.safe_dump({k: v for k, v in shipped.items() if k != "containment"}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown \['surprise'\]"):
        sheet.SheetConfig.from_yaml(extra)
    with pytest.raises(ValueError, match=r"missing \['containment'\]"):
        sheet.SheetConfig.from_yaml(short)


# ---------------------------------------------------------------- a failing model says which one


def test_a_failing_model_is_named_with_its_step_and_character(config, drawing):
    class SegmentFails(FakeModels):
        def segment(self, image, box):
            raise ValueError("boom")

    built, _ = recogniser(config, SegmentFails())

    with pytest.raises(RuntimeError, match="character 0: segment failed: boom"):
        built.analyse(drawing)


def test_a_malformed_detection_names_the_detect_step(config, drawing):
    class DetectFails(FakeModels):
        def detect(self, image):
            raise ValueError("zip() argument 2 is shorter than argument 1")

    built, _ = recogniser(config, DetectFails())

    with pytest.raises(RuntimeError, match="detect failed: zip"):
        built.analyse(drawing)


def test_a_failing_pose_names_the_character_that_was_being_posed(config, drawing):
    class PoseFails(FakeModels):
        def pose(self, image, box):
            raise ValueError("no joints")

    built, _ = recogniser(config, PoseFails())

    with pytest.raises(RuntimeError, match="character 0: pose failed: no joints"):
        built.analyse(drawing)
