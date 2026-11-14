"""The zero-shot recognition baseline (KAN-38, plan slice S6a).

What this slice has to get right is not accuracy -- it is wiring, and specifically the two places
where a silent bug costs a wrong number later: the positional pairing of scores to labels, and the
agreement between `label` and `candidates[0]` that `drawing_top5_acc` is read from.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fakes import TEST_LABELS, RecordingEncoder

from primer.drawing.interface import BodyType, DrawingRecognizer
from primer.drawing.model import (
    CONFIG_PATH,
    DrawingConfig,
    DrawingRecognizerModel,
    Encoding,
    SiglipEncoder,
    l2_normalised,
    pooled,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "primer" / "drawing"

#: Scores that make "dog" the winner and rank the rest, over `TEST_LABELS`.
DOG_WINS = (0.10, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40)


def recogniser(config, scores=DOG_WINS, **kwargs):
    encoder = RecordingEncoder(scores, **kwargs)
    return DrawingRecognizerModel(config, encoder=encoder), encoder


# ------------------------------------------------------------------ the contract serving imports


def test_the_model_satisfies_the_recogniser_protocol(config):
    """Serving composes areas by Protocol, so this is what lets the Fake be swapped out."""
    model, _ = recogniser(config)
    assert isinstance(model, DrawingRecognizer)


def test_building_the_model_loads_nothing(config):
    """Construction must be free: serving imports every area at start-up.

    The encoder is only built on first use, and `SiglipEncoder.__init__` itself holds no weights,
    so this stays true whether or not one was injected.
    """
    model = DrawingRecognizerModel(config)
    assert model._encoder is None, "no encoder until something is analysed"
    assert isinstance(model.encoder, SiglipEncoder), "and then it is the real one"
    assert model.encoder._model is None, "built, but still no weights"


@pytest.mark.parametrize("module", ["model.py", "smoke.py"])
def test_the_heavy_imports_are_not_at_module_level(module):
    """`torch`, `transformers` and `PIL` come from the `drawing` extra, which CI does not install.

    An import at the top of `model.py` would break every area's tests and serving's start-up, and
    it is the kind of thing an editor's auto-import adds back without anyone noticing. Checked by
    reading the source rather than `sys.modules`, so the test says the same thing on a machine
    where the extra happens to be installed.
    """
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    heavy = {"torch", "transformers", "PIL", "diffusers"}
    offenders = []
    for node in tree.body:  # module level only; inside a function is the whole point
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        offenders += [name for name in names if name.split(".")[0] in heavy]
    assert not offenders, f"{module} imports {offenders} at module level"


# ------------------------------------------------------------------------------------- analysing


def test_analyse_returns_one_ranked_character(config, drawing):
    """One character per drawing for now: the whole image is the subject."""
    model, _ = recogniser(config)
    analysis = model.analyse(drawing)

    assert len(analysis.characters) == 1
    character = analysis.main
    assert character is not None
    assert character.label == "dog"
    assert character.confidence == pytest.approx(0.90)
    assert [candidate.label for candidate in character.candidates] == [
        "dog", "house", "dragon", "tree", "sun",
    ]
    confidences = [candidate.confidence for candidate in character.candidates]
    assert confidences == sorted(confidences, reverse=True)


def test_top_five_has_five_entries(config, drawing):
    """S6a's acceptance criterion, and what `drawing_top5_acc` will read in S6c."""
    model, _ = recogniser(config)
    analysis = model.analyse(drawing)
    assert len(analysis.top_labels(5)) == 5
    assert analysis.top_labels(5) == analysis.main.top_labels(5), "scored on the main character"


def test_scores_are_paired_with_labels_positionally(config, drawing):
    """The bug this guards is invisible in a metric: every label shifted by one.

    The encoder is given prompts in label order and returns confidences in that same order. A test
    that only checked "the winner has the highest score" would pass under exactly that mutation,
    so this moves the winning score to each index in turn and demands the matching label.
    """
    model = DrawingRecognizerModel(config)
    for index, expected in enumerate(TEST_LABELS):
        scores = [0.1] * len(TEST_LABELS)
        scores[index] = 0.99
        model._encoder = RecordingEncoder(scores)
        assert model.analyse(drawing).main.label == expected


def test_the_encoder_is_given_one_templated_prompt_per_label(config, drawing):
    """The checkpoint was trained to read a sentence, not a bare noun, and that changes the score."""
    model, encoder = recogniser(config)
    model.analyse(drawing)

    assert encoder.prompts == tuple(f"This is a photo of {label}." for label in TEST_LABELS)
    assert len(encoder.prompts) == len(TEST_LABELS)


def test_the_drawing_is_encoded_once(config, drawing):
    """Scoring 345 prompts must not mean 345 passes over the image (KAN-71's latency budget)."""
    model, encoder = recogniser(config)
    model.analyse(drawing)
    assert len(encoder.calls) == 1
    assert encoder.calls[0][0] == drawing, "the bytes reach the tower unchanged"


def test_the_embedding_comes_from_the_same_tower(config, drawing):
    """KAN-71's cache key. Rounded, so float noise cannot change `identity_key`."""
    model, _ = recogniser(config, dimensions=6)
    character = model.analyse(drawing).main

    assert character.has_embedding
    assert len(character.embedding) == 6
    assert character.identity_key.startswith("dog:")
    assert model.analyse(drawing).main.identity_key == character.identity_key, "stable across runs"


def test_the_structural_fields_keep_their_defaults(config, drawing):
    """S6e fills these from different models. A guess here is worse than an empty field.

    `body_type` in particular: KAN-42 picks its reposing branch from it, and the interface says
    UNKNOWN means "do not repose" rather than "pick the default branch".
    """
    model, _ = recogniser(config)
    analysis = model.analyse(drawing)
    character = analysis.main

    assert analysis.description == "", "the description is a captioning VLM, not this classifier"
    assert character.body_type is BodyType.UNKNOWN
    assert character.keypoints == ()
    assert character.crop_ref == "" and character.mask_ref == ""
    assert not character.approved and character.reference_image_ref == ""


# ----------------------------------------------------------------------------- refusing bad input


def test_an_empty_drawing_is_refused(config):
    model, _ = recogniser(config)
    with pytest.raises(ValueError, match="got none"):
        model.analyse(b"")


def test_a_score_vector_of_the_wrong_length_is_refused(config, drawing):
    """Positional pairing only holds if the lengths match; silently zipping short mislabels.

    This is the failure mode of a vocabulary changed on one side only -- a labels file edited
    without re-encoding the prompts -- and it must be an error rather than a shifted ranking.
    """
    model, _ = recogniser(config, scores=DOG_WINS[:3])
    with pytest.raises(ValueError, match="confidences are positional"):
        model.analyse(drawing)


# ------------------------------------------------------------------------------------ the config


def test_the_shipped_config_loads_and_pins_its_weights():
    """A revision that is not pinned means a rerun can measure different weights."""
    config = DrawingConfig.from_yaml()

    assert config.encoder == "google/siglip2-base-patch16-224"
    assert len(config.encoder_revision) == 40, "a full commit sha, not a branch name"
    assert config.encoder_license == "apache-2.0"
    assert config.top_k >= 5, "drawing_top5_acc needs five candidates"


def test_the_shipped_vocabulary_is_the_345_quickdraw_categories():
    """The gate is measured over these labels, so the count is part of the contract.

    Sorted and unique because that is the bucket listing order `primer.data.sources.quickdraw`
    counts category indexes in; a reordering here would quietly change what index 7 means.
    """
    labels = DrawingConfig.from_yaml().labels()

    assert len(labels) == 345
    assert len(set(labels)) == 345, "a duplicated label would be scored twice"
    assert list(labels) == sorted(labels)
    assert "cat" in labels and "The Eiffel Tower" in labels
    assert not any(label.startswith("#") for label in labels), "comments are not labels"


def test_a_config_missing_a_field_fails_loudly(tmp_path):
    """`from_yaml` has no fallbacks on purpose.

    A default for `prompt_template` or `top_k` would let a config drift from the settings a number
    was measured at, and the drift would show up as an unexplained change in `drawing_top5_acc`
    rather than as an error.
    """
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    assert "encoder_revision:" in raw, "the shipped config pins its weights"

    kept = [line for line in raw.splitlines() if not line.startswith("top_k:")]
    path = tmp_path / "incomplete.yaml"
    path.write_text("\n".join(kept), encoding="utf-8")

    with pytest.raises(KeyError, match="top_k"):
        DrawingConfig.from_yaml(path)


# ------------------------------------------------- the one version-dependent bit of the real tower


class _Pooling:
    """What `transformers` 5.x hands back from `get_text_features` / `get_image_features`."""

    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


def test_pooled_unwraps_a_model_output_and_passes_a_tensor_through():
    """A regression test for a bug the fake encoder cannot reach.

    `smoke.py` found it on the first real call: `get_text_features` returns a
    `BaseModelOutputWithPooling`, not a tensor, so normalising its result raised `AttributeError`.
    The tests inject an `Encoder` and never run `SiglipEncoder`, so nothing here would have caught
    it -- which is why the one version-dependent assumption now lives in a function of its own.
    """
    vectors = [[1.0, 2.0]]
    assert pooled(_Pooling(vectors)) is vectors, "the 5.x shape is unwrapped"
    assert pooled(vectors) is vectors, "a bare tensor, as the model card's example shows, passes"


def test_l2_normalised_divides_by_the_last_axis_norm():
    """Checked against a stand-in rather than `torch`, which CI does not install.

    SigLIP scores unit vectors; skipping this leaves the logits scaled by each vector's magnitude
    and saturates the sigmoid, which would look like "every label is confident" rather than a crash.
    """

    class Vector:
        def __init__(self, value):
            self.value = value

        def norm(self, p, dim, keepdim):
            assert (p, dim, keepdim) == (2, -1, True), "normalise along the feature axis, per row"
            return 2.0

        def __truediv__(self, other):
            return Vector(self.value / other)

    assert l2_normalised(Vector(8.0)).value == 4.0


def test_encoding_is_positional():
    """The dataclass the seam is written in: confidences line up with prompts, by index."""
    encoding = Encoding(confidences=(0.1, 0.9), embedding=(1.0,))
    assert encoding.confidences[1] == 0.9
    assert len(encoding.embedding) == 1
