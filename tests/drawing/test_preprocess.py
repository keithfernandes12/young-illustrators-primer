"""`preprocess.py`: both directions of KAN-31, and the guards that keep them the only ones.

Everything here runs on `numpy` alone -- no Pillow, no torch -- so CI covers all of it. The images
are built from arrays or drawn by `render_strokes`, never decoded, and none is a child's drawing.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_render import REPO, owned_python_files

from primer.drawing import model, preprocess
from primer.drawing.preprocess import (
    AugmentConfig,
    PrepareConfig,
    augment_canvas,
    default_augment_config,
    default_prepare_config,
    prepare_drawing,
)
from primer.drawing.render import RenderConfig, default_config, render_strokes

#: A 4px canvas: small enough to assert every pixel, and `size` is all the inference side reads.
FOUR = RenderConfig(name="four", size=4, line_width=1, padding=0, background=255, ink=0, source_grid=4)

#: Everything off and the paper plain white, so any change in an output is the operation under test.
QUIET = replace(
    default_augment_config(),
    wobble_probability=0.0,
    erasure_probability=0.0,
    ink_colour_probability=0.0,
    fill_probability=0.0,
    lighting_probability=0.0,
    paper_tints=((255, 255, 255),),
    paper_texture=0.0,
    paper_grain=0.0,
)

#: A closed square with a diagonal through it, at the shipped render settings.
SQUARE = [[[40, 215, 215, 40, 40], [40, 40, 215, 215, 40]], [[40, 215], [40, 215]]]


def colour_only(**overrides: object) -> PrepareConfig:
    return replace(default_prepare_config(), **overrides)


def rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[:, :, None], 3, axis=2)


@pytest.fixture
def canvas() -> np.ndarray:
    return render_strokes(SQUARE)


# ------------------------------------------------------------------------------------- inference


def test_a_rendered_canvas_reaches_the_recogniser_exactly_as_the_renderer_drew_it(canvas):
    """The contract between the directions: inference does not move a training image."""
    assert np.array_equal(prepare_drawing(canvas), rgb(canvas))


@pytest.mark.parametrize("shape", [(224, 224), (512, 384, 3), (300, 512), (7, 3, 3), (1, 1)])
def test_every_drawing_arrives_as_a_size_square_of_three_uint8_channels(shape):
    image = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)

    prepared = prepare_drawing(image)

    assert prepared.shape == (224, 224, 3)
    assert prepared.dtype == np.uint8


def test_a_wide_drawing_is_padded_with_its_own_paper_not_stretched():
    """The recogniser's processor squashes to a square; this is why it never gets the chance."""
    wide = np.full((2, 4), 200, dtype=np.uint8)  # grey paper, as a photograph has
    wide[0, 1] = 0  # one mark

    prepared = prepare_drawing(wide, colour_only(render=FOUR))[:, :, 0]

    # Stretched, the mark would cover two rows; padded with white, rows 0 and 3 would be 255.
    assert prepared.tolist() == [[200] * 4, [200, 0, 200, 200], [200] * 4, [200] * 4]


def test_a_tall_drawing_is_centred_between_two_margins_of_paper():
    tall = np.full((4, 2), 230, dtype=np.uint8)
    tall[1, 0] = 10

    prepared = prepare_drawing(tall, colour_only(render=FOUR))[:, :, 0]

    assert prepared.tolist() == [[230] * 4, [230, 10, 230, 230], [230] * 4, [230] * 4]


def test_the_padding_takes_the_paper_colour_from_the_drawings_edge():
    cream = np.zeros((2, 4, 3), dtype=np.uint8)
    cream[:, :] = (246, 240, 225)
    cream[1, 2] = (200, 30, 40)

    prepared = prepare_drawing(cream, colour_only(render=FOUR))

    assert prepared[0, 0].tolist() == [246, 240, 225] and prepared[3, 3].tolist() == [246, 240, 225]


def test_binarising_a_non_square_photograph_thresholds_the_drawing_not_the_padding():
    """The measured bug: white padding was the brightest class, so the whole page became ink.

    On 60 real Amateur Drawings the median came out 74.2% ink, the worst exactly the 75% of a
    384 x 512 drawing that is left after padding it square. Grey paper here, as there.
    """
    photo = np.full((4, 8), 180, dtype=np.uint8)
    photo[1:3, 2] = 40
    binary = colour_only(render=replace(FOUR, size=8), greyscale=True, binarise=True)

    prepared = prepare_drawing(photo, binary)[:, :, 0]

    assert (prepared == 0).sum() == 2, "the two marks, and nothing else"
    assert prepared[3, 0] == 255 and prepared[0, 0] == 255, "page and padding are both paper"


def test_shrinking_averages_the_pixels_each_output_pixel_covers():
    """Area resampling: a thin line fades rather than falling between samples."""
    image = np.full((8, 8), 255, dtype=np.uint8)
    image[:, 1] = 0  # a one-pixel line, a quarter of each 2x2 block it crosses

    prepared = prepare_drawing(image, colour_only(render=FOUR))[:, :, 0]

    assert prepared[:, 0].tolist() == [128] * 4, "(255 + 0) / 2 = 127.5, rounded half up"
    assert prepared[:, 1:].tolist() == [[255, 255, 255]] * 4


def test_shrinking_by_a_fraction_weights_partly_covered_pixels_by_their_overlap():
    image = np.array([[0, 255, 255]], dtype=np.uint8).repeat(3, axis=0)
    two = RenderConfig(name="two", size=2, line_width=1, padding=0, background=255, ink=0, source_grid=2)

    prepared = prepare_drawing(image, colour_only(render=two))[:, :, 0]

    # Each output column covers 1.5 source columns: [0, 1.5) and [1.5, 3).
    assert prepared[0].tolist() == [85, 255], "(0 * 1 + 255 * 0.5) / 1.5 = 85"


def test_enlarging_a_drawing_keeps_its_levels():
    image = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    prepared = prepare_drawing(image, colour_only(render=FOUR))[:, :, 0]

    assert prepared.tolist() == [[0, 0, 255, 255], [0, 0, 255, 255], [255, 255, 0, 0], [255, 255, 0, 0]]


def test_greyscale_uses_the_bt601_weights_on_all_three_channels():
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 20, 30]]], dtype=np.uint8)
    image = image.repeat(4, axis=0)

    prepared = prepare_drawing(image, colour_only(render=FOUR, greyscale=True))

    assert prepared[0].tolist() == [[76] * 3, [150] * 3, [29] * 3, [18] * 3]


def test_colour_survives_when_greyscale_is_off():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    prepared = prepare_drawing(image, colour_only(render=FOUR))

    assert prepared[0, 0].tolist() == [255, 0, 0]


def faint_drawing_on_shaded_paper() -> np.ndarray:
    """Paper shading from 200 to 250 across the page, with a pencil line at 150 on each side.

    Both the line and the paper are above 128, so a fixed mid-grey threshold calls the whole page
    paper.
    """
    page = np.tile(np.linspace(200, 250, 64), (64, 1))
    page[10:54, 5:8] = 150
    page[10:54, 56:59] = 150
    return np.floor(page + 0.5).astype(np.uint8)


def drawing_on_steeply_shaded_paper() -> np.ndarray:
    """Paper from 140 in shadow to 250 in light, a mark on each side, each 20%+ darker than its paper.

    What defeats any single threshold for the page: the shadowed paper is darker than the mark on
    the lit side, so no one level separates marks from paper. Global Otsu on the drawing returned
    the shaded half as ink, which is what real Amateur Drawings did to it (see `_marks`).
    """
    page = np.tile(np.linspace(140, 250, 64), (64, 1))
    page[10:54, 5:8] = 100
    page[10:54, 56:59] = 150
    return np.floor(page + 0.5).astype(np.uint8)


def test_binarising_lands_on_the_renderers_two_levels():
    binary = colour_only(render=replace(FOUR, size=64), greyscale=True, binarise=True)

    prepared = prepare_drawing(faint_drawing_on_shaded_paper(), binary)

    assert set(np.unique(prepared).tolist()) == {0, 255}


def test_a_faint_line_on_shaded_paper_survives_binarising():
    """Why the threshold is chosen per drawing: a fixed 128 would return this page blank."""
    binary = colour_only(render=replace(FOUR, size=64), greyscale=True, binarise=True)

    prepared = prepare_drawing(faint_drawing_on_shaded_paper(), binary)[:, :, 0]

    assert (prepared[10:54, 5:8] == 0).all() and (prepared[10:54, 56:59] == 0).all()
    assert (prepared[:, 20:50] == 255).all(), "the paper, light or shaded, is all background"


def test_steeply_shaded_paper_stays_paper_when_binarised():
    """The case that retired a global threshold: lighting must not become ink."""
    binary = colour_only(render=replace(FOUR, size=64), greyscale=True, binarise=True)

    prepared = prepare_drawing(drawing_on_steeply_shaded_paper(), binary)[:, :, 0]

    assert (prepared[10:54, 5:8] == 0).all() and (prepared[10:54, 56:59] == 0).all()
    assert (prepared[:, 12:52] == 255).all(), "shadowed paper and lit paper are both background"


def test_a_solid_dark_area_binarises_to_its_outline():
    """The local threshold's trade, pinned so a change of method shows up as a change here.

    A filled-in area is its own neighbourhood, so only its edge is darker than its surroundings.
    Quick, Draw! is outlines too, but this is a change to the drawing, and S6c2 measures its cost.
    """
    page = np.full((64, 64), 240, dtype=np.uint8)
    page[16:48, 16:48] = 30
    binary = colour_only(render=replace(FOUR, size=64), greyscale=True, binarise=True)

    prepared = prepare_drawing(page, binary)[:, :, 0]

    assert prepared[16, 32] == 0 and prepared[32, 16] == 0, "the edge is ink"
    assert prepared[32, 32] == 255, "the middle, far from any lighter pixel, is not"


def test_binarising_follows_the_renderers_levels_not_fixed_ones():
    inverted = replace(FOUR, size=64, ink=255, background=0)
    binary = colour_only(render=inverted, greyscale=True, binarise=True)

    prepared = prepare_drawing(faint_drawing_on_shaded_paper(), binary)[:, :, 0]

    assert (prepared[10:54, 5:8] == 255).all(), "the marks are ink, whatever level ink is"


def test_a_blank_page_binarises_to_blank_paper():
    binary = colour_only(render=FOUR, greyscale=True, binarise=True)

    prepared = prepare_drawing(np.full((9, 9), 37, dtype=np.uint8), binary)

    assert (prepared == 255).all()


def test_binarising_without_greyscale_is_refused():
    with pytest.raises(ValueError, match="needs greyscale"):
        colour_only(binarise=True, greyscale=False)


def test_a_box_crops_to_the_character_before_anything_else():
    image = np.full((10, 10), 255, dtype=np.uint8)
    image[2:6, 3:7] = 0  # the character
    image[9, 9] = 0  # something else on the page

    prepared = prepare_drawing(image, colour_only(render=FOUR), box=(3, 2, 4, 4))

    assert (prepared == 0).all()


def test_a_fractional_box_is_rounded_outwards_so_it_never_trims_the_character():
    image = np.full((10, 10), 255, dtype=np.uint8)
    image[2, 3] = image[5, 6] = 0  # opposite corners of the character

    prepared = prepare_drawing(image, colour_only(render=FOUR), box=(3.4, 2.6, 3.2, 2.8))

    assert prepared[0, 0, 0] == 0 and prepared[3, 3, 0] == 0


@pytest.mark.parametrize(
    ("box", "error", "message"),
    [
        ((-1.5, 0, 4, 4), ValueError, "outside"),
        ((0, 0, 11.5, 4), ValueError, "outside"),
        ((0, 8, 4, 3.5), ValueError, "outside"),
        ((10.5, 0, 0.4, 4), ValueError, "outside"),
        ((0, 0, 0, 4), ValueError, "no area"),
        ((0, 0, 4), ValueError, "4 values|not 3"),
        ((0, 0, 4, float("nan")), ValueError, "finite"),
        ((0, True, 4, 4), TypeError, "number"),
        ("0,0,4,4", TypeError, "x, y, width, height"),
    ],
)
def test_a_box_that_does_not_fit_the_drawing_is_refused(box, error, message):
    with pytest.raises(error, match=message):
        prepare_drawing(np.zeros((10, 10), dtype=np.uint8), box=box)


@pytest.mark.parametrize("box", [(-1.0, 0, 4, 4), (0, 0, 11.0, 4), (6.3, 6.3, 4.7, 4.7)])
def test_a_box_reaching_a_pixel_past_the_edge_is_clamped_not_refused(box):
    """1.89% of Amateur Drawings' boxes do this once rounded outwards, all by at most 1.00px."""
    image = np.zeros((10, 10), dtype=np.uint8)

    assert prepare_drawing(image, colour_only(render=FOUR), box=box).shape == (4, 4, 3)


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        (b"\x89PNG", TypeError, "does not decode"),
        ([[0, 255]], TypeError, "does not decode"),
        (np.zeros((4, 4), dtype=np.float32), TypeError, "uint8"),
        (np.zeros((4, 4, 4), dtype=np.uint8), ValueError, "flattened"),
        (np.zeros((4, 4, 2), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((0, 4), dtype=np.uint8), ValueError, "no pixels"),
    ],
)
def test_anything_but_a_decoded_drawing_is_refused(image, error, message):
    with pytest.raises(error, match=message):
        prepare_drawing(image)


def test_preparing_a_drawing_leaves_the_callers_array_alone():
    image = np.random.default_rng(1).integers(0, 256, size=(30, 50, 3), dtype=np.uint8)
    before = image.copy()

    prepare_drawing(image, colour_only(greyscale=True, binarise=True), box=(1, 1, 20, 20))

    assert np.array_equal(image, before)


def test_the_shipped_inference_settings_take_their_geometry_from_the_renderer():
    shipped = default_prepare_config()

    assert shipped.render == default_config()
    assert shipped.size == 224


def test_the_shipped_settings_keep_what_s6a_already_did():
    """Colour, no binarising: not a decision about either, just this slice not making one."""
    shipped = default_prepare_config()

    assert (shipped.greyscale, shipped.binarise) == (False, False)


# ---------------------------------------------------------------------------------------- config


def write_config(tmp_path: Path, inference: str, training: str | None = None) -> Path:
    shipped = preprocess.CONFIG_PATH.read_text(encoding="utf-8")
    training_block = shipped[shipped.index("training:") :] if training is None else training
    path = tmp_path / "preprocess.yaml"
    path.write_text(f"name: test\ninference:\n{inference}\n{training_block}", encoding="utf-8")
    return path


#: The two threshold settings, so a test's inference block is complete apart from what it varies.
WINDOW = "  binarise_window: 0.125\n  binarise_offset: 0.15"


def test_a_misspelt_setting_is_refused_rather_than_ignored(tmp_path):
    path = write_config(tmp_path, f"  greyscale: true\n  binarise: false\n{WINDOW}\n  binarize: true")

    with pytest.raises(ValueError, match="binarize, which nothing reads"):
        PrepareConfig.from_yaml(path)


def test_a_missing_setting_says_which_one(tmp_path):
    with pytest.raises(ValueError, match="does not set binarise"):
        PrepareConfig.from_yaml(write_config(tmp_path, "  greyscale: true"))


@pytest.mark.parametrize("value", ['"true"', "1", "null"])
def test_a_switch_that_is_not_a_boolean_is_refused(tmp_path, value):
    inference = f"  greyscale: {value}\n  binarise: false\n{WINDOW}"
    with pytest.raises(TypeError, match="true or false"):
        PrepareConfig.from_yaml(write_config(tmp_path, inference))


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        ({"binarise_window": 0.0}, ValueError, "fraction of `size`"),
        ({"binarise_window": 1.5}, ValueError, "fraction of `size`"),
        ({"binarise_offset": 1.0}, ValueError, "exclusive"),
        ({"binarise_offset": -0.1}, ValueError, "exclusive"),
        ({"binarise_window": True}, TypeError, "number"),
    ],
)
def test_a_threshold_setting_that_would_mislead_is_refused(override, error, message):
    with pytest.raises(error, match=message):
        colour_only(**override)


def test_the_shipped_file_loads_both_directions():
    assert PrepareConfig.from_yaml() == default_prepare_config()
    assert AugmentConfig.from_yaml() == default_augment_config()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"wobble_probability": 1.5}, "in 0-1"),
        ({"erasure_probability": True}, "number"),
        ({"erasure_patches": 0}, ">= 1"),
        ({"erasure_size": 0.0}, "fraction"),
        ({"fill_opacity_min": 0.9, "fill_opacity_max": 0.5}, "min <= max"),
        ({"lighting_gradient": 1.0}, "exclusive"),
        ({"paper_grain": -1.0}, "negative"),
        ({"wobble_wavelength": 0.0}, "positive"),
        ({"ink_colours": ()}, "non-empty"),
        ({"fill_colours": ((255, 0),)}, "three whole levels"),
        ({"paper_tints": ((256, 0, 0),)}, "three whole levels"),
    ],
)
def test_an_augmentation_setting_that_would_mislead_is_refused(override, message):
    with pytest.raises((TypeError, ValueError), match=message):
        replace(default_augment_config(), **override)


# -------------------------------------------------------------------------------------- training


def test_augmentation_is_deterministic_given_a_seed(canvas):
    first = augment_canvas(canvas, seed=7)
    second = augment_canvas(canvas, seed=7, config=AugmentConfig.from_yaml())

    assert np.array_equal(first, second)


def test_different_seeds_give_different_drawings(canvas):
    assert not np.array_equal(augment_canvas(canvas, seed=1), augment_canvas(canvas, seed=2))


@pytest.mark.parametrize("seed", range(5))
def test_an_augmented_canvas_is_something_inference_accepts(canvas, seed):
    """Same dtype, same geometry, three channels -- the constraint that keeps the two from drifting."""
    augmented = augment_canvas(canvas, seed=seed)

    assert augmented.shape == (*canvas.shape, 3)
    assert augmented.dtype == np.uint8
    assert prepare_drawing(augmented).shape == (224, 224, 3)


def test_with_every_operation_off_the_canvas_comes_back_as_it_was(canvas):
    """Pins the compositing arithmetic: coverage, paper and ink recombine to the original."""
    assert np.array_equal(augment_canvas(canvas, seed=3, config=QUIET), rgb(canvas))


def ink(image: np.ndarray) -> int:
    return int((image.min(axis=2) < 128).sum())


def test_wobble_moves_the_lines_without_adding_or_losing_much_ink(canvas):
    wobbled = augment_canvas(canvas, seed=3, config=replace(QUIET, wobble_probability=1.0))

    assert not np.array_equal(wobbled, rgb(canvas))
    assert abs(ink(wobbled) - ink(rgb(canvas))) < 0.15 * ink(rgb(canvas))


def test_erasure_only_ever_removes_ink(canvas):
    """Over several seeds, because one random patch can land on bare paper and remove nothing."""
    erasing = replace(QUIET, erasure_probability=1.0)
    counts = []
    for seed in range(8):
        erased = augment_canvas(canvas, seed=seed, config=erasing)
        assert (erased[canvas == 255] == 255).all(), "paper stays paper"
        counts.append(ink(erased))

    assert max(counts) <= ink(rgb(canvas))
    assert min(counts) < ink(rgb(canvas))


def test_coloured_ink_comes_from_the_palette(canvas):
    crayon = augment_canvas(canvas, seed=3, config=replace(QUIET, ink_colour_probability=1.0))

    marks = {tuple(pixel) for pixel in crayon[canvas == 0].tolist()}
    assert len(marks) == 1 and marks <= set(QUIET.ink_colours)


def test_fill_colours_inside_the_lines_and_nowhere_else(canvas):
    filled = augment_canvas(canvas, seed=3, config=replace(QUIET, fill_probability=1.0))

    changed = (filled != rgb(canvas)).any(axis=2)
    assert changed.any()
    assert not changed[:40, :].any() and not changed[:, :40].any(), "outside the square is untouched"
    assert not changed[canvas == 0].any(), "the lines keep their ink"


@pytest.mark.parametrize("seed", range(8))
def test_lighting_always_reaches_the_paper_and_never_clips(canvas, seed):
    """Seed 3 is the one that used to fire and change nothing: brightened paper clipped to 255."""
    lit = augment_canvas(canvas, seed=seed, config=replace(QUIET, lighting_probability=1.0))

    paper, marks = lit[canvas == 255], lit[canvas == 0]
    assert (paper < 255).any(), "the page is darker somewhere than a render's white"
    assert paper.max() <= 255 and marks.min() >= 0 and (marks > 0).any(), "flattened, not clipped"


def test_paper_texture_reaches_every_pixel_of_paper(canvas):
    textured = replace(QUIET, paper_texture=6.0, paper_grain=3.0)

    assert len(np.unique(augment_canvas(canvas, seed=3, config=textured)[canvas == 255])) > 10


def test_turning_one_operation_off_leaves_the_others_where_they_were(canvas):
    """One random stream per operation, so an ablation changes only what it ablates."""
    textured = replace(QUIET, paper_texture=6.0, paper_grain=3.0, lighting_probability=1.0)
    wobbled = replace(textured, wobble_probability=1.0)

    plain, moved = augment_canvas(canvas, 11, textured), augment_canvas(canvas, 11, wobbled)

    far_from_ink = np.ones(canvas.shape, dtype=bool)
    far_from_ink[30:226, 30:226] = False  # every stroke, plus the wobble's reach
    assert np.array_equal(plain[far_from_ink], moved[far_from_ink])


@pytest.mark.parametrize(
    ("canvas_value", "seed", "error"),
    [
        (np.zeros((8, 8, 3), dtype=np.uint8), 0, TypeError),
        (np.zeros((8, 8), dtype=np.float64), 0, TypeError),
        (np.zeros((8, 8), dtype=np.uint8), True, TypeError),
        (np.zeros((8, 8), dtype=np.uint8), 1.0, TypeError),
        (np.zeros((8, 8), dtype=np.uint8), -1, ValueError),
    ],
)
def test_augmentation_refuses_what_is_not_a_rendered_canvas_and_a_seed(canvas_value, seed, error):
    with pytest.raises(error):
        augment_canvas(canvas_value, seed)


def test_augmentation_leaves_the_callers_canvas_alone(canvas):
    before = canvas.copy()

    augment_canvas(canvas, seed=5, config=replace(default_augment_config(), wobble_probability=1.0))

    assert np.array_equal(canvas, before)


# ---------------------------------------------------------------------------------------- guards

#: The two functions this module exists to be the only home of.
ENTRY_POINTS = ("prepare_drawing", "augment_canvas")


def definitions(name: str) -> list[str]:
    found = []
    for path in owned_python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            defined = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
            assigned = isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            )
            if defined or assigned:
                found.append(path.relative_to(REPO).as_posix())
    return found


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_each_direction_has_exactly_one_entry_point(name):
    assert definitions(name) == ["src/primer/drawing/preprocess.py"]


def test_the_recogniser_prepares_drawings_with_this_modules_function():
    """Imported, not re-implemented: the encoder's `prepare_drawing` is this one.

    The torch-gated `test_siglip_encoder.py` checks the call itself; this half runs in CI. The
    training loader (S6b2) does not exist yet -- it adds the same assertion for `augment_canvas`.
    """
    assert model.prepare_drawing is preprocess.prepare_drawing


#: Everything `preprocess.py` may import. An allowlist rather than a list of image libraries, so a
#: new one cannot slip in under a name nobody thought to ban -- and CI, which installs no Pillow,
#: is what keeps the module honest.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "functools",
        "math",
        "numpy",
        "numpy.typing",
        "pathlib",
        "primer.drawing.render",
        "typing",
        "yaml",
    }
)


def imported_modules(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_the_module_imports_no_image_library():
    tree = ast.parse(Path(preprocess.__file__).read_text(encoding="utf-8"))

    assert imported_modules(tree) - ALLOWED_IMPORTS == set()


#: Libraries whose job is preprocessing or augmenting images. Any of them in the drawing path is a
#: second path arriving, whatever its function is called.
IMAGE_PROCESSING = (
    "PIL.ImageOps",
    "PIL.ImageEnhance",
    "PIL.ImageFilter",
    "albumentations",
    "cv2",
    "imgaug",
    "kornia",
    "scipy.ndimage",
    "skimage",
    "torchvision.transforms",
)

#: Pillow modes `.convert()` may target. Decoding and compositing stay at the call site by design
#: (plan S6b1: `model.flattened` reads `.mode` and `.info` before an array exists, so it cannot move
#: here without bringing Pillow with it), and they need exactly these two. Any other target --
#: `"L"`, `"1"`, `"P"` -- is greyscale, binarisation or quantisation, which is this module's job.
DECODE_MODES = frozenset({"RGB", "RGBA"})


def array_preprocessing(tree: ast.Module) -> set[str]:
    found = {
        module
        for module in imported_modules(tree)
        for banned in IMAGE_PROCESSING
        if module == banned or module.startswith(banned + ".")
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "PIL":
            found |= {f"PIL.{alias.name}" for alias in node.names if f"PIL.{alias.name}" in IMAGE_PROCESSING}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "convert"
            and node.args
            and not (isinstance(node.args[0], ast.Constant) and node.args[0].value in DECODE_MODES)
        ):
            found.add(f".convert({ast.unparse(node.args[0])})")
    return found


def in_the_drawing_path(path: Path, tree: ast.Module) -> bool:
    """The drawing area itself, and anything outside it that hands it a drawing.

    Other areas process their own images for their own models -- the illustration area resizes
    pictures for CLIP -- and that is not a recogniser input. Anything that imports `primer.drawing`
    can feed one, so it is in scope. Tests are not: their fixtures build inputs (a palette PNG, a
    transparent canvas) rather than prepare them for the recogniser.
    """
    relative = path.relative_to(REPO).as_posix()
    if relative.startswith("tests/"):
        return False
    if relative.startswith("src/primer/drawing/"):
        return True
    return any(module.startswith("primer.drawing") for module in imported_modules(tree))


def test_nothing_else_in_the_drawing_path_preprocesses_an_image():
    """The guard that catches a second path under another name -- KAN-87's other half."""
    offenders = {}
    for path in owned_python_files():
        if path.resolve() == Path(preprocess.__file__).resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if in_the_drawing_path(path, tree) and (found := array_preprocessing(tree)):
            offenders[path.relative_to(REPO).as_posix()] = found

    assert offenders == {}


def test_the_guard_would_see_a_greyscale_conversion():
    """The guard above passes on today's tree; this shows it is not passing because it is blind."""
    tree = ast.parse("from PIL import ImageOps\nimage = image.convert('L')\nimport cv2.typing\n")

    assert array_preprocessing(tree) == {"PIL.ImageOps", ".convert('L')", "cv2.typing"}
