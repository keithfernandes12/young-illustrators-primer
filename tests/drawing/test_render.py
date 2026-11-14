"""The renderer (KAN-87, plan slice S6b).

Three kinds of test here. Most pin the pixels: a known stroke list, a known config, the inked set
written out by hand, because "deterministic" is worth nothing unless someone wrote down what the
output is. The PNG tests walk the file's chunks and recompute every CRC, so that CI -- which does
not install Pillow -- still catches a malformed encode rather than only a wrong pixel. The last two
guard the reason the ticket exists, that there is exactly one renderer, which no amount of testing
`render_strokes` itself can establish.

No bucket, no network, no image library: `numpy` is a declared dependency, so all of this runs in
CI. One test needs a real PNG decoder and skips where Pillow is absent.
"""

from __future__ import annotations

import ast
import io
import struct
import zlib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import yaml

from primer.drawing.render import (
    CONFIG_PATH,
    RenderConfig,
    default_config,
    render_strokes,
    render_strokes_png,
    to_png,
)

REPO = Path(__file__).resolve().parents[2]

#: Directories that are not this repository's own source, however many `.py` files they hold. The
#: two guards below scan everything else -- a second renderer is as likely to appear in a DAG, a
#: dashboard or a one-off script as in `src/`.
NOT_OURS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)

#: A 10x10 canvas over a 5x5 source grid, 1px strokes. Small enough to assert every pixel, and the
#: scale is 7/4, so rounding is actually exercised rather than coming out whole.
TINY = RenderConfig(
    name="tiny", size=10, line_width=1, padding=1, background=255, ink=0, source_grid=5
)

#: Where each source coordinate lands on TINY's canvas: 0 -> padding, 4 -> size - padding - 1.
TINY_COLUMNS = (1, 3, 5, 6, 8)


@pytest.fixture
def tiny() -> RenderConfig:
    return TINY


def inked(image: np.ndarray, config: RenderConfig) -> set[tuple[int, int]]:
    rows, columns = np.where(image == config.ink)
    return set(zip(rows.tolist(), columns.tolist(), strict=True))


def test_a_horizontal_stroke_lands_on_the_pixels_it_should(tiny):
    image = render_strokes([[[0, 4], [0, 0]]], tiny)

    assert image.shape == (10, 10)
    assert image.dtype == np.uint8
    assert inked(image, tiny) == {(1, column) for column in range(1, 9)}


def test_a_vertical_stroke_lands_on_the_pixels_it_should(tiny):
    image = render_strokes([[[2, 2], [0, 4]]], tiny)

    assert inked(image, tiny) == {(row, 5) for row in range(1, 9)}


def test_a_45_degree_stroke_lands_on_the_pixels_it_should(tiny):
    image = render_strokes([[[0, 4], [0, 4]]], tiny)

    assert inked(image, tiny) == {(step, step) for step in range(1, 9)}


def test_a_shallow_diagonal_lands_on_the_pixels_it_should(tiny):
    """The interior of a segment, pixel by pixel -- not merely its endpoints and its row count."""
    image = render_strokes([[[0, 4], [0, 2]]], tiny)

    assert inked(image, tiny) == {(1, 1), (2, 2), (2, 3), (3, 4), (3, 5), (4, 6), (4, 7), (5, 8)}


def test_a_steep_diagonal_lands_on_the_pixels_it_should(tiny):
    """|dy| > |dx|: the other half of Bresenham, which no axis-aligned or 45-degree case reaches."""
    image = render_strokes([[[0, 2], [0, 4]]], tiny)

    assert inked(image, tiny) == {(1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (6, 4), (7, 4), (8, 5)}


def test_a_downhill_diagonal_lands_on_the_pixels_it_should(tiny):
    """Negative slope: every other stroke here runs left to right and top to bottom."""
    image = render_strokes([[[0, 4], [2, 0]]], tiny)

    assert inked(image, tiny) == {(5, 1), (4, 2), (4, 3), (3, 4), (3, 5), (2, 6), (2, 7), (1, 8)}


def test_the_tie_break_between_two_equally_close_pixels_is_pinned(tiny):
    """Where a line passes exactly between two pixels, which one takes the ink is a decision.

    This stroke ties. Loosening `_line`'s `>=` to `>` moves it, and thousands of other segments
    with it; with only the endpoints and a connectivity check asserted, that edit is invisible.
    """
    image = render_strokes([[[2, 3], [0, 1]]], tiny)

    assert inked(image, tiny) == {(1, 5), (2, 6), (3, 6)}


def test_the_other_tie_break_is_pinned_too(tiny):
    """`_line` breaks two ties, and pinning one of them is not pinning the other.

    A tie in the second branch needs an even `dx`, and every other stroke in this file has `dx` of
    7 or 0 -- so loosening `doubled <= dx` to `doubled < dx` moved 36 of this grid's 625
    coordinate pairs while the whole suite stayed green.
    """
    image = render_strokes([[[0, 2], [0, 1]]], tiny)

    assert inked(image, tiny) == {(1, 1), (2, 2), (2, 3), (3, 4), (3, 5)}


def test_a_coordinate_exactly_between_two_pixels_rounds_up(tiny):
    """Half up, not to even. At this scale coordinate 2 falls 4.5 pixels past the padding edge."""
    half = RenderConfig(**{**vars(tiny), "size": 12})  # content 10, so the scale is 9/4

    assert inked(render_strokes([[[2], [0]]], half), half) == {(1, 6)}, "4.5 rounds to 5, not 4"
    assert inked(render_strokes([[[0], [0]]], half), half) == {(1, 1)}
    assert inked(render_strokes([[[4], [0]]], half), half) == {(1, 10)}


def test_the_source_grid_maps_onto_the_whole_content_square(tiny):
    corners = render_strokes([[[0, 4, 4, 0, 0], [0, 0, 4, 4, 0]]], tiny)

    drawn = inked(corners, tiny)
    border = {(row, column) for row in (1, 8) for column in range(1, 9)}
    border |= {(row, column) for column in (1, 8) for row in range(1, 9)}
    assert drawn == border, "the four corners of the grid are the four corners of the content square"


def test_each_source_coordinate_rounds_to_a_known_column(tiny):
    for coordinate, column in enumerate(TINY_COLUMNS):
        dot = render_strokes([[[coordinate], [0]]], tiny)
        assert inked(dot, tiny) == {(1, column)}


def test_a_single_point_stroke_is_a_dot_rather_than_nothing(tiny):
    assert inked(render_strokes([[[2], [2]]], tiny), tiny) == {(5, 5)}


def test_strokes_are_drawn_in_the_same_image_not_one_each(tiny):
    both = render_strokes([[[0, 4], [0, 0]], [[2, 2], [0, 4]]], tiny)

    assert inked(both, tiny) == {(1, column) for column in range(1, 9)} | {
        (row, 5) for row in range(1, 9)
    }


def test_a_stroke_is_stamped_with_a_disc_rather_than_a_square(tiny):
    """The shape `_stamp` argues for, asserted: at radius 1 a dot is a plus, not a 3x3 block."""
    thick = RenderConfig(**{**vars(tiny), "line_width": 3})

    dot = render_strokes([[[2], [2]]], thick)

    assert inked(dot, thick) == {(5, 5), (4, 5), (6, 5), (5, 4), (5, 6)}
    assert inked(render_strokes([[[2], [2]]], tiny), tiny) == {(5, 5)}, "1px is one pixel"


def test_line_width_thickens_the_stroke_along_its_whole_length(tiny):
    thick = RenderConfig(**{**vars(tiny), "line_width": 3})

    drawn = inked(render_strokes([[[0, 4], [0, 0]]], thick), thick)

    spine = {(1, column) for column in range(1, 9)}
    assert drawn == spine | {(0, column) for column in range(1, 9)} | {
        (2, column) for column in range(1, 9)
    } | {(1, 0), (1, 9)}


def test_the_same_strokes_and_config_give_the_same_pixels_every_time(tiny):
    strokes = [[[0, 2, 4], [0, 3, 1]], [[4, 1], [4, 2]]]

    first = render_strokes(strokes, tiny)
    second = render_strokes(strokes, tiny)

    assert np.array_equal(first, second)
    assert render_strokes_png(strokes, tiny) == render_strokes_png(strokes, tiny)


def test_the_png_helper_encodes_the_array_the_renderer_made(tiny):
    """One code path, not two: the bytes are the pixels, with nothing in between."""
    strokes = [[[0, 2, 4], [0, 3, 1]]]

    assert render_strokes_png(strokes, tiny) == to_png(render_strokes(strokes, tiny))


def test_numpy_arrays_render_the_same_as_lists(tiny):
    """A loader holding strokes as arrays is the likely caller; it must not get other pixels."""
    strokes = [[[0, 2, 4], [0, 3, 1]]]

    as_arrays = [np.array(stroke, dtype=np.int16) for stroke in strokes]

    assert np.array_equal(render_strokes(as_arrays, tiny), render_strokes(strokes, tiny))


def test_the_shipped_config_renders_at_the_checkpoint_resolution():
    config = default_config()

    image = render_strokes([[[0, 255], [0, 255]]], config)

    assert (config.size, config.line_width, config.padding) == (224, 3, 8)
    assert (config.background, config.ink, config.source_grid) == (255, 0, 256)
    assert image.shape == (224, 224)
    assert image[8, 8] == config.ink and image[215, 215] == config.ink


def test_the_shipped_config_is_the_file_on_disk():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = default_config()

    assert raw == {field.name: getattr(config, field.name) for field in fields(RenderConfig)}


def test_the_config_file_is_read_once_per_run_not_once_per_drawing(monkeypatch):
    reads: list[Path] = []
    real = RenderConfig.from_yaml

    def counted(path: Path = CONFIG_PATH) -> RenderConfig:
        reads.append(path)
        return real(path)

    try:
        monkeypatch.setattr(RenderConfig, "from_yaml", staticmethod(counted))
        default_config.cache_clear()
        render_strokes([[[0], [0]]])
        render_strokes([[[1], [1]]])
        assert reads == [CONFIG_PATH]
    finally:
        default_config.cache_clear()


def test_a_config_missing_a_setting_says_which_one(tmp_path):
    partial = tmp_path / "drawing_render.yaml"
    partial.write_text("name: partial\nsize: 10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not set line_width, padding"):
        RenderConfig.from_yaml(partial)


@pytest.mark.parametrize(("text", "message"), [("", "an empty document"), ("- 1\n- 2\n", "list")])
def test_a_config_that_is_not_a_mapping_says_so(tmp_path, text, message):
    """A commented-out or empty file parses to None, and iterating that names NoneType, not it."""
    odd = tmp_path / "drawing_render.yaml"
    odd.write_text(text, encoding="utf-8")

    with pytest.raises(TypeError, match=message):
        RenderConfig.from_yaml(odd)


@pytest.mark.parametrize(
    ("strokes", "message"),
    [
        ("[[[0],[0]]]", "json.loads"),
        ([["12", "34"]], "x run is text"),
        ([[[0], "34"]], "y run is text"),
        ([{"x": [0], "y": [0]}], "is a mapping"),
        ([[["a"], ["b"]]], "could not convert string to float"),
        ([[[0, None], [0, 1]]], "not sequences of numbers"),
        (None, "null for image sources"),
        ([], "at least one stroke"),
        ([5], "not a pair of runs"),
        ([[0, 4], [0, 0]], "not sequences of numbers"),
        ([[[0], [0], [0]]], "timings"),
        ([[[0, 1], [0]]], "x values"),
        ([[[], []]], "no points"),
        ([[[0], [5]]], "source grid"),
        ([[[0], [-1]]], "source grid"),
        ([[[0], [float("nan")]]], "source grid"),
        ([[[0], [float("inf")]]], "source grid"),
    ],
)
def test_a_malformed_drawing_is_refused(strokes, message, tiny):
    with pytest.raises((ValueError, TypeError), match=message):
        render_strokes(strokes, tiny)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"line_width": 2}, "odd"),
        ({"line_width": 5}, "padding of at least"),
        ({"ink": 255}, "every render would be blank"),
        ({"padding": 5}, "at least 2"),
        ({"size": 1}, "at least 2"),
        ({"padding": -1}, "cannot be negative"),
        ({"background": 300}, "0-255"),
        ({"source_grid": 1}, "source_grid"),
        ({"size": 224.0}, "whole number"),
        ({"padding": True}, "whole number"),
    ],
)
def test_a_config_that_would_render_something_misleading_is_refused(override, message):
    with pytest.raises((ValueError, TypeError), match=message):
        RenderConfig(**{**vars(TINY), **override})


def chunks(encoded: bytes) -> list[tuple[bytes, bytes]]:
    """Every chunk of a PNG, with its length and CRC checked on the way past."""
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n"), "PNG signature"
    found: list[tuple[bytes, bytes]] = []
    offset = 8
    while offset < len(encoded):
        (length,) = struct.unpack(">I", encoded[offset : offset + 4])
        tag = encoded[offset + 4 : offset + 8]
        payload = encoded[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", encoded[offset + 8 + length : offset + 12 + length])
        assert len(payload) == length, f"{tag!r} is shorter than its declared length"
        assert crc == zlib.crc32(tag + payload), f"{tag!r} carries a wrong CRC"
        found.append((tag, payload))
        offset += 12 + length
    assert offset == len(encoded), "the chunks account for every byte of the file"
    assert found[-1] == (b"IEND", b""), "IEND comes last and carries nothing"
    return found


def test_png_is_well_formed_and_carries_the_pixels_it_was_given(tiny):
    image = render_strokes([[[0, 4], [0, 0]]], tiny)

    found = chunks(to_png(image))

    assert [tag for tag, _ in found] == [b"IHDR", b"IDAT", b"IEND"]
    width, height, depth, colour, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", found[0][1]
    )
    assert (width, height) == (10, 10)
    assert (depth, colour) == (8, 0), "8 bits per sample, greyscale"
    assert (compression, filtering, interlace) == (0, 0, 0), "deflate, standard filters, no interlace"
    body = zlib.decompress(found[1][1])
    rows = [body[row * 11 : (row + 1) * 11] for row in range(10)]
    assert len(body) == 10 * 11 and all(row[0] == 0 for row in rows), "filter byte 0 on every row"
    pixels = np.frombuffer(b"".join(row[1:] for row in rows), dtype=np.uint8)
    assert np.array_equal(pixels.reshape(10, 10), image)


def test_png_records_width_and_height_the_right_way_round():
    """Every other image here is square, so swapping the two in the IHDR pack would survive."""
    found = chunks(to_png(np.zeros((3, 7), dtype=np.uint8)))

    assert struct.unpack(">II", found[0][1][:8]) == (7, 3), "IHDR is width then height"


def test_a_corrupted_png_does_not_pass_the_chunk_check(tiny):
    """The check above is only worth having if it fails on a file a decoder would reject."""
    encoded = bytearray(to_png(render_strokes([[[0, 4], [0, 0]]], tiny)))
    encoded[29] ^= 0xFF  # a byte of IHDR's CRC

    with pytest.raises(AssertionError, match="wrong CRC"):
        chunks(bytes(encoded))


def test_png_is_a_png_that_a_real_decoder_reads(tiny):
    """Pillow is not installed by CI, so this skips there; the chunk checks above are not gated."""
    image_module = pytest.importorskip("PIL.Image", reason="needs Pillow, which CI does not install")

    image = render_strokes([[[0, 2, 4], [0, 3, 1]]], tiny)

    decoded = image_module.open(io.BytesIO(to_png(image)))

    assert decoded.mode == "L" and decoded.size == (10, 10)
    assert np.array_equal(np.asarray(decoded), image)


def test_png_refuses_an_array_it_cannot_encode():
    with pytest.raises(ValueError, match="2-D greyscale"):
        to_png(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        to_png(np.zeros((4, 4), dtype=np.float32))
    for shape in ((0, 0), (0, 5), (5, 0)):
        # Zero in either IHDR field is invalid PNG: well-formed chunks that no decoder will open.
        with pytest.raises(ValueError, match="at least one pixel"):
            to_png(np.zeros(shape, dtype=np.uint8))


def owned_python_files() -> list[Path]:
    """Every Python file this repository owns, pruning as it walks.

    Pruned rather than filtered afterwards because walking into `.venv` costs more than parsing
    everything that is kept. A directory with its own `.git` is pruned too: an agent worktree
    under `.claude/worktrees` holds a second copy of this repository, which would report every
    module twice on a developer's machine while CI, which has no worktrees, stayed green.
    """
    found = list(REPO.glob("*.py"))
    unvisited = [path for path in REPO.iterdir() if path.is_dir()]
    while unvisited:
        directory = unvisited.pop()
        if directory.name in NOT_OURS or (directory / ".git").exists():
            continue
        found.extend(directory.glob("*.py"))
        unvisited.extend(child for child in directory.iterdir() if child.is_dir())
    return found


def modules() -> list[tuple[Path, ast.Module]]:
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in owned_python_files()]


def test_render_strokes_is_defined_exactly_once():
    """The ticket in one assertion: a second renderer is the failure KAN-87 exists to prevent."""
    defined = [
        path
        for path, tree in modules()
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == "render_strokes"
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "render_strokes"
                for target in node.targets
            )
        )
    ]

    # Repo-relative, not basenames: when this fails it has to say *where* the second one is.
    assert [path.relative_to(REPO).as_posix() for path in defined] == [
        "src/primer/drawing/render.py"
    ]


#: Modules that exist to turn geometry into pixels. `PIL.Image` is not one of them -- opening and
#: resizing an image is what the recogniser does with a drawing someone else rendered.
RASTERISERS = frozenset({"cv2", "PIL.ImageDraw", "skimage.draw", "aggdraw"})


def drawing_names(tree: ast.Module) -> set[str]:
    """Names in one module that only a rasteriser reaches for."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "ImageDraw":
            found.add("ImageDraw")
        elif isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names if alias.name in RASTERISERS}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found |= {
                f"{node.module}.{alias.name}"
                for alias in node.names
                if f"{node.module}.{alias.name}" in RASTERISERS or node.module in RASTERISERS
            }
    return found


def test_nothing_in_the_repository_draws_with_an_image_library():
    """An `ImageDraw` anywhere is the second renderer arriving, whatever the function is called."""
    drawers = {
        path.relative_to(REPO).as_posix(): names
        for path, tree in modules()
        if (names := drawing_names(tree))
    }

    assert drawers == {}
