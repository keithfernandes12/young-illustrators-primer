"""Loading a drawing from its reference, and turning it into a ControlNet-Scribble control image."""

import io

import pytest
from PIL import Image

from primer.illustration.sketch import load_drawing, prepare_sketch


def _pixels(image: Image.Image) -> bytes:
    return image.convert("L").tobytes()


def test_a_file_ref_loads_the_original_bytes(tmp_path, child_style_sketch):
    path = tmp_path / "upload.png"
    path.write_bytes(child_style_sketch)

    assert load_drawing(path.as_uri()) == child_style_sketch


def test_a_gs_ref_goes_through_the_gs_reader_without_touching_gcs(child_style_sketch):
    seen: list[str] = []

    def fake_gs(ref: str) -> bytes:
        seen.append(ref)
        return child_style_sketch

    assert load_drawing("gs://drawings/u1/d1.png", readers={"gs": fake_gs}) == child_style_sketch
    assert seen == ["gs://drawings/u1/d1.png"]


@pytest.mark.parametrize(
    "ref",
    [
        "",
        "fixture://smoke/cat-crayon.png",  # test-only by contract; resolves to nothing
        "https://example.com/drawing.png",  # never fetch a child's drawing from an unvetted host
        "drawings/d1.png",  # no scheme
        "file://otherhost/tmp/d1.png",  # file:// must be local
    ],
)
def test_refs_outside_the_contract_are_refused(ref):
    with pytest.raises(ValueError):
        load_drawing(ref)


def test_an_empty_upload_is_refused(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_drawing(path.as_uri())


def test_dark_lines_on_paper_become_white_strokes_on_black(child_style_sketch):
    control = prepare_sketch(child_style_sketch, size=64)
    pixels = _pixels(control)

    assert set(pixels) <= {0, 255}, "strokes are thresholded to pure black and white"
    assert pixels.count(0) > pixels.count(255), "the background is black; strokes are the minority"
    assert control.getpixel((0, 0)) == (0, 0, 0)


def test_a_drawing_that_is_already_white_on_black_is_not_inverted(child_style_sketch, scribble_style_sketch):
    """Both drawings carry the same strokes, so both must produce the same control image."""
    assert _pixels(prepare_sketch(scribble_style_sketch, 64)) == _pixels(
        prepare_sketch(child_style_sketch, 64)
    )


def test_the_control_image_is_square_rgb_at_the_generation_size(child_style_sketch):
    control = prepare_sketch(child_style_sketch, size=512)
    assert control.size == (512, 512)
    assert control.mode == "RGB"


def test_a_wide_drawing_is_padded_not_stretched(child_style_sketch):
    """200x120 in, so 40 rows of padding top and bottom before resizing; stretching leaves none."""
    control = prepare_sketch(child_style_sketch, size=200)
    top_band = control.crop((0, 0, 200, 38))
    assert set(_pixels(top_band)) == {0}, "the padding above the drawing stays empty"


def test_the_photo_orientation_is_honoured(child_style_sketch):
    """A phone photo stores rotation in EXIF; ignoring it conditions on a sideways drawing."""
    sideways = Image.open(io.BytesIO(child_style_sketch)).rotate(90, expand=True)  # stored 90° CCW
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation 6: rotate 90° clockwise to display upright
    buffer = io.BytesIO()
    sideways.save(buffer, format="PNG", exif=exif.tobytes())

    assert _pixels(prepare_sketch(buffer.getvalue(), 64)) == _pixels(prepare_sketch(child_style_sketch, 64))


def test_a_thin_line_on_a_large_photo_survives_the_downscale():
    """A phone photo is thousands of pixels wide, so a crayon line is thin relative to the shrink.

    Sampling every fourth pixel misses a 1 px line at an offset that is not a multiple of four.
    """
    image = Image.new("L", (2048, 2048), 255)
    image.paste(0, (1027, 200, 1028, 1800))  # one vertical 1 px line
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    control = prepare_sketch(buffer.getvalue(), size=512)
    column = control.convert("L").crop((255, 100, 259, 400)).tobytes()
    assert 255 in column, "the line vanished when the drawing was shrunk"


@pytest.mark.parametrize(
    "ink,paper,mode",
    [
        ((0, 0, 0, 255), (0, 0, 0, 0), "RGBA"),  # web canvas export: black strokes, transparent paper
        (160, 255, "L"),  # grey pencil on white
        ((255, 230, 0), (255, 255, 255), "RGB"),  # yellow crayon on white
        (30, 110, "L"),  # a dim phone photo: grey paper, darker ink
    ],
    ids=["transparent-canvas", "grey-pencil", "yellow-crayon", "dim-photo"],
)
def test_faint_or_unusual_drawings_still_produce_the_same_strokes(draw, ink, paper, mode, child_style_sketch):
    """Each of these is the house drawn in black ink; the control image must not depend on the medium."""
    drawn = draw(ink=ink, paper=paper, mode=mode)
    assert _pixels(prepare_sketch(drawn, 64)) == _pixels(prepare_sketch(child_style_sketch, 64))


@pytest.mark.parametrize("paper", [255, 0, 110], ids=["white", "black", "grey"])
def test_a_blank_page_is_refused_rather_than_generated_from_nothing(draw, paper):
    blank = draw(ink=paper, paper=paper)
    with pytest.raises(ValueError, match="no strokes"):
        prepare_sketch(blank)


def test_an_unreadable_file_ref_does_not_expose_the_file_name(tmp_path):
    """The file name is the child's: `zip-the-dragon-by-maya.png` must not reach a log via the error."""
    missing = tmp_path / "zip-the-dragon-by-maya.png"

    with pytest.raises(ValueError) as caught:
        load_drawing(missing.as_uri())

    chain = [caught.value, caught.value.__cause__, caught.value.__context__]
    text = " ".join(str(error) for error in chain if error is not None)
    assert "maya" not in text and "zip-the-dragon" not in text, text
    assert caught.value.__context__ is None, "the OS error names the file; it must not ride along"


def test_an_unreadable_gs_ref_does_not_expose_the_object_name():
    def failing_gs(ref: str) -> bytes:
        raise FileNotFoundError(f"No such object: {ref}")

    with pytest.raises(ValueError) as caught:
        load_drawing("gs://drawings/zip-the-dragon-by-maya.png", readers={"gs": failing_gs})
    assert "maya" not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_bytes_that_are_not_an_image_are_refused():
    with pytest.raises(ValueError, match="not a readable image"):
        prepare_sketch(b"definitely not a png")
