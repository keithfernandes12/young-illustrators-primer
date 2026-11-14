"""Synthetic sketches for the illustration tests. No child data: every drawing here is generated."""

import io

import pytest
from PIL import Image, ImageDraw


def draw_house(size=(200, 120), ink=0, paper=255, mode="L") -> bytes:
    """A house on a cloud, the way a child draws it: dark lines on light paper, wider than tall.

    ``ink`` and ``paper`` are values for ``mode``: grey levels for "L", tuples for "RGB" or "RGBA".
    Greyscale drawings are saved as RGB, the way a phone or scanner delivers them; other modes are
    saved as they are, so an RGBA canvas export keeps its transparency.
    """
    image = Image.new(mode, size, paper)
    pen = ImageDraw.Draw(image)
    pen.rectangle((70, 50, 130, 100), outline=ink, width=4)
    pen.line([(60, 52), (100, 15), (140, 52)], fill=ink, width=4)
    pen.ellipse((20, 95, 180, 118), outline=ink, width=4)
    buffer = io.BytesIO()
    (image.convert("RGB") if mode == "L" else image).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def draw():
    """`draw_house` itself, for tests that vary the medium."""
    return draw_house


@pytest.fixture
def child_style_sketch() -> bytes:
    return draw_house()


@pytest.fixture
def scribble_style_sketch() -> bytes:
    """Already white-on-black, like the baseline's synthetic sketch."""
    return draw_house(ink=255, paper=0)
