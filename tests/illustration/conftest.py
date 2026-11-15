"""Synthetic sketches and books for the illustration tests. No child data: every image here is generated."""

import io
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from primer.drawing.interface import Character
from primer.illustration.interface import Illustration


def draw_house(size=(200, 120), ink=0, paper=255, mode="L") -> bytes:
    """A house on a cloud, the way a child draws it: dark lines on light paper, wider than tall.

    ``ink`` and ``paper`` are values for ``mode``: grey levels for "L", tuples for "RGB" or "RGBA".
    Greyscale drawings are saved as RGB, the way a phone or scanner delivers them; other modes are
    saved as they are, so an RGBA canvas export keeps its transparency.
    """
    image = Image.new(mode, size, paper)
    for box in ((70, 50, 130, 54), (70, 96, 130, 100), (70, 54, 74, 96), (126, 54, 130, 96)):
        image.paste(ink, box)
    for y in range(15, 52):
        offset = (y - 15) * 40 // 37
        image.paste(ink, (100 - offset, y, 104 - offset, y + 1))
        image.paste(ink, (100 + offset, y, 104 + offset, y + 1))
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


# --- Synthetic multi-page books, for the consistency score (KAN-105) -------------------------------
#
# Rendered pages and parent-approved references for a handful of books whose right answer is known by
# construction. They are what `IllustratorModel.consistency()` will be tested on once it calls the
# stage 6 scorer (KAN-103), and they assume nothing about that scorer: no embedder, no threshold. Each
# book records only what any correct scorer has to conclude, and `expect_pass` is None where the team
# has not yet decided what that is.
#
# `expect_pass` is the verdict once a threshold exists. `ConsistencyScorer` (#24) gives no verdict
# until KAN-104 sets one, so a test that needs pass or fail builds the scorer with an explicit
# threshold, e.g. `ConsistencyScorer(FakeEmbedder(), threshold=...)`. With no references, `score()`
# returns 0.0; what `consistency()` and `build_book()` do then is the open KAN-105 question.
#
# Characters are storybook-style renders, not crayon: the reference a page is scored against is the
# approved `reference_image_ref`, never the child's crop.


@dataclass(frozen=True)
class Look:
    """How one synthetic character is drawn: a body, two eyes and something on top."""

    body: tuple[int, int, int]
    accent: tuple[int, int, int]
    round_body: bool


PIP = Look(body=(220, 40, 40), accent=(255, 150, 0), round_body=True)  # red, round, orange ears
ZOG = Look(body=(40, 80, 220), accent=(250, 220, 0), round_body=False)  # blue, square, yellow antenna

#: (centre x, centre y, scale, background) per page: the same character, moved and resized the way a
#: real book poses it, so a scorer cannot pass a book by matching pixels.
POSES = [(128, 150, 1.0, "day"), (80, 160, 0.8, "grass"), (176, 140, 1.2, "night"), (128, 170, 0.7, "day")]
BACKGROUNDS = {"day": (170, 215, 255), "grass": (150, 210, 120), "night": (30, 30, 80)}


# Pages are filled shapes painted with numpy masks, not strokes: `primer.drawing.render` is the only
# strokes-to-pixels renderer, and `tests/drawing/test_render.py` forbids a rasteriser such as
# `ImageDraw` anywhere else in the repository.
_ROWS, _COLS = np.mgrid[0:256, 0:256]


def _disc(cx: int, cy: int, radius: int) -> np.ndarray:
    return (_COLS - cx) ** 2 + (_ROWS - cy) ** 2 <= radius**2


def _box(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return (_COLS >= x0) & (_COLS <= x1) & (_ROWS >= y0) & (_ROWS <= y1)


def _triangle(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> np.ndarray:
    """Points on the same side of all three edges."""

    def side(p: tuple[int, int], q: tuple[int, int]) -> np.ndarray:
        return (q[0] - p[0]) * (_ROWS - p[1]) - (q[1] - p[1]) * (_COLS - p[0])

    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))


def render(look: Look, x: int = 128, y: int = 128, scale: float = 1.0, background: str = "plain") -> bytes:
    """One 256x256 PNG of ``look``. Deterministic: the same arguments give the same bytes."""
    canvas = np.empty((256, 256, 3), dtype=np.uint8)
    canvas[:] = BACKGROUNDS.get(background, (255, 255, 255))
    if background == "day":
        canvas[_disc(228, 28, 18)] = (255, 235, 120)  # a sun, so pages are not just recoloured
    if background == "grass":
        canvas[_box(0, 200, 255, 255)] = (90, 160, 70)
    r = int(40 * scale)
    if look.round_body:
        canvas[_disc(x, y, r)] = look.body
        canvas[_triangle((x - r, y - r // 2), (x - r // 2, y - r - r // 2), (x - r // 4, y - r))] = (
            look.accent
        )
        canvas[_triangle((x + r, y - r // 2), (x + r // 2, y - r - r // 2), (x + r // 4, y - r))] = (
            look.accent
        )
    else:
        canvas[_box(x - r, y - r, x + r, y + r)] = look.body
        half = max(1, r // 16)
        canvas[_box(x - half, y - r - r // 2, x + half, y - r)] = look.accent  # antenna stalk
        canvas[_disc(x, y - r - r // 2, r // 5)] = look.accent
    for dx in (-r // 3, r // 3):
        canvas[_disc(x + dx, y - r // 3, r // 6)] = (255, 255, 255)
        canvas[_disc(x + dx, y - r // 3, r // 14)] = (0, 0, 0)
    buffer = io.BytesIO()
    Image.fromarray(canvas, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class SyntheticBook:
    """A book with a known answer, as `IllustratorModel.consistency()` receives it."""

    characters: tuple[Character, ...]
    illustrations: list[Illustration]
    #: Which look each page actually shows, for checking the fixture itself.
    page_looks: tuple[Look, ...]
    #: What a correct score must conclude. None: the team has not decided (see the KAN-105 question).
    expect_pass: bool | None
    #: The page a scorer should name as weakest, when exactly one page is wrong.
    weakest_page: int | None = None


def _write(directory: Path, data: bytes) -> str:
    # A random name, as the illustrator itself does: nothing is derived from the drawing.
    path = directory / f"{uuid.uuid4().hex}.png"
    path.write_bytes(data)
    return path.as_uri()


def _book(
    directory: Path,
    references: list[tuple[str, Look, bool | None]],
    pages: list[Look],
    expect_pass: bool | None,
    weakest_page: int | None = None,
    same_pose: bool = False,
) -> SyntheticBook:
    characters = []
    for label, look, approved in references:
        character = Character(
            label=label, confidence=0.9, embedding=(1.0, 0.0) if look is PIP else (0.0, 1.0)
        )
        if approved is None:  # recognised, but nothing has rendered a reference for it
            characters.append(character)
            continue
        ref = _write(directory, render(look))
        characters.append(character.approve(ref) if approved else replace(character, reference_image_ref=ref))
    # Each page is conditioned on the character it shows, as IP-Adapter would record it.
    identity = {look: c.identity_key for (_, look, _), c in zip(references, characters, strict=True)}
    illustrations = []
    for index, look in enumerate(pages):
        x, y, scale, background = POSES[0] if same_pose else POSES[index % len(POSES)]
        illustrations.append(
            Illustration(
                page_index=index,
                image_ref=_write(directory, render(look, x, y, scale, background)),
                alt_text=f"page {index}",
                subject="creature",
                conditioned_on_sketch=True,
                identity=identity.get(look),
            )
        )
    return SyntheticBook(tuple(characters), illustrations, tuple(pages), expect_pass, weakest_page)


@pytest.fixture
def synthetic_books(tmp_path) -> dict[str, SyntheticBook]:
    """Six books, each one case `consistency()` has to get right. All images live under tmp_path."""
    return {
        # Pip on every page, in four poses: the book that must pass.
        "consistent": _book(tmp_path, [("pip", PIP, True)], [PIP] * 4, expect_pass=True),
        # Four identical pages of the wrong character. Page-to-page cohesion is perfect, so a score
        # that only compares pages with each other passes it; one that compares against the
        # reference does not. KAN-103's first "done when".
        "wrong_character": _book(
            tmp_path, [("pip", PIP, True)], [ZOG] * 4, expect_pass=False, same_pose=True
        ),
        # Pip, except page 2. The page to regenerate first is page 2.
        "one_page_drifts": _book(
            tmp_path, [("pip", PIP, True)], [PIP, PIP, ZOG, PIP], expect_pass=False, weakest_page=2
        ),
        # Two approved characters, alternating. Each page matches its closest reference, so it passes.
        "two_characters": _book(
            tmp_path, [("pip", PIP, True), ("zog", ZOG, True)], [PIP, ZOG, PIP, ZOG], expect_pass=True
        ),
        # A reference exists but no parent approved it. Undecided: see the KAN-105 question.
        "unapproved_reference": _book(tmp_path, [("pip", PIP, False)], [PIP] * 4, expect_pass=None),
        # A recognised character with no reference at all -- every book today, since nothing renders
        # or approves one yet. Undecided, like the one above.
        "no_reference": _book(tmp_path, [("pip", PIP, None)], [PIP] * 4, expect_pass=None),
    }


@pytest.fixture
def looks() -> tuple[Look, Look]:
    """The two characters the books are drawn with: Pip, then Zog."""
    return PIP, ZOG


@pytest.fixture
def render_look():
    """`render` itself, for tests of the fixtures."""
    return render
