"""Synthetic sketches and books for the illustration tests. No child data: every image here is generated."""

import io
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from primer.drawing.interface import Character
from primer.illustration.interface import Illustration


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


# --- Synthetic multi-page books, for the consistency score (KAN-105) -------------------------------
#
# Rendered pages and parent-approved references for a handful of books whose right answer is known by
# construction. They are what `IllustratorModel.consistency()` will be tested on once it calls the
# stage 6 scorer (KAN-103), and they assume nothing about that scorer: no embedder, no threshold. Each
# book records only what any correct scorer has to conclude, and `expect_pass` is None where the team
# has not yet decided what that is.
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


def render(look: Look, x: int = 128, y: int = 128, scale: float = 1.0, background: str = "plain") -> bytes:
    """One 256x256 PNG of ``look``. Deterministic: the same arguments give the same bytes."""
    image = Image.new("RGB", (256, 256), BACKGROUNDS.get(background, (255, 255, 255)))
    pen = ImageDraw.Draw(image)
    if background == "day":
        pen.ellipse((210, 10, 246, 46), fill=(255, 235, 120))  # a sun, so pages are not just recoloured
    if background == "grass":
        pen.rectangle((0, 200, 256, 256), fill=(90, 160, 70))
    r = int(40 * scale)
    box = (x - r, y - r, x + r, y + r)
    if look.round_body:
        pen.ellipse(box, fill=look.body)
        pen.polygon(
            [(x - r, y - r // 2), (x - r // 2, y - r - r // 2), (x - r // 4, y - r)], fill=look.accent
        )
        pen.polygon(
            [(x + r, y - r // 2), (x + r // 2, y - r - r // 2), (x + r // 4, y - r)], fill=look.accent
        )
    else:
        pen.rectangle(box, fill=look.body)
        pen.line([(x, y - r), (x, y - r - r // 2)], fill=look.accent, width=max(2, r // 8))
        pen.ellipse(
            (x - r // 5, y - r - r // 2 - r // 5, x + r // 5, y - r - r // 2 + r // 5), fill=look.accent
        )
    for dx in (-r // 3, r // 3):
        pen.ellipse(
            (x + dx - r // 6, y - r // 3 - r // 6, x + dx + r // 6, y - r // 3 + r // 6), fill="white"
        )
        pen.ellipse(
            (x + dx - r // 14, y - r // 3 - r // 14, x + dx + r // 14, y - r // 3 + r // 14), fill="black"
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
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
