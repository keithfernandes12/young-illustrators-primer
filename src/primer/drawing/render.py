"""The one renderer: Quick, Draw! strokes in, pixels out (KAN-87, plan slice S6b).

``render_strokes`` is the only place in this repository where strokes become pixels. That is the
whole of the ticket, not a style preference: the training loader and anything that scores a drawing
have to agree on the image, and two renderers that drift make every measured ``drawing_top5_acc`` a
comparison between a model and a rasteriser. So do not add a second one -- no ``ImageDraw`` loop in
a data loader, no quick redraw in a test fixture, no "just for the demo" copy.
``tests/drawing/test_render.py`` asserts this rather than trusting it.

Every setting that changes a pixel lives in `configs/drawing_render.yaml`, read into
`RenderConfig`, never as a default at a call site. Same strokes and same config give the same array
on any machine: one float multiply scales each coordinate and everything after it -- Bresenham, the
stamp, the PNG -- is integer and exact. IEEE-754 multiply and floor are correctly rounded, so the
scaling is reproducible too, but it is float arithmetic and worth knowing as such.

No image library is involved, by choice. The pixels therefore do not depend on which Pillow or
OpenCV happens to be installed, and -- since `numpy` is a declared dependency and Pillow is not
installed by CI at all -- the whole strokes-to-bytes path is covered by CI rather than skipped
there. The one test that needs a real PNG decoder is skipped wherever Pillow is absent, so the
chunk and CRC checks beside it are written against `zlib` instead.

Strokes may be nested numpy arrays as well as lists: the type alias below names the shape, not the
container, and every run is materialised before it is used.

**Licence and attribution.** This file never sees a record, only coordinates, so it cannot drop the
``license`` and ``attribution`` columns that `AGENTS.md` requires on every processed record. The
slice that maps records to images (S6b2) carries them through; rendering a stroke list is not where
that obligation is discharged.
"""

from __future__ import annotations

import itertools
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_PATH = CONFIG_DIR / "drawing_render.yaml"

#: One stroke as Quick, Draw! stores it: ``[[x, x, ...], [y, y, ...]]``, two equal-length runs.
#:
#: A numpy array is admitted alongside a sequence because `ndarray` is not a `Sequence`, and the
#: loader S6b2 builds -- the primary caller -- holds its strokes as arrays. Without the union its
#: one honest call site is a mypy error, and the usual answer to that is a `# type: ignore`.
Stroke = Sequence[Sequence[float]] | npt.NDArray[Any]
Strokes = Sequence[Stroke] | npt.NDArray[Any]

Image = npt.NDArray[np.uint8]

#: Fields that are a whole number of pixels or levels, checked as such before anything else.
WHOLE_NUMBERS = ("size", "line_width", "padding", "background", "ink", "source_grid")


@dataclass(frozen=True)
class RenderConfig:
    """The rasterisation settings, read from `configs/drawing_render.yaml`.

    Validated on construction rather than at the first render: every one of these can be wrong in a
    way that produces a plausible blank or clipped image instead of an error, and a loader finds
    out a few hundred thousand records later.
    """

    name: str
    size: int
    line_width: int
    padding: int
    background: int
    ink: int
    source_grid: int

    def __post_init__(self) -> None:
        for name in WHOLE_NUMBERS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                # YAML reads `true` as a boolean and `224.0` as a float. Both would construct, both
                # would change every pixel, and the float would fail much later inside numpy.
                raise TypeError(f"{name} must be a whole number, not {value!r}")
        if self.size < 2:
            raise ValueError(f"size must be at least 2 pixels, not {self.size}")
        if self.padding < 0:
            raise ValueError(f"padding cannot be negative, not {self.padding}")
        if self.content_size < 2:
            raise ValueError(
                f"padding {self.padding} leaves {self.content_size} pixels of a {self.size}px "
                "canvas to draw on; it must leave at least 2"
            )
        if self.line_width < 1 or self.line_width % 2 == 0:
            raise ValueError(
                f"line_width must be a positive odd number, not {self.line_width}: a stroke is "
                "centred on its line, so its width is 2 * radius + 1"
            )
        if self.radius > self.padding:
            # Without this the stamp would index outside the canvas. Clipping instead would draw a
            # stroke that silently flattens against the border, which looks like a drawing.
            raise ValueError(
                f"line_width {self.line_width} needs padding of at least {self.radius}, not "
                f"{self.padding}, or a stroke on the edge of the grid falls off the canvas"
            )
        for name in ("background", "ink"):
            value = getattr(self, name)
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be a greyscale level in 0-255, not {value}")
        if self.ink == self.background:
            raise ValueError(f"ink and background are both {self.ink}; every render would be blank")
        if self.source_grid < 2:
            raise ValueError(f"source_grid must be at least 2, not {self.source_grid}")

    @property
    def content_size(self) -> int:
        """The square, in pixels, that strokes are drawn inside."""
        return self.size - 2 * self.padding

    @property
    def radius(self) -> int:
        """Half a stroke's width, in pixels. ``line_width`` is odd, so this is exact."""
        return self.line_width // 2

    @property
    def scale(self) -> float:
        """Source-grid units to pixels. Pins 0 to ``padding`` and the last column to the last."""
        return (self.content_size - 1) / (self.source_grid - 1)

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> RenderConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            # An empty or commented-out file parses to None, and iterating that below would raise
            # a TypeError naming NoneType rather than the file.
            kind = "an empty document" if raw is None else type(raw).__name__
            raise TypeError(f"{path.name} should be a mapping of settings, not {kind}")
        missing = [f.name for f in fields(cls) if f.name not in raw]
        if missing:
            raise ValueError(f"{path.name} does not set {', '.join(missing)}")
        return cls(**{f.name: raw[f.name] for f in fields(cls)})


@lru_cache(maxsize=1)
def default_config() -> RenderConfig:
    """The shipped settings, parsed once.

    Cached because the training loader calls `render_strokes` once per record, and re-reading and
    re-parsing a YAML file per drawing is a real cost for a value that cannot change within a run.
    The config is frozen, so sharing one instance is safe.
    """
    return RenderConfig.from_yaml()


@lru_cache(maxsize=8)
def _stamp(radius: int) -> npt.NDArray[np.int64]:
    """Offsets of the disc that gives a stroke its width, as ``(dy, dx)`` pairs.

    A disc rather than a square so that stroke weight does not depend on direction. Measured at the
    shipped ``line_width: 3`` it does not manage that: ``radius`` is 1, so the "disc" is a five-cell
    plus, and a horizontal stroke lays down 3.0 pixels of ink per pixel of length against a
    diagonal's 2.1, where a 3x3 square stamp would give 3.5. The plus is 29% thin where the square
    is 18% thick; neither is even.

    It is kept because the gap closes as the radius grows and a square's does not, and because a
    thin diagonal is the safer error on a sketch. That is a choice, not a result: S6c is the first
    slice that can say whether it costs any accuracy.
    """
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    ]
    return np.array(offsets, dtype=np.int64)


def _line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Every pixel on the segment, Bresenham, as ``(y, x)`` pairs.

    Integer-only, so the same segment gives the same pixels everywhere. A zero-length segment --
    a stroke the child tapped rather than dragged -- is one pixel, not nothing.

    Not direction-symmetric: where a line passes exactly between two pixels the tie is broken by
    which end it started from, so a segment and its reverse can differ by a pixel. Deterministic
    either way, and invisible at a stroke's width, but it means reversing or mirroring a stroke is
    not a pixel-for-pixel operation -- worth knowing if KAN-31's augmentation ever does that.
    """
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return points
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y


def _runs(stroke: Stroke, index: int) -> tuple[list[float], list[float]]:
    """One stroke's x and y runs, materialised as floats.

    Materialised rather than iterated twice so that validating a run and then drawing it cannot see
    two different sequences, and so that numpy arrays, tuples and `range` objects all behave.
    """
    if isinstance(stroke, Mapping):
        # `list(a_dict)` gives its keys, so without this a record-shaped stroke falls through to
        # the text check below and is told to parse itself with `json.loads` -- advice that is
        # exactly wrong for the thing that arrived.
        raise TypeError(
            f"stroke {index} is a mapping; a stroke is two runs, [[x...], [y...]], not a record"
        )
    try:
        runs = list(stroke)
    except TypeError as error:
        raise TypeError(
            f"stroke {index} is not a pair of runs but a {type(stroke).__name__}; `strokes` is a "
            "list of strokes, and each stroke is [[x...], [y...]]"
        ) from error
    if len(runs) != 2:
        raise ValueError(
            f"stroke {index} has {len(runs)} runs; it must be [xs, ys]. Quick, Draw!'s raw format "
            "carries a third run of timings -- render the simplified format, which drops it"
        )
    for axis, run in zip(("x", "y"), runs, strict=True):
        if isinstance(run, (str, bytes, bytearray)):
            # Text iterates character by character, so a run of "12" would become the coordinates
            # 1 and 2 and render as a perfectly plausible drawing. Refused rather than drawn.
            raise TypeError(
                f"stroke {index}'s {axis} run is text, not a sequence of numbers; parse the record "
                "with json.loads before rendering it"
            )
    try:
        return [float(value) for value in runs[0]], [float(value) for value in runs[1]]
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"stroke {index} has runs that are not sequences of numbers ({error}); a single stroke "
            "passed on its own, rather than a list of strokes, arrives looking like this"
        ) from error


def _points(strokes: Strokes, config: RenderConfig) -> npt.NDArray[np.int64]:
    """Validate the stroke list and return every ``(y, x)`` pixel it covers."""
    if isinstance(strokes, (str, bytes, bytearray)):
        # The `strokes` column is JSON text (`primer.data.schemas`), so this is the likely mistake
        # at a call site, and iterating a string would otherwise fail somewhere far less obvious.
        raise TypeError("strokes is JSON text; parse it with json.loads before rendering")
    if isinstance(strokes, Mapping):
        # The whole record rather than its `strokes` column. Without this it iterates the dict's
        # keys, and the first key's *length* is reported as a run count -- so a one-character slip
        # at the call site comes back as advice about the raw format's timings.
        raise TypeError(
            "strokes is a mapping; pass the record's parsed `strokes` column, not the record"
        )
    try:
        count = len(strokes)
    except TypeError as error:
        raise TypeError(
            f"strokes must be a list of [xs, ys] strokes, not {type(strokes).__name__}; the column "
            "is null for image sources (`primer.data.schemas`), which have no strokes to render"
        ) from error
    if count == 0:
        raise ValueError("a drawing has at least one stroke; an empty list is a malformed record")

    limit = config.source_grid - 1
    scale, padding = config.scale, config.padding
    covered: list[tuple[int, int]] = []
    for index, stroke in enumerate(strokes):
        xs, ys = _runs(stroke, index)
        if len(xs) != len(ys):
            raise ValueError(f"stroke {index} has {len(xs)} x values and {len(ys)} y values")
        if not xs:
            raise ValueError(f"stroke {index} has no points")
        for value in (*xs, *ys):
            # Two checks rather than one `or`: a chained comparison against NaN is already False,
            # so folding them together leaves the first looking load-bearing while the second
            # silently does the work -- and reports a NaN as "outside the source grid", which is
            # not what is wrong with it.
            if not math.isfinite(value):
                raise ValueError(f"stroke {index} has the non-finite coordinate {value}")
            if not 0 <= value <= limit:
                raise ValueError(
                    f"stroke {index} has the coordinate {value}, outside the 0-{limit} source grid"
                )
        # Rounded half up. The endpoints are pinned, so a drawing that uses the whole grid uses the
        # whole content square.
        pixels = [
            (padding + math.floor(y * scale + 0.5), padding + math.floor(x * scale + 0.5))
            for x, y in zip(xs, ys, strict=True)
        ]
        if len(pixels) == 1:
            # A tap rather than a drag: there is no segment to walk, but it is still ink.
            covered.append(pixels[0])
        for start, end in itertools.pairwise(pixels):
            covered.extend(_line(start[1], start[0], end[1], end[0]))
    return np.array(covered, dtype=np.int64)


def render_strokes(strokes: Strokes, config: RenderConfig | None = None) -> Image:
    """Draw ``strokes`` onto a square greyscale canvas and return it as a ``uint8`` array.

    ``strokes`` is the Quick, Draw! simplified shape, ``[[[x...], [y...]], ...]`` on the 0-255 grid
    -- the parsed form of the ``strokes`` column, not the JSON text. The array is indexed
    ``[row, column]``, which is ``[y, x]``: the source grid is screen coordinates, y downwards.

    This is the function the ticket is about. Import it; do not reimplement it.
    """
    config = config or default_config()
    points = _points(strokes, config)
    low, high = int(np.min(points)), int(np.max(points))
    if low < config.radius or high >= config.size - config.radius:
        # Enforced rather than argued. Validation and the scale should make this impossible, but
        # numpy indexing wraps on a negative index: were they ever to drift apart, a stroke would
        # be drawn onto the opposite edge with no error at all.
        raise RuntimeError(
            f"a stroke mapped to {low}..{high}, outside the "
            f"{config.radius}..{config.size - config.radius - 1} a {config.line_width}px stroke "
            "can occupy; the coordinate check and the scale have drifted apart"
        )
    canvas: Image = np.full((config.size, config.size), config.background, dtype=np.uint8)
    stamped = (points[:, None, :] + _stamp(config.radius)[None, :, :]).reshape(-1, 2)
    canvas[stamped[:, 0], stamped[:, 1]] = config.ink
    return canvas


def to_png(image: Image) -> bytes:
    """Encode a ``uint8`` greyscale array as an 8-bit PNG.

    Written against `zlib` and `struct` rather than Pillow so that the whole strokes-to-bytes path
    has one set of dependencies and is exercised by CI, which does not install Pillow. A greyscale
    PNG is a header and one deflated block; there is no image processing here, and
    `test_render.py` walks the chunks, recomputes every CRC, and -- wherever Pillow is installed --
    checks that a real decoder reads the result.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D greyscale array, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 levels, got {image.dtype}")
    height, width = image.shape
    if height == 0 or width == 0:
        # Zero in either IHDR field is invalid PNG, and every decoder rejects the file. Caught here
        # rather than shipped as bytes that look well formed until something tries to open them.
        raise ValueError(f"a PNG needs at least one pixel each way, not {width}x{height}")

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    # Filter byte 0 ("none") in front of each row: no filtering keeps the output a function of the
    # pixels alone, so the same drawing gives the same bytes.
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    # Default compression, not level 9. A sketch is long runs of one byte, deflate's worst case for
    # the effort: measured on 50 real drawings at the shipped config, level 9 cost 2.647 ms against
    # the default's 0.312 ms -- 8.5x the time -- to save 23% of the bytes, which made it the single
    # largest cost in the whole strokes-to-bytes path.
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def render_strokes_png(strokes: Strokes, config: RenderConfig | None = None) -> bytes:
    """``render_strokes``, encoded as PNG bytes.

    The form anything taking image bytes wants -- `DrawingRecognizer.analyse`, a cache, an HTTP
    response. It is a thin wrapper rather than a second code path so that the pixels it encodes are
    the pixels the training loader trained on.
    """
    return to_png(render_strokes(strokes, config))
