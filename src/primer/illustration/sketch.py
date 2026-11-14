"""Turn the child's drawing into the control image ControlNet-Scribble conditions on.

Two steps, kept apart so each can be tested without model weights:

- ``load_drawing`` resolves a ``drawing_ref`` -- the durable handle the ``Illustrator`` contract
  defines -- to the original upload's bytes.
- ``prepare_sketch`` turns those bytes into what ControlNet-Scribble v1.1 expects: white strokes on
  black, square, at the generation size.

The drawing is a child's. Nothing here logs it, writes it anywhere, or derives a name from it.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

#: Schemes `load_drawing` resolves, from the `drawing_ref` definition in `interface.py`.
READABLE_SCHEMES = ("file", "gs")

#: The least a pixel must stand out from the paper (out of 255) to count as stroke. The threshold sits
#: halfway between the paper and the strongest ink, but never closer to the paper than this, so faint
#: pencil and yellow crayon survive while the grain of a phone photo's paper does not.
MIN_STROKE_CONTRAST = 24

#: After shrinking, an output pixel is stroke if at least this share of it (out of 255) was stroke.
#: 32 keeps a 1 px line through an 8x shrink, which is a 4000 px phone photo at 512.
MIN_STROKE_COVERAGE = 32


def _read_file(ref: str) -> bytes:
    parsed = urlparse(ref)
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(f"file:// drawing_ref must be a local absolute path, got host {parsed.netloc!r}")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise ValueError("file:// drawing_ref must be an absolute path")
    return path.read_bytes()  # an OSError here names the file; load_drawing strips it


def _read_gs(ref: str) -> bytes:
    import fsspec  # imported here so a file:// read never touches the GCS client

    with fsspec.open(ref, "rb") as handle:
        data: bytes = handle.read()
    return data


DEFAULT_READERS: dict[str, Callable[[str], bytes]] = {"file": _read_file, "gs": _read_gs}


def load_drawing(drawing_ref: str, readers: dict[str, Callable[[str], bytes]] | None = None) -> bytes:
    """The original upload's bytes for a ``gs://`` or ``file://`` drawing reference.

    Anything else is refused rather than guessed at: ``fixture://`` resolves to nothing by contract,
    and an ``http(s)://`` ref would mean fetching a child's drawing from somewhere unvetted.
    """
    if not drawing_ref:
        raise ValueError("drawing_ref is empty")
    scheme = urlparse(drawing_ref).scheme
    if scheme not in READABLE_SCHEMES:
        raise ValueError(f"drawing_ref scheme {scheme!r} is not readable; expected one of {READABLE_SCHEMES}")
    reader = (readers or DEFAULT_READERS)[scheme]
    # The OS error carries the path, and the path can be the child's file name. Raising inside the
    # `except` would still attach it as `__context__` (even with `from None`), where anything that
    # logs or inspects the chain can read it, so the replacement is raised outside the handler.
    data: bytes | None
    try:
        data = reader(drawing_ref)
    except OSError:
        data = None
    if data is None:
        raise ValueError("the drawing could not be read from its drawing_ref")
    if not data:
        raise ValueError("drawing_ref resolved to an empty file")
    return data


def prepare_sketch(data: bytes, size: int = 512) -> Image.Image:
    """White strokes on black, padded to a square and resized to ``size`` x ``size``, RGB.

    Children draw dark lines on light paper; the scribble model was trained on the opposite. The
    paper is judged from the border pixels, which are paper in almost every drawing: when the drawing
    as a whole is darker than its border, the lines are darker than the paper and the image is
    inverted. A drawing that is already light-on-dark (the synthetic baseline sketch) is left as it is.
    Transparency is paper. A drawing with no strokes left is refused rather than generated from an
    empty control image, because the page would then claim a sketch it never followed.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("the drawing is not a readable image") from error

    gray = _on_paper(ImageOps.exif_transpose(image)).convert("L")
    if ImageStat.Stat(gray).mean[0] <= _border_mean(gray):
        gray = ImageOps.invert(gray)
    paper = _border_mean(gray)
    threshold = paper + max(MIN_STROKE_CONTRAST, (_percentile(gray, 0.99) - paper) / 2)
    strokes = gray.point(lambda value: 255 if value >= threshold else 0)

    # Pad rather than stretch: squashing a tall drawing changes the shapes ControlNet must keep.
    side = max(strokes.size)
    square = Image.new("L", (side, side), 0)
    square.paste(strokes, ((side - strokes.width) // 2, (side - strokes.height) // 2))
    # Average over each output pixel's area, then keep any pixel a stroke covers enough of. Sampling
    # one source pixel per output pixel drops crayon lines thinner than the shrink factor.
    shrunk = square.resize((size, size), Image.Resampling.BOX)
    control = shrunk.point(lambda value: 255 if value >= MIN_STROKE_COVERAGE else 0)
    if control.getextrema()[1] == 0:
        raise ValueError("no strokes found in the drawing")
    return control.convert("RGB")


def _on_paper(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white: a canvas export's transparent background is the paper."""
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(paper, rgba)
    return image


def _percentile(gray: Image.Image, share: float) -> float:
    """The brightness below which ``share`` of the pixels fall."""
    histogram = gray.histogram()
    target = share * sum(histogram)
    seen = 0
    for value, count in enumerate(histogram):
        seen += count
        if seen >= target:
            return float(value)
    return 255.0


def _border_mean(gray: Image.Image) -> float:
    """Mean brightness of the outermost pixel ring."""
    width, height = gray.size
    edges = [
        (0, 0, width, 1),
        (0, height - 1, width, height),
        (0, 0, 1, height),
        (width - 1, 0, width, height),
    ]
    strips = [ImageStat.Stat(gray.crop(box)) for box in edges]
    return sum(stat.sum[0] for stat in strips) / sum(stat.count[0] for stat in strips)
