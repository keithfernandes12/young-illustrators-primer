"""What a drawing looks like by the time the recogniser sees it, in both directions (KAN-31, S6b1).

Two entry points, one module:

- ``prepare_drawing`` -- **inference**. A decoded drawing in, the recogniser's input out: optionally
  cropped to a box, optionally collapsed to greyscale, resized until its long side is
  ``drawing_render.yaml``'s ``size``, optionally binarised to the renderer's two levels, and
  padded to a square with its own paper.
  ``SiglipEncoder.encode`` calls it on every drawing it scores.
- ``augment_canvas`` -- **training**. A canvas from ``render_strokes`` in, something that looks more
  like a photographed crayon drawing out: line wobble, partial erasure, coloured ink, colour fill,
  paper texture and lighting, deterministic given a seed. The training loader (S6b2) applies it
  before ``prepare_drawing``, never at evaluation.

They are in one file for KAN-87's reason a second time: if training degraded images one way and
inference normalised them another, a measured ``drawing_top5_acc`` would compare a model with two
preprocessing paths, and nobody could say which one moved. ``tests/drawing/test_preprocess.py``
asserts there is one entry point per direction, and fails if anything else in the drawing path
imports or references one of a listed set of image-processing modules (torchvision's transforms,
scipy.ndimage, OpenCV, PIL.ImageOps and the like), or calls Pillow's resize, crop, thumbnail or a
non-RGB ``convert`` -- the forms a second path takes in practice. That is a guard, not a proof: a second
path written in plain numpy under another name would pass it, as it would pass KAN-87's.

The contract between the two: ``augment_canvas``'s output is always something ``prepare_drawing``
accepts -- ``uint8``, ``H x W x 3``, 0-255 -- and ``prepare_drawing`` leaves an un-augmented canvas
exactly as the renderer drew it at the shipped settings. So a training image is a rendered canvas,
perturbed, then put through the very function an upload goes through.

**Arrays in, arrays out. Nothing here decodes, and no image library is imported.** Decoding an
upload means JPEG, which means Pillow, which is in the ``drawing`` extra that CI does not install;
keeping decode at the call site is what lets CI cover every line of this file. That is also why
``model.flattened`` -- which composites transparency onto paper -- stays in ``model.py``: it is a
Pillow function that runs before an array exists, so decode and composite are one step at the call
site, and this module takes the composited RGB array.

Every probability, magnitude, range and scale lives in ``configs/drawing_preprocess.yaml``, except
the three that belong to the renderer -- ``size``, ``ink`` and ``background`` -- which are read from
``drawing_render.yaml`` so that the inference output lands on exactly the canvas the renderer
draws. What stays in code is structure, named below: how many waves make a smooth field, how much
coverage counts as a line, and where contrast pivots.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from primer.drawing.render import RenderConfig
from primer.drawing.render import default_config as default_render_config

CONFIG_PATH = Path(__file__).parent / "configs" / "drawing_preprocess.yaml"

Image = npt.NDArray[np.uint8]
Pixels = npt.NDArray[np.float64]
Colour = tuple[int, int, int]

#: ITU-R BT.601 luma weights: what Pillow's ``convert("L")`` uses, so a greyscale drawing here
#: matches one an image tool would have produced.
LUMA = np.array([0.299, 0.587, 0.114])

#: ``augment_canvas``'s operations, one random stream each, in this order.
OPERATIONS = ("wobble", "erasure", "paper", "fill", "ink", "lighting")

#: Plane waves summed into one smooth random field (``_waves``). Structure, not a setting: the
#: field's scale and strength are configured wherever it is used.
WAVES = 3
#: Ink coverage above which a pixel counts as part of a line, for deciding what is enclosed.
LINE_COVERAGE = 0.5
#: The level contrast pivots on (``_light``): mid-grey, so the pivot does not depend on the drawing.
PIVOT = 127.5


# ---------------------------------------------------------------------------------------- config


def _section(path: Path, name: str) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        kind = "an empty document" if raw is None else type(raw).__name__
        raise TypeError(f"{path.name} should be a mapping of settings, not {kind}")
    if name not in raw:
        raise ValueError(f"{path.name} has no `{name}:` section of settings")
    section = raw[name]
    if not isinstance(section, dict):
        raise TypeError(f"{path.name}'s `{name}:` should be a mapping of settings, not {section!r}")
    return section


def _settings(cls: type, section: dict[str, Any], path: Path, name: str) -> dict[str, Any]:
    """The section's values for ``cls``, refusing a missing setting **and an unknown one**.

    Unknown keys are refused because the likeliest one is a spelling: ``binarize: true`` would
    otherwise be ignored without a word, and the run would measure the colour path while its config
    said otherwise.
    """
    wanted = [f.name for f in fields(cls) if f.name != "render"]
    missing = [key for key in wanted if key not in section]
    if missing:
        raise ValueError(f"{path.name} `{name}:` does not set {', '.join(missing)}")
    unknown = sorted(set(section) - set(wanted))
    if unknown:
        raise ValueError(f"{path.name} `{name}:` sets {', '.join(unknown)}, which nothing reads")
    return {key: section[key] for key in wanted}


def _real(name: str, value: object) -> float:
    # `bool` is an `int`, so `true` in YAML would otherwise read as 1.0. numpy numbers are welcome:
    # a box read from a dataframe column of int64 or float32 is still four numbers.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, not {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, not {value}")
    return float(value)


def _colours(name: str, value: object) -> tuple[Colour, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence) or not value:
        raise TypeError(f"{name} must be a non-empty list of [r, g, b] colours, not {value!r}")
    colours = []
    for colour in value:
        if isinstance(colour, (str, bytes, Mapping)) or not isinstance(colour, Sequence):
            raise TypeError(f"{name} holds {colour!r}, which is not an [r, g, b] colour")
        channels = tuple(colour)
        if len(channels) != 3 or any(
            isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 255 for c in channels
        ):
            raise ValueError(f"{name} holds {colour!r}; a colour is three whole levels in 0-255")
        colours.append((channels[0], channels[1], channels[2]))
    return tuple(colours)


@dataclass(frozen=True)
class PrepareConfig:
    """The inference direction's settings. ``render`` supplies ``size``, ``ink`` and ``background``.

    A variant for an experiment is ``dataclasses.replace(default_prepare_config(), greyscale=True)``
    -- S6c2's table of variants is exactly that -- so the shipped settings stay in the YAML.
    """

    greyscale: bool
    binarise: bool
    binarise_window: float
    binarise_offset: float
    render: RenderConfig = field(default_factory=default_render_config)

    def __post_init__(self) -> None:
        for name in ("greyscale", "binarise"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                # YAML reads `yes` and `on` as booleans but `"true"` and `1` as other things, both
                # truthy -- so a quoted value would switch the setting on by accident.
                raise TypeError(f"{name} must be true or false, not {value!r}")
        for name in ("binarise_window", "binarise_offset"):
            object.__setattr__(self, name, _real(name, getattr(self, name)))
        if not 0.0 < self.binarise_window <= 1.0:
            raise ValueError(f"binarise_window is a fraction of `size`, not {self.binarise_window}")
        if not 0.0 <= self.binarise_offset < 1.0:
            # At 1, nothing is ever darker than (1 - 1) x its neighbourhood: every page is blank.
            raise ValueError(f"binarise_offset must be in 0-1 (exclusive), not {self.binarise_offset}")
        if self.binarise and not self.greyscale:
            raise ValueError(
                "binarise: true needs greyscale: true -- binarising discards colour whatever the "
                "config says, so say so rather than have the file claim a colour path it is not"
            )

    @property
    def size(self) -> int:
        return self.render.size

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH, render: RenderConfig | None = None) -> PrepareConfig:
        values = _settings(cls, _section(path, "inference"), path, "inference")
        return cls(**values, render=render or default_render_config())


@dataclass(frozen=True)
class AugmentConfig:
    """The training direction's settings. See ``configs/drawing_preprocess.yaml`` for each one.

    ``render`` supplies the canvas's ``ink`` and ``background``, so the augmentation reads a
    rendered canvas with the levels it was rendered at.
    """

    wobble_probability: float
    wobble_amplitude: float
    wobble_wavelength: float
    erasure_probability: float
    erasure_patches: int
    erasure_size: float
    erasure_strength_min: float
    erasure_strength_max: float
    ink_colour_probability: float
    ink_colours: tuple[Colour, ...]
    fill_probability: float
    fill_opacity_min: float
    fill_opacity_max: float
    fill_coverage_wavelength: float
    fill_coverage_spread: float
    fill_colours: tuple[Colour, ...]
    paper_tints: tuple[Colour, ...]
    paper_texture: float
    paper_texture_wavelength: float
    paper_grain: float
    lighting_probability: float
    lighting_gradient: float
    lighting_exposure: float
    lighting_contrast: float
    render: RenderConfig = field(default_factory=default_render_config)

    def __post_init__(self) -> None:
        # Normalised in place (the dataclass is frozen, hence `object.__setattr__`) so that YAML's
        # lists and ints arrive as the tuples and floats the annotations promise.
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "render":
                continue
            if f.name.endswith(("_colours", "_tints")):
                object.__setattr__(self, f.name, _colours(f.name, value))
            elif f.name == "erasure_patches":
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"erasure_patches must be a whole number >= 1, not {value!r}")
            else:
                object.__setattr__(self, f.name, _real(f.name, value))

        for f in fields(self):
            if f.name.endswith("_probability") and not 0.0 <= getattr(self, f.name) <= 1.0:
                raise ValueError(f"{f.name} must be in 0-1, not {getattr(self, f.name)}")
        for name in ("wobble_amplitude", "paper_texture", "paper_grain"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative, not {getattr(self, name)}")
        for name in ("wobble_wavelength", "fill_coverage_wavelength", "paper_texture_wavelength"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, not {getattr(self, name)}")
        if not 0.0 < self.erasure_size <= 1.0:
            raise ValueError(f"erasure_size is a fraction of the canvas side, not {self.erasure_size}")
        for low, high in (
            ("erasure_strength_min", "erasure_strength_max"),
            ("fill_opacity_min", "fill_opacity_max"),
        ):
            if not 0.0 <= getattr(self, low) <= getattr(self, high) <= 1.0:
                raise ValueError(
                    f"{low} and {high} need 0 <= min <= max <= 1, not "
                    f"{getattr(self, low)}...{getattr(self, high)}"
                )
        if not 0.0 <= self.fill_coverage_spread <= 1.0:
            raise ValueError(f"fill_coverage_spread must be in 0-1, not {self.fill_coverage_spread}")
        # At 1, the gradient can shade one side to black, the exposure the whole page, and the
        # contrast flatten it to uniform grey -- each erases the drawing it is meant to perturb.
        for name in ("lighting_gradient", "lighting_exposure", "lighting_contrast"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in 0-1 (exclusive), not {getattr(self, name)}")
        # Two ways a valid-looking config draws nothing. Texture on paper near white is clipped at
        # 255 rather than seen -- measured with a pure-white tint, 54.8% of paper values came out at
        # exactly 255, the failure `_light` was already fixed for. And uncoloured ink that is not
        # darker than the paper composites a drawing into a blank page.
        headroom = self.paper_texture + 3.0 * self.paper_grain
        for tint in self.paper_tints:
            if max(tint) + headroom > 255:
                raise ValueError(
                    f"paper tint {tint} plus texture {self.paper_texture} and three standard "
                    f"deviations of grain {self.paper_grain} passes 255, so the texture is clipped "
                    "away; use a darker tint, as a photographed page is"
                )
            if min(tint) <= self.render.ink:
                raise ValueError(
                    f"paper tint {tint} is no lighter than the renderer's ink ({self.render.ink}), "
                    "so an uncoloured drawing would vanish into the paper"
                )

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH, render: RenderConfig | None = None) -> AugmentConfig:
        values = _settings(cls, _section(path, "training"), path, "training")
        return cls(**values, render=render or default_render_config())


@lru_cache(maxsize=1)
def default_prepare_config() -> PrepareConfig:
    """The shipped inference settings, parsed once -- the encoder calls this once per drawing."""
    return PrepareConfig.from_yaml()


@lru_cache(maxsize=1)
def default_augment_config() -> AugmentConfig:
    """The shipped training settings, parsed once -- the loader calls this once per record."""
    return AugmentConfig.from_yaml()


# ------------------------------------------------------------------------------------- shared


def _levels(pixels: Pixels) -> Image:
    """Floats back to 0-255 levels, rounded half up -- the convention ``render_strokes`` uses.

    The only rounding in either direction: everything before it stays in float, so a pipeline of
    operations rounds once rather than once per step.
    """
    return np.clip(np.floor(pixels + 0.5), 0, 255).astype(np.uint8)


@lru_cache(maxsize=32)
def _area_weights(n_in: int, n_out: int) -> npt.NDArray[np.float64]:
    """An ``n_out x n_in`` matrix: each output pixel as the overlap-weighted mean of input pixels.

    Area resampling, the right filter for shrinking a sketch: every input pixel contributes in
    proportion to how much of it an output pixel covers, so a thin line fades rather than vanishing
    between samples, which point sampling or an un-prefiltered bilinear does to a 2px crayon line.
    Separable, so a resize is two matrix products. Cached, and read-only because it is shared.
    """
    scale = n_in / n_out
    starts = np.arange(n_out) * scale
    ends = starts + scale
    source = np.arange(n_in)
    overlap = np.clip(
        np.minimum(ends[:, None], source[None, :] + 1) - np.maximum(starts[:, None], source[None, :]),
        0.0,
        None,
    )
    weights = overlap / overlap.sum(axis=1, keepdims=True)
    weights.flags.writeable = False
    return weights


def _resize(pixels: Pixels, rows: int, columns: int) -> Pixels:
    """``pixels`` (``H x W x C``) resampled to ``rows x columns x C``. Exact when nothing changes."""
    height, width = pixels.shape[:2]
    if height == rows and width == columns:
        return pixels
    resampled = np.tensordot(_area_weights(height, rows), pixels, axes=(1, 0))
    return np.tensordot(resampled, _area_weights(width, columns), axes=(1, 1)).transpose(0, 2, 1)


# ----------------------------------------------------------------------------- inference direction

#: How far, in pixels, an annotated box may reach past the image before it is refused rather than
#: clamped. Measured, not chosen: of Amateur Drawings' 174,981 boxes, 3,311 (1.89%) reach past
#: their image once rounded outwards to whole pixels, and every one of them by at most 1.00px --
#: an annotation that ends on the image's last pixel boundary, not a box from another image.
BOX_TOLERANCE = 1.0


def _image(image: object) -> Image:
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"expected a decoded drawing as a numpy array, not {type(image).__name__}; this module "
            "does not decode -- open the bytes and composite them with `model.flattened` first"
        )
    if image.dtype != np.uint8:
        # A float array is ambiguous between 0-1 and 0-255, and guessing wrong is a white page.
        raise TypeError(f"expected uint8 levels, as a decoder produces, not {image.dtype}")
    if image.ndim == 3 and image.shape[2] == 4:
        raise ValueError(
            "the drawing has four channels; composite its alpha with `model.flattened` first -- "
            "dropping alpha turns a sketch on a transparent canvas into a black square"
        )
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError(f"expected H x W greyscale or H x W x 3 RGB, not shape {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"the drawing has no pixels: shape {image.shape}")
    return image


def _crop(image: Image, box: object) -> Image:
    """``image`` cut to ``box`` -- ``(x, y, width, height)`` in pixels, as Amateur Drawings stores it.

    The box is rounded **outwards** to whole pixels, so a fractional annotation never trims the
    character it encloses, then clamped to the image. A box reaching more than ``BOX_TOLERANCE``
    past the image is refused instead: it belongs to a different image, or to this one before it
    was resized, and cropping what is left would score part of a drawing as if it were all of it.
    """
    if isinstance(box, (str, bytes, Mapping)) or not isinstance(box, (Sequence, np.ndarray)):
        raise TypeError(f"box must be (x, y, width, height), not {box!r}")
    values = [_real("box", value) for value in box]
    if len(values) != 4:
        raise ValueError(f"box must be (x, y, width, height), not {len(values)} values")
    x, y, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError(f"box has no area: {width} x {height}")
    rows, columns = image.shape[:2]
    if (
        x < -BOX_TOLERANCE
        or y < -BOX_TOLERANCE
        or x + width > columns + BOX_TOLERANCE
        or y + height > rows + BOX_TOLERANCE
    ):
        raise ValueError(
            f"box {tuple(values)} reaches more than {BOX_TOLERANCE:g}px outside the {columns} x {rows} image"
        )
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(columns, math.ceil(x + width)), min(rows, math.ceil(y + height))
    if right <= left or bottom <= top:
        raise ValueError(f"box {tuple(values)} lies outside the {columns} x {rows} image")
    return image[top:bottom, left:right]


def _fit(pixels: Pixels, size: int) -> Pixels:
    """``pixels`` resized so the long side is ``size``, keeping the aspect ratio. Never stretched.

    The renderer preserves each drawing's aspect ratio (Quick, Draw!'s simplified format was scaled
    uniformly), so a squashed upload would show the recogniser shapes no training image has. The
    recogniser's own processor resizes straight to a square -- measured on a 512 x 384 Amateur
    Drawing, it returns 224 x 224 -- and 98.65% of that corpus is not square (88% portrait, median
    aspect 0.75), so the squash is the normal case, not an edge one. This step is why the processor
    never gets the chance.
    """
    height, width = pixels.shape[:2]
    scale = size / max(height, width)
    rows = max(1, math.floor(height * scale + 0.5))
    columns = max(1, math.floor(width * scale + 0.5))
    return _resize(pixels, rows, columns)


def _border_colour(content: Image) -> tuple[int, ...]:
    """The paper's colour, as the per-channel median of the drawing's outermost pixels.

    What a non-square drawing is padded with. Not the renderer's white: a photographed page is grey
    or cream, and white bars beside it are a frame no training image has and no parent drew. The
    median of the edge is the paper wherever ink covers less than half of it, which on a drawing
    centred on a page is everywhere.
    """
    edge = np.concatenate([content[0], content[-1], content[:, 0], content[:, -1]])
    return tuple(math.floor(value + 0.5) for value in np.median(edge, axis=0))


def _square(content: Image, colour: tuple[int, ...], size: int) -> Image:
    """``content`` centred on a ``size x size`` square of ``colour``."""
    rows, columns = content.shape[:2]
    if rows == size and columns == size:
        return content
    square = np.empty((size, size, content.shape[2]), dtype=np.uint8)
    square[:, :] = colour
    top, left = (size - rows) // 2, (size - columns) // 2
    square[top : top + rows, left : left + columns] = content
    return square


def _local_mean(values: Pixels, half: int) -> Pixels:
    """Each pixel's mean over the ``(2 * half + 1)``-pixel square around it, clipped to the image.

    A summed-area table, so the cost does not grow with the window. At the border the window is
    cut to the pixels that exist rather than padded, so the edge of a photograph is compared with
    the photograph and not with an invented margin.
    """
    rows, columns = values.shape
    table = np.zeros((rows + 1, columns + 1))
    table[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    top = np.clip(np.arange(rows) - half, 0, rows)
    bottom = np.clip(np.arange(rows) + half + 1, 0, rows)
    left = np.clip(np.arange(columns) - half, 0, columns)
    right = np.clip(np.arange(columns) + half + 1, 0, columns)
    sums = table[bottom][:, right] - table[top][:, right] - table[bottom][:, left] + table[top][:, left]
    counts = (bottom - top)[:, None] * (right - left)[None, :]
    mean: Pixels = sums / counts
    return mean


def _marks(values: Pixels, config: PrepareConfig) -> npt.NDArray[np.bool_]:
    """Which pixels are marks: clearly darker than their own neighbourhood.

    A local threshold (Bradley and Roth's): a pixel is ink when it is more than ``binarise_offset``
    darker than the mean of the ``binarise_window`` square around it. Paper is uniform at the scale
    of the window wherever the light falls, so it stays paper on the bright side of a page and the
    shaded side alike, and so does the inside of a uniform table round the sheet. **The edge where
    sheet meets table does not**: within a window of it, the table is darker than its neighbourhood
    and comes out as a band of ink about half a window wide -- measured on a blank 230-grey sheet
    on a 120-grey table, 6,828 ink pixels, all along that seam. Same mechanism as the outline below.

    **Two global thresholds failed before this one, both measured on 60 real Amateur Drawings.**
    Otsu applied after padding split the white padding from the page and returned the whole
    photograph as ink -- median 74.2%, worst exactly the 75.0% of a 384 x 512 drawing left after
    padding it square. Otsu on the drawing alone fixed that, and then binarised the photographs'
    own edges to ink: in the median drawing 39.8% of the outermost pixels, and in 42 of the 60 more
    than a quarter -- shadows, vignetting and the table round the paper, which one threshold for a
    whole page cannot tell from a mark. A global threshold does not remove lighting, which is what
    S6c2's binarised row exists to do; it turns lighting into ink.

    The trade it makes: a large solid area -- a figure coloured in hard with a dark crayon -- is its
    own neighbourhood, so it binarises to its outline. Quick, Draw! drawings are outlines too, so
    that is the direction the renderer's convention points, but it is a change to the drawing and
    S6c2 is what measures whether it costs anything.
    """
    half = max(1, math.floor(config.binarise_window * config.size / 2 + 0.5))
    result: npt.NDArray[np.bool_] = values < (1.0 - config.binarise_offset) * _local_mean(values, half)
    return result


def prepare_drawing(image: Image, config: PrepareConfig | None = None, *, box: object = None) -> Image:
    """The recogniser's input for one decoded drawing: ``size x size x 3``, ``uint8``.

    ``image`` is ``H x W`` greyscale or ``H x W x 3`` RGB, already decoded and composited -- see the
    module docstring for why decoding is not done here. ``box``, if given, is ``(x, y, width,
    height)``: the drawing is cut to it first. Nothing in the product has a box at upload time --
    it is for evaluating against annotated corpora such as Amateur Drawings, and a figure measured
    with it is an upper bound the product cannot reach until S6e detects characters.

    In order: crop; greyscale; resize by area until the long side is the renderer's ``size``;
    binarise to its ``ink`` and ``background``; centre on a square. Binarising comes after the
    resize so the output has exactly the renderer's two levels at exactly its resolution -- a
    threshold applied before it would be averaged back into greys -- and before the padding, so
    the threshold sees the drawing alone (see ``_marks`` for what padding first did). A
    binarised drawing is padded with ``background``; any other with its own paper colour
    (``_border_colour``). The output is always three channels, so every variant reaches the
    recogniser in the same shape and only the setting being compared differs.

    At the shipped settings a canvas from ``render_strokes`` comes back exactly as it went in,
    repeated on three channels -- checked on 150 real Quick, Draw! drawings -- so a rendered
    training image and an upload meet the recogniser through the same function without the
    function moving the training image.

    Binarising assumes dark marks on lighter paper -- what is darker than its surroundings becomes
    ``ink``. Chalk on a blackboard would come out as its outline's surroundings.
    """
    config = config or default_prepare_config()
    pixels = _image(image)
    if box is not None:
        pixels = _crop(pixels, box)
    values = pixels.astype(np.float64)
    if values.ndim == 2:
        values = values[:, :, None]
    elif config.greyscale:
        values = (values @ LUMA)[:, :, None]
    values = _fit(values, config.size)
    if config.binarise:
        grey = values[:, :, 0]
        binary = np.full(grey.shape, config.render.background, dtype=np.uint8)
        binary[_marks(grey, config)] = config.render.ink
        square = _square(binary[:, :, None], (config.render.background,), config.size)
    else:
        content = _levels(values)
        square = _square(content, _border_colour(content), config.size)
    return square if square.shape[2] == 3 else np.repeat(square, 3, axis=2)


# ------------------------------------------------------------------------------ training direction


def _canvas(canvas: object) -> Image:
    if not isinstance(canvas, np.ndarray) or canvas.dtype != np.uint8 or canvas.ndim != 2:
        shape = getattr(canvas, "shape", None)
        raise TypeError(
            "augment a canvas from `render_strokes` -- a 2-D uint8 array -- not "
            f"{type(canvas).__name__} {shape or ''}; augmentation starts from the renderer's "
            "output, and an RGB image has already been through it"
        )
    if canvas.shape[0] == 0 or canvas.shape[1] == 0:
        raise ValueError(f"the canvas has no pixels: shape {canvas.shape}")
    return canvas


def _seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be a whole number, not {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, not {seed}")
    return int(seed)


def _waves(shape: tuple[int, int], rng: np.random.Generator, wavelength: float) -> Pixels:
    """A smooth random field in -1..1: the mean of three plane waves in random directions.

    The one source of smoothness for the wobble's displacement, the paper's mottling and which part
    of a drawing gets coloured in. Sinusoids rather than upsampled noise so that no resampling --
    and no second resize -- is involved.
    """
    rows, columns = np.indices(shape, dtype=np.float64)
    total = np.zeros(shape)
    for _ in range(WAVES):
        angle, phase = rng.uniform(0.0, 2.0 * math.pi, size=2)
        direction = columns * math.cos(angle) + rows * math.sin(angle)
        total += np.sin(2.0 * math.pi * direction / wavelength + phase)
    return total / WAVES


def _wobble(alpha: Pixels, rng: np.random.Generator, config: AugmentConfig) -> Pixels:
    """Every pixel fetched from a smoothly displaced position. Nearest-neighbour, so ink stays ink.

    Displacement reaching past the edge takes the edge pixel, which on a rendered canvas is paper
    (``padding`` keeps strokes off the border).
    """
    shape = (alpha.shape[0], alpha.shape[1])
    rows, columns = np.indices(shape, dtype=np.float64)
    shift_rows = config.wobble_amplitude * _waves(shape, rng, config.wobble_wavelength)
    shift_columns = config.wobble_amplitude * _waves(shape, rng, config.wobble_wavelength)
    source_rows = np.clip(np.floor(rows + shift_rows + 0.5), 0, shape[0] - 1).astype(np.intp)
    source_columns = np.clip(np.floor(columns + shift_columns + 0.5), 0, shape[1] - 1).astype(np.intp)
    return alpha[source_rows, source_columns]


def _erase(alpha: Pixels, rng: np.random.Generator, config: AugmentConfig) -> Pixels:
    """Ink lifted by ``erasure_strength_min``-``max`` of itself inside a few random rectangles."""
    erased = alpha.copy()
    rows, columns = alpha.shape
    for _ in range(int(rng.integers(1, config.erasure_patches + 1))):
        height = int(rng.integers(1, max(1, int(config.erasure_size * rows)) + 1))
        width = int(rng.integers(1, max(1, int(config.erasure_size * columns)) + 1))
        top = int(rng.integers(0, rows - height + 1))
        left = int(rng.integers(0, columns - width + 1))
        erased[top : top + height, left : left + width] *= 1.0 - rng.uniform(
            config.erasure_strength_min, config.erasure_strength_max
        )
    return erased


def _paper(shape: tuple[int, int], rng: np.random.Generator, config: AugmentConfig) -> Pixels:
    """A sheet of paper: a tint, low-frequency mottling and per-pixel grain, ``H x W x 3``."""
    tint = np.array(config.paper_tints[int(rng.integers(len(config.paper_tints)))], dtype=np.float64)
    mottling = config.paper_texture * _waves(shape, rng, config.paper_texture_wavelength * max(shape))
    grain = rng.normal(0.0, 1.0, size=shape) * config.paper_grain
    return tint[None, None, :] + (mottling + grain)[:, :, None]


def _enclosed(lines: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Paper pixels with a line somewhere to their left, right, above and below.

    An approximation of "inside the drawing", deliberately: a true flood fill is a Python loop over
    the canvas, and this is four cumulative passes. It over-fills concave shapes -- the notch of a
    C counts as inside -- which for a child colouring in is closer to life than an exact fill.
    """
    left = np.logical_or.accumulate(lines, axis=1)
    right = np.logical_or.accumulate(lines[:, ::-1], axis=1)[:, ::-1]
    above = np.logical_or.accumulate(lines, axis=0)
    below = np.logical_or.accumulate(lines[::-1], axis=0)[::-1]
    result: npt.NDArray[np.bool_] = left & right & above & below & ~lines
    return result


def _fill(paper: Pixels, alpha: Pixels, rng: np.random.Generator, config: AugmentConfig) -> Pixels:
    """Part of the drawing's enclosed area coloured in, translucently."""
    colour = np.array(config.fill_colours[int(rng.integers(len(config.fill_colours)))], dtype=np.float64)
    opacity = rng.uniform(config.fill_opacity_min, config.fill_opacity_max)
    shape = (alpha.shape[0], alpha.shape[1])
    # Which part gets coloured: a smooth field above a random level, so a drawing is coloured in
    # patches rather than all or nothing.
    field = _waves(shape, rng, config.fill_coverage_wavelength * max(shape))
    coverage = field > rng.uniform(-config.fill_coverage_spread, config.fill_coverage_spread)
    region = _enclosed(alpha > LINE_COVERAGE) & coverage
    filled = paper.copy()
    filled[region] = filled[region] * (1.0 - opacity) + colour * opacity
    return filled


def _light(image: Pixels, rng: np.random.Generator, config: AugmentConfig) -> Pixels:
    """A phone photograph's lighting: shade towards one side, lose contrast, lose exposure.

    **Only ever darkens or flattens**, so nothing is pushed past 255 and clipped back. The first
    version shifted brightness and stretched contrast both ways, and on white paper -- which is
    every rendered canvas -- a brighter, punchier draw clipped the whole page back to 255: measured
    on the test square, one seed in four fired the operation and changed no pixel at all. A
    photographed page is darker and flatter than a render, never cleaner, so over-exposure is left
    out on purpose; a blown-out white page is what the renderer already produces.

    Contrast pivots on mid-grey, not on the image's mean: the mean depends on where the ink is, so
    pivoting on it would let wobble or erasure shift every paper pixel through this step and undo
    the one-stream-per-operation independence ``augment_canvas`` promises.
    """
    shape = (image.shape[0], image.shape[1])
    angle = rng.uniform(0.0, 2.0 * math.pi)
    depth = rng.uniform(0.0, config.lighting_gradient)
    exposure = 1.0 - rng.uniform(0.0, config.lighting_exposure)
    contrast = 1.0 - rng.uniform(0.0, config.lighting_contrast)
    rows, columns = np.indices(shape, dtype=np.float64)
    ramp: Pixels = columns * math.cos(angle) + rows * math.sin(angle)
    low, high = float(np.min(ramp)), float(np.max(ramp))
    ramp = (ramp - low) / (high - low) if high > low else np.zeros(shape)
    shaded = image * (1.0 - depth * ramp)[:, :, None]
    result: Pixels = ((shaded - PIVOT) * contrast + PIVOT) * exposure
    return result


def augment_canvas(canvas: Image, seed: int, config: AugmentConfig | None = None) -> Image:
    """A rendered canvas made to look photographed and coloured in: ``H x W x 3``, ``uint8``.

    ``canvas`` is ``render_strokes``'s output. The same canvas, seed and config give the same array
    on every call. Each operation draws from its own stream, spawned from ``seed`` by
    ``numpy.random.SeedSequence`` -- so whether one operation fires, and how many numbers it uses,
    never shifts another's **random draws**. That is independence of the dice, not of the pixels:
    fill colours whatever the lines enclose *after* wobble and erasure, and its enclosure test runs
    along whole rows and columns, so erasing one gap can change which paper is filled far from it
    (measured: in 18 of 50 seeds with fill forced on). Paper, ink colour and lighting depend on
    nothing upstream, so their pixels do stay put. Repeatable run to run on one machine and one
    numpy; bit-identical across platforms is not promised, because ``sin`` is not.

    The canvas is read as ink coverage between the renderer's ``ink`` and ``background``, so the
    operations act on the drawing and not on two particular grey levels: wobble and erasure move
    and lift coverage, paper and fill build the sheet, then the ink is laid over it in its colour
    and the lighting falls on the whole page.

    Training only. Evaluation and inference use ``prepare_drawing`` alone, and the loader applies
    ``prepare_drawing`` after this, so augmented and real drawings meet the recogniser through the
    same function.
    """
    config = config or default_augment_config()
    canvas = _canvas(canvas)
    streams = dict(
        zip(
            OPERATIONS,
            (np.random.default_rng(child) for child in np.random.SeedSequence(_seed(seed)).spawn(6)),
            strict=True,
        )
    )
    ink, paper_level = config.render.ink, config.render.background
    alpha = np.clip((paper_level - canvas.astype(np.float64)) / (paper_level - ink), 0.0, 1.0)
    shape = (canvas.shape[0], canvas.shape[1])

    # The firing draw comes first in every stream, and is made whether or not the probability is
    # zero, so a stream's later draws do not depend on its own setting either.
    if streams["wobble"].random() < config.wobble_probability:
        alpha = _wobble(alpha, streams["wobble"], config)
    if streams["erasure"].random() < config.erasure_probability:
        alpha = _erase(alpha, streams["erasure"], config)
    page = _paper(shape, streams["paper"], config)
    if streams["fill"].random() < config.fill_probability:
        page = _fill(page, alpha, streams["fill"], config)
    coloured = streams["ink"].random() < config.ink_colour_probability
    choice = config.ink_colours[int(streams["ink"].integers(len(config.ink_colours)))]
    colour = np.array(choice if coloured else (ink, ink, ink), dtype=np.float64)
    image = page * (1.0 - alpha[:, :, None]) + colour[None, None, :] * alpha[:, :, None]
    if streams["lighting"].random() < config.lighting_probability:
        image = _light(image, streams["lighting"], config)
    return _levels(image)
