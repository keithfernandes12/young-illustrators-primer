"""The rest of the character sheet: one sheet per character, every field filled (KAN-38, S6e).

``CharacterSheetRecognizer.analyse`` is the full ``DrawingRecognizer``. For one drawing it returns:

- ``description`` -- Florence-2's caption of the whole drawing, for the story writer.
- one ``Character`` per character Florence-2 detects (parts of a figure folded into the figure,
  see ``characters_in``), **most confident first** -- the order ``DrawingAnalysis.characters`` and
  ``.main`` promise, and ``main`` is the story's subject -- each with:

  - ``label`` / ``candidates`` / ``embedding`` from the SigLIP 2 recogniser, scored on that
    character's box through ``prepare_drawing`` -- the one preprocessing path;
  - ``body_type`` from the detector's word for it (``body_type_of``);
  - ``crop_ref`` / ``mask_ref`` -- the child's own lines cut out with SAM 2.1's mask, written to an
    ``ArtifactStore``;
  - ``keypoints`` for humanoids, from ViTPose, normalised to the crop.

``reference_image_ref`` and ``approved`` stay empty: the stylise-and-approve step sets them
(``interface.py`` says who fills what).

**The models' outputs are not ground truth, and were measured against it** on 40 real Amateur
Drawings before shipping (plan, Verified facts, 2026-11-15): the figure is found at IoU >= 0.5 on
37/40, SAM's mask matches the annotated one at IoU >= 0.5 on 26/40, and ViTPose places 61% of
joints within 10% of the figure's size. Every one of them was trained on photographs.

**Child data.** Crops, masks, keypoints and the embedding are derived from a child's drawing.
``ArtifactStore`` refs are content hashes, never names or ids, and nothing here logs a field.

The pure functions below (box filtering, body type, keypoints, cut-outs, orchestration) run in CI
with fakes; ``HuggingFaceSheetModels`` needs the ``sheet`` extra (torchvision included) and cached
weights, and loads lazily on the first drawing.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import yaml

from primer.drawing.interface import KEYPOINT_NAMES, BodyType, Character, DrawingAnalysis, Keypoint
from primer.drawing.model import DrawingRecognizerModel, flattened
from primer.drawing.render import to_png

CONFIG_PATH = Path(__file__).parent / "configs" / "drawing_sheet.yaml"

#: ``(x0, y0, x1, y1)`` in the decoded drawing's pixels -- the detector's convention.
Box = tuple[float, float, float, float]
RGB = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class SheetConfig:
    """See ``configs/drawing_sheet.yaml`` for every setting and why it has its value."""

    florence: str
    florence_revision: str
    florence_license: str
    caption_task: str
    detect_task: str
    sam: str
    sam_revision: str
    sam_license: str
    vitpose: str
    vitpose_revision: str
    vitpose_license: str
    keypoint_threshold: float
    containment: float
    min_area: float
    max_characters: int
    humanoid_words: tuple[str, ...]
    creature_words: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("humanoid_words", "creature_words"):
            object.__setattr__(self, name, tuple(str(word).casefold() for word in getattr(self, name)))
        for name in ("keypoint_threshold", "containment", "min_area"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a share in 0-1, not {value!r}")
        if (
            isinstance(self.max_characters, bool)
            or not isinstance(self.max_characters, int)
            or self.max_characters < 1
        ):
            raise ValueError(f"max_characters must be a whole number >= 1, not {self.max_characters!r}")
        overlap = set(self.humanoid_words) & set(self.creature_words)
        if overlap:
            raise ValueError(f"{sorted(overlap)} cannot be both humanoid and creature")

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> SheetConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        wanted = [f.name for f in fields(cls)]
        unknown = sorted(set(raw) - set(wanted) - {"name"})
        missing = [key for key in wanted if key not in raw]
        if missing or unknown:
            raise ValueError(f"{path.name}: missing {missing}, unknown {unknown}")
        return cls(**{key: raw[key] for key in wanted})


@lru_cache(maxsize=1)
def default_sheet_config() -> SheetConfig:
    return SheetConfig.from_yaml()


# ------------------------------------------------------------------------------------- the seams


class SheetModels(Protocol):
    """The three models beside the recogniser. Tests pass a fake; production loads weights."""

    def caption(self, image: RGB) -> str: ...

    def detect(self, image: RGB) -> list[tuple[str, Box]]: ...

    def segment(self, image: RGB, box: Box) -> Mask: ...

    def pose(self, image: RGB, box: Box) -> list[tuple[float, float, float]]: ...


class ArtifactStore(Protocol):
    """Where a character's crop and mask go. Returns a reference, never echoes the bytes."""

    def put(self, data: bytes, kind: str) -> str: ...


class MemoryArtifactStore:
    """Keeps artifacts in memory under content-hash refs. The default; serving supplies GCS.

    The ref is the SHA-256 of the bytes, so it says nothing about the child, the drawing or the
    character -- `AGENTS.md`: no PII in object names -- and the same crop stored twice is one entry.
    """

    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}

    def put(self, data: bytes, kind: str) -> str:
        if not re.fullmatch(r"[a-z]+", kind):
            raise ValueError(f"kind is a plain word like 'crop', not {kind!r}")
        ref = f"mem://{kind}/{hashlib.sha256(data).hexdigest()}"
        self.items[ref] = data
        return ref


# ------------------------------------------------------------------------------- pure decisions


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inside(inner: Box, outer: Box) -> float:
    """The share of ``inner`` that lies inside ``outer``."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = _area(inner)
    return ix * iy / area if area else 0.0


def characters_in(
    detections: Sequence[tuple[str, Box]], size: tuple[int, int], config: SheetConfig | None = None
) -> list[tuple[str | None, Box]]:
    """The detections that are characters, largest first; the whole drawing if there are none.

    Largest first is the order grouping needs (a part is folded into the larger box that holds it);
    ``analyse`` puts the finished sheets in confidence order.

    A box lying ``containment`` or more inside a larger kept box is a part of that character -- a
    face, a shoe -- not a second one; boxes under ``min_area`` of the drawing are specks. With no
    box left, the whole drawing is one character whose label is ``None`` (body type unknown):
    the recogniser still scores it, so ``analyse`` never returns an empty sheet list.
    """
    config = config or default_sheet_config()
    width, height = size
    clipped = [
        (label, (max(0.0, b[0]), max(0.0, b[1]), min(float(width), b[2]), min(float(height), b[3])))
        for label, b in detections
    ]
    # `_area(b) > 0` as well as the share: a box wholly outside the drawing clips to a reversed
    # rectangle of zero area, which `min_area = 0` would otherwise keep and crash on later.
    big = [(label, b) for label, b in clipped if _area(b) > 0 and _area(b) >= config.min_area * width * height]
    kept: list[tuple[str | None, Box]] = []
    for label, box in sorted(big, key=lambda item: _area(item[1]), reverse=True):
        if all(_inside(box, outer) < config.containment for _, outer in kept):
            kept.append((label, box))
    return kept[: config.max_characters] or [(None, (0.0, 0.0, float(width), float(height)))]


def body_type_of(label: str | None, config: SheetConfig | None = None) -> BodyType:
    """The reposing branch for a detector label. Whole-word, case-insensitive matching."""
    if label is None:
        return BodyType.UNKNOWN
    config = config or default_sheet_config()
    text = f" {re.sub(r'[^a-z]+', ' ', label.casefold())} "
    if any(f" {word} " in text for word in config.humanoid_words):
        return BodyType.HUMANOID
    if any(f" {word} " in text for word in config.creature_words):
        return BodyType.CREATURE
    return BodyType.OBJECT


def crop_bounds(box: Box) -> tuple[int, int, int, int]:
    """The whole-pixel rectangle ``box`` is cut out at: top-left rounded down, bottom-right up.

    One definition for the saved crop (``cutouts``) and for what keypoints are normalised against,
    so a joint lands on the same pixel of the crop that a consumer reconstructs the skeleton on.
    """
    return math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])


def keypoints_in(
    points: Sequence[tuple[float, float, float]], box: Box, threshold: float
) -> tuple[Keypoint, ...]:
    """ViTPose's 17 joints, as ``Keypoint``s normalised to the crop and clipped to 0..1.

    Normalised against ``crop_bounds(box)`` -- the saved crop -- not the raw float box, which differs
    by up to a pixel on each side for a fractional detection. Positional: index 3 is the same joint
    on every character (``interface.Keypoint``). A joint below ``threshold`` is kept with
    ``visible`` False, never dropped, and so is one whose coordinates or score are not finite (a
    NaN would otherwise clip to the crop's corner and be reported as a seen joint).
    """
    if len(points) != len(KEYPOINT_NAMES):
        raise ValueError(f"expected {len(KEYPOINT_NAMES)} COCO joints, got {len(points)}")
    if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
        raise ValueError(f"box {box} has no area")
    x0, y0, x1, y1 = crop_bounds(box)
    width, height = x1 - x0, y1 - y0

    def joint(name: str, x: float, y: float, score: float) -> Keypoint:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(score)):
            return Keypoint(name=name, x=0.0, y=0.0, visible=False)
        return Keypoint(
            name=name,
            x=float(min(1.0, max(0.0, (x - x0) / width))),
            y=float(min(1.0, max(0.0, (y - y0) / height))),
            visible=bool(score >= threshold),
        )

    return tuple(joint(name, x, y, score) for name, (x, y, score) in zip(KEYPOINT_NAMES, points, strict=True))


def cutouts(image: RGB, mask: Mask, box: Box) -> tuple[RGB, Mask]:
    """The character's pixels and mask within its box: paper outside the mask, as array slices.

    Not a preprocessing step for the recogniser -- the recogniser scores the box through
    ``prepare_drawing`` -- but the child's own lines kept for KAN-42 and the parent's
    confirmation screen. Outside the mask is set to white paper, not transparency, so the crop
    reads the same everywhere it is shown.

    A mask with no pixels in it -- SAM found nothing in the box -- is not a cut-out: whitening
    everything outside it would erase the child's lines. The unmasked box is returned instead, with
    a mask that covers the whole box (the honest "no cut-out was made").
    """
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match the drawing {image.shape[:2]}")
    x0, y0, x1, y1 = crop_bounds(box)
    region, inside = image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    if region.size == 0:
        raise ValueError(f"box {box} is outside the drawing")
    if not inside.any():
        return region, np.ones_like(inside)
    region[~inside] = 255
    return region, inside


def encode_png(pixels: RGB | Mask) -> bytes:
    """A crop or a mask as PNG. Greyscale masks need nothing but `render.to_png`; colour needs Pillow."""
    if pixels.dtype == np.bool_:
        return to_png(np.where(pixels, 255, 0).astype(np.uint8))
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def decode(image: bytes, background: int) -> RGB:
    """The drawing as an RGB array: decoded and composited exactly as the recogniser does it."""
    from PIL import Image

    array: RGB = np.asarray(flattened(Image.open(io.BytesIO(image)), background))
    return array


# ------------------------------------------------------------------------------- the recogniser


def _step(name: str, call: Callable[[], Any], character: int | None = None) -> Any:
    """Run one model call, naming the step (and the character) if it fails.

    A malformed model output otherwise surfaces as a bare ``zip() argument 2 is shorter`` with no
    hint which of four models, or which character, produced it. The message carries the step and
    the position of the character, never its pixels or any field of the sheet.
    """
    try:
        return call()
    except Exception as error:
        where = "" if character is None else f"character {character}: "
        raise RuntimeError(f"{where}{name} failed: {error}") from error


class CharacterSheetRecognizer:
    """A ``DrawingRecognizer`` that fills every field ``analyse`` owns. See the module docstring."""

    def __init__(
        self,
        recognizer: DrawingRecognizerModel | None = None,
        models: SheetModels | None = None,
        store: ArtifactStore | None = None,
        config: SheetConfig | None = None,
        *,
        decoder: Callable[[bytes], RGB] | None = None,
        encoder: Callable[[Any], bytes] = encode_png,
    ) -> None:
        self.config = config or default_sheet_config()
        self.recognizer = recognizer or DrawingRecognizerModel()
        self.models = models or HuggingFaceSheetModels(self.config)
        self.store = store or MemoryArtifactStore()
        self._decode = decoder or (lambda data: decode(data, self.recognizer.config.alpha_background))
        self._encode = encoder

    def analyse(self, image: bytes) -> DrawingAnalysis:
        if not image:
            raise ValueError("analyse needs the drawing's bytes; got none")
        pixels = self._decode(image)
        height, width = pixels.shape[:2]
        detections = _step("detect", lambda: self.models.detect(pixels))
        found = characters_in(detections, (width, height), self.config)
        sheets = [self._sheet(index, image, pixels, label, box) for index, (label, box) in enumerate(found)]
        # Most confident first, as `DrawingAnalysis.characters` and `.main` promise and `build_book`
        # relies on: detection order is by box area, and a large low-confidence background object
        # must not become the story's subject ahead of a smaller, confidently recognised character.
        # The sort is stable, so equal confidences keep their largest-first order.
        characters = tuple(sorted(sheets, key=lambda character: character.confidence, reverse=True))
        description = _step("caption", lambda: self.models.caption(pixels))
        return DrawingAnalysis(description=description.strip(), characters=characters)

    def _sheet(self, index: int, image: bytes, pixels: RGB, label: str | None, box: Box) -> Character:
        x0, y0, x1, y1 = box
        ranked = _step("score", lambda: self.recognizer.character(image, box=(x0, y0, x1 - x0, y1 - y0)), index)
        body_type = body_type_of(label, self.config)
        mask = _step("segment", lambda: self.models.segment(pixels, box), index)
        crop, inside = cutouts(pixels, mask, box)
        keypoints = (
            keypoints_in(
                _step("pose", lambda: self.models.pose(pixels, box), index), box, self.config.keypoint_threshold
            )
            if body_type is BodyType.HUMANOID
            else ()
        )
        return replace(
            ranked,
            body_type=body_type,
            crop_ref=self.store.put(self._encode(crop), "crop"),
            mask_ref=self.store.put(self._encode(inside), "mask"),
            keypoints=keypoints,
        )


class HuggingFaceSheetModels:
    """Florence-2, SAM 2.1 and ViTPose through ``transformers``, loaded on the first drawing.

    Needs the ``sheet`` extra (``drawing``'s packages plus ``torchvision``, which SAM 2's image
    processor imports; AGENTS.md -> Commands has the CUDA pair), and the three
    pinned checkpoints. bf16 on CUDA for Florence-2, fp32 elsewhere and for the two small models.
    Measured together on the RTX 4060: 1.29 GiB peak VRAM.
    """

    def __init__(self, config: SheetConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device
        self._loaded: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _models(self) -> dict[str, Any]:
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            import torch
            from transformers import (
                AutoProcessor,
                Florence2ForConditionalGeneration,
                Sam2Model,
                Sam2Processor,
                VitPoseForPoseEstimation,
            )

            c = self.config
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            half = torch.bfloat16 if device == "cuda" else torch.float32
            # Held as `Any`: transformers types `from_pretrained` on a concrete class as a decorated
            # call mypy cannot follow, and the return is used as a module either way.
            florence: Any = Florence2ForConditionalGeneration
            sam: Any = Sam2Model
            vitpose: Any = VitPoseForPoseEstimation
            loaded = {
                "device": device,
                "half": half,
                "fp": AutoProcessor.from_pretrained(c.florence, revision=c.florence_revision),
                "fm": florence.from_pretrained(c.florence, revision=c.florence_revision, dtype=half)
                .to(device)
                .eval(),
                "sp": Sam2Processor.from_pretrained(c.sam, revision=c.sam_revision),
                "sm": sam.from_pretrained(c.sam, revision=c.sam_revision).to(device).eval(),
                "pp": AutoProcessor.from_pretrained(c.vitpose, revision=c.vitpose_revision),
                "pm": vitpose.from_pretrained(c.vitpose, revision=c.vitpose_revision).to(device).eval(),
            }
            self._loaded = loaded
            return loaded

    def _florence(self, task: str, image: RGB) -> Any:
        import torch
        from PIL import Image

        m = self._models()
        picture = Image.fromarray(image)
        inputs = m["fp"](text=task, images=picture, return_tensors="pt").to(m["device"], m["half"])
        with torch.no_grad():
            tokens = m["fm"].generate(**inputs, max_new_tokens=128, num_beams=3, do_sample=False)
        text = m["fp"].batch_decode(tokens, skip_special_tokens=False)[0]
        return m["fp"].post_process_generation(text, task=task, image_size=picture.size)[task]

    def caption(self, image: RGB) -> str:
        return str(self._florence(self.config.caption_task, image))

    def detect(self, image: RGB) -> list[tuple[str, Box]]:
        found = self._florence(self.config.detect_task, image)
        return [
            (str(label), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for label, b in zip(found["labels"], found["bboxes"], strict=True)
        ]

    def segment(self, image: RGB, box: Box) -> Mask:
        import torch
        from PIL import Image

        m = self._models()
        inputs = m["sp"](images=Image.fromarray(image), input_boxes=[[list(box)]], return_tensors="pt").to(
            m["device"]
        )
        with torch.no_grad():
            out = m["sm"](**inputs, multimask_output=False)
        masks = m["sp"].post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])
        result: Mask = masks[0][0, 0].numpy() > 0
        return result

    def pose(self, image: RGB, box: Box) -> list[tuple[float, float, float]]:
        import torch
        from PIL import Image

        m = self._models()
        xywh = [[[box[0], box[1], box[2] - box[0], box[3] - box[1]]]]
        inputs = m["pp"](Image.fromarray(image), boxes=xywh, return_tensors="pt").to(m["device"])
        with torch.no_grad():
            out = m["pm"](**inputs)
        found = m["pp"].post_process_pose_estimation(out, boxes=xywh)[0][0]
        points = found["keypoints"].cpu().tolist()
        scores = found["scores"].cpu().tolist()
        return [(float(x), float(y), float(s)) for (x, y), s in zip(points, scores, strict=True)]
