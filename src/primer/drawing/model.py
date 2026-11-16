"""The real drawing implementation. Weights load lazily, never at import time.

Importing this module must stay cheap: the serving layer imports every area, and a test that
imports it must not pull a checkpoint off disk or out of GCS. ``torch`` and ``transformers`` are
imported only inside ``SiglipEncoder``, and only when the first drawing is analysed; they come from
the ``drawing`` extra (``pip install -e ".[drawing]"``), which CI does not install.

This is the pretrained baseline (KAN-38, plan slice S6a): SigLIP 2's image tower scores the 345
Quick, Draw! categories zero-shot, filling one ``Character`` with ``label``, ``confidence``, the
ranked ``candidates`` that ``drawing_top5_acc`` is measured from, and an ``embedding`` from the same
tower. One character per drawing: the whole image is the subject.

Not here yet, and deliberately so -- the per-drawing ``description`` (a separate captioning VLM),
multi-character detection, ``body_type``, ``crop_ref``/``mask_ref`` and ``keypoints`` are S6e; the
fine-tune is S6d; S6c measured ``drawing_top5_acc`` over S6b's rendered split (see CARD.md). Those
fields keep their contract defaults here rather than being guessed, because a guessed
``body_type`` sends KAN-42 down the wrong reposing branch silently.

**Zero-shot is the floor, not the answer.** `docs/briefs/drawing-recognition-model-candidates.md`
collects the sketch literature: plain CLIP scores ~43% top-1 on Quick, Draw!, and fine-tuning is
what closes the gap to the 0.85 top-5 gate. The point of this slice is the wiring and the number
S6d has to beat.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from primer.drawing.interface import Candidate, Character, DrawingAnalysis
from primer.drawing.preprocess import PrepareConfig, default_prepare_config, prepare_drawing

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_PATH = CONFIG_DIR / "drawing_baseline.yaml"

#: How many decimals a confidence and an embedding component keep. Rounding is monotonic, so a
#: ranked list stays ranked.
#:
#: The embedding is rounded so that `Character.identity_key`, which hashes it, is not changed by
#: float noise that means nothing. The confidence is rounded for a weaker reason -- a character
#: sheet is serialised, compared and read by people, and seventeen significant digits of a sigmoid
#: are noise in all three. Four decimals keeps one or two significant figures on the tail of a
#: zero-shot ranking: measured across two drawings the fifth candidate scored 0.0023 and 0.00045,
#: so on a weak drawing the tail rounds coarsely and candidates can tie. Ties are legal, and the
#: gate reads ranking rather than magnitude -- but anything that comes to depend on the tail's
#: precision should raise this rather than assume it.
#:
#: **It does not make the key portable across dtypes.** The tower runs in bf16 on CUDA and fp32 on
#: CPU, and near a unit vector's typical component (1/sqrt(768) is about 0.0361) one bf16 step is
#: 2.44e-4 -- 244 times this rounding step -- so the same drawing analysed on a GPU and
#: on a CPU keys to two different cache entries. No rounding fixes that; it is a difference in what
#: the tower computed, not noise on top of it. Anything that shares a KAN-71 cache between devices
#: has to say which dtype the key was taken at.
CONFIDENCE_DECIMALS = 4
EMBEDDING_DECIMALS = 6



def pooled(output: Any) -> Any:
    """The feature vectors out of ``get_text_features`` / ``get_image_features``.

    `transformers` 5.x returns a ``BaseModelOutputWithPooling`` from both, and the vectors are its
    ``pooler_output``; the shape in the model card's own example is a bare tensor. Treating the
    first as the second is an ``AttributeError`` at the first real call -- which is how this was
    found, by `smoke.py`, since the tests inject an `Encoder` and never reach this code.

    Written as a function so that the one version-dependent assumption in this file is in one
    place and can be tested without a checkpoint.
    """
    return getattr(output, "pooler_output", output)


def l2_normalised(features: Any, floor: float) -> Any:
    """``features`` scaled to unit length along the last axis.

    SigLIP's score is a dot product of unit vectors, so this is not cosmetic: skipping it leaves
    the logits scaled by each vector's magnitude and the sigmoid saturates.

    The norm is floored before dividing. ``Siglip2Model.forward`` does not bother, but a
    zero-magnitude row there is a NaN that propagates into every confidence, and a NaN ranks
    arbitrarily rather than loudly -- `analyse` would return a mislabelled character that
    `Character.__post_init__` does not reject, because NaN compares equal to itself by identity.

    ``floor`` is a parameter rather than a constant because the right value depends on the dtype
    the tower is running in. float16's smallest normal is about 6e-5, so a fixed 1e-12 clamps to
    0.0 there and the 0/0 happens regardless -- the guard is a no-op in the dtype it most needs to
    work in. bfloat16, which the CUDA path now selects, keeps fp32's exponent range and would be
    fine with a constant; the parameter is what stops that from being an accident of the dtype
    anyone happens to pick. Call sites pass ``torch.finfo(features.dtype).tiny``.
    """
    return features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(floor)


def flattened(drawing: Any, background: int) -> Any:
    """``drawing`` composited onto an opaque ``background``-grey canvas, in RGB.

    ``convert("RGB")`` on an image with alpha does not composite it, it **drops** it. That is not a
    rounding error, it is the whole drawing: a sketch exported from a canvas app is black ink on a
    transparent background -- RGBA (0, 0, 0, 0) where there is no ink and (0, 0, 0, 255) where
    there is -- so dropping alpha leaves every pixel black. Measured on a four-pixel fixture, all
    three transparent shapes below convert to a uniform ``(0, 0, 0)`` square. Nothing raises; the
    tower scores a black square and returns a confident, wrong label.

    Palette and greyscale images carry transparency in ``info`` rather than in a channel, which is
    why the mode alone does not decide this -- a ``P``-mode PNG reports mode ``"P"`` and still has
    a transparent background.

    ``background`` comes from the config rather than from here because it changes every score: the
    tower sees the composited pixels, so flattening the same drawing onto white and onto black
    produces two different rankings.
    """
    from PIL import Image

    transparent = drawing.mode in ("RGBA", "LA", "PA") or "transparency" in drawing.info
    if not transparent:
        return drawing.convert("RGB")
    canvas = Image.new("RGBA", drawing.size, (background, background, background, 255))
    canvas.alpha_composite(drawing.convert("RGBA"))
    return canvas.convert("RGB")


@dataclass(frozen=True)
class DrawingConfig:
    """The recognition settings, read from `configs/drawing_baseline.yaml`.

    Every field here changes a score, which is why none of them lives at a call site: a measured
    `drawing_top5_acc` is only comparable with another one taken at the same prompt and vocabulary.
    """

    name: str
    encoder: str
    encoder_revision: str
    encoder_license: str
    labels_file: str
    prompt_template: str
    text_max_length: int
    text_padding: str
    top_k: int
    #: The grey level, 0-255, that a transparent drawing is composited onto before it reaches the
    #: tower. See `flattened`.
    alpha_background: int
    #: Where `labels_file` is resolved from: the folder the config itself came from, not a module
    #: global. A config loaded from elsewhere brings its own vocabulary rather than silently
    #: scoring against the shipped one, which `train.py --config` will do in S6d.
    config_dir: Path = CONFIG_DIR

    def __post_init__(self) -> None:
        # Checked here rather than in `DrawingRecognizerModel.__init__` with the rest, because
        # `SiglipEncoder` reads this field and can be built without a model around it -- and a bad
        # value is not an error there, it is a silently different image. The other settings need
        # the labels file read off disk, which is why they are checked where that happens.
        # `bool` is rejected explicitly: YAML reads `true` as 1, which would composite onto
        # near-black and look like a contrast problem rather than a config one.
        if isinstance(self.alpha_background, bool) or not isinstance(self.alpha_background, int):
            raise TypeError(
                f"alpha_background must be a whole number 0-255, not {self.alpha_background!r}"
            )
        if not 0 <= self.alpha_background <= 255:
            raise ValueError(
                f"alpha_background must be a grey level 0-255, not {self.alpha_background}"
            )

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> DrawingConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings = {f.name: raw[f.name] for f in fields(cls) if f.name != "config_dir"}
        return cls(**settings, config_dir=path.parent)

    def labels(self) -> tuple[str, ...]:
        """The label vocabulary, in the order the file lists it.

        Comment and blank lines are skipped so the file can carry its own provenance. The order is
        the bucket listing order the ingestion counts in; keeping it means a label index means the
        same thing here and in `primer.data.sources.quickdraw`.
        """
        text = (self.config_dir / self.labels_file).read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines()]
        return tuple(line for line in lines if line and not line.startswith("#"))

    def prompts(self) -> tuple[str, ...]:
        """One text prompt per label, in label order, through the configured `prompt_template`."""
        return tuple(self.prompt_template.format(label=label) for label in self.labels())


@dataclass(frozen=True)
class Encoding:
    """One drawing through the tower: a confidence per prompt, in prompt order, plus the embedding.

    Both come out of a single pass so the image is encoded once. ``confidences`` is positional --
    index *i* is the prompt at index *i* -- because pairing them by label here would make a
    reordering invisible.
    """

    confidences: tuple[float, ...]
    embedding: tuple[float, ...]


class Encoder(Protocol):
    """Scores a drawing against text prompts. The real one wraps SigLIP 2; tests pass a fake.

    ``box`` -- ``(x, y, width, height)`` -- restricts scoring to one character of the drawing
    (S6e). ``None`` scores the whole drawing.
    """

    def encode(self, image: bytes, prompts: Sequence[str], *, box: object = None) -> Encoding: ...


class SiglipEncoder:
    """SigLIP 2's image and text towers through ``transformers``, loaded on the first call.

    The text side is encoded once and kept: the 345 prompts do not change between drawings, and
    re-encoding them per image would dominate the per-drawing cost that KAN-71 budgets.

    Confidences are **sigmoid of the image-text logits, not a softmax**. SigLIP is trained with a
    pairwise sigmoid loss, so each score is "does this label fit", independent of the others; they
    do not sum to 1 and must not be read as a distribution. Ranking -- which is all
    `drawing_top5_acc` needs -- is unaffected, but anything that later thresholds a confidence
    should know which of the two it is holding.
    """

    def __init__(
        self, config: DrawingConfig, device: str | None = None, prepare: PrepareConfig | None = None
    ) -> None:
        self.config = config
        self.device = device
        #: How a decoded drawing becomes the tower's input (`preprocess.prepare_drawing`). None is
        #: the shipped `drawing_preprocess.yaml`; S6c2 passes its variants here.
        self.prepare = prepare
        self._model: Any = None
        self._processor: Any = None
        self._dtype: Any = None
        #: (prompts, their features) or None. One field rather than two so the pair is swapped in
        #: a single assignment: with two, a reader between them sees new features under the old
        #: prompts, which is stale scoring with no error anywhere.
        self._text_cache: tuple[tuple[str, ...], Any] | None = None
        #: Serialises `load`. The serving layer runs sync endpoints in a threadpool, so two
        #: requests arriving before the first finishes loading are the normal case on a cold
        #: start, not a rare interleaving.
        self._lock = threading.Lock()

    @property
    def dtype_name(self) -> str | None:
        """The tower's dtype as text (``float32``, ``bfloat16``), or None before it is loaded."""
        return None if self._dtype is None else str(self._dtype).removeprefix("torch.")

    def _check_resolution(self, processor: Any, prepare: PrepareConfig | None = None) -> None:
        """Refuse a processor whose input size is not the square `prepare_drawing` produces.

        `prepare_drawing` is the only resampling between a decoded drawing and the tower only while
        `drawing_render.yaml`'s `size` -- marked provisional there -- equals the checkpoint's own
        resolution. If they part, the processor quietly resizes every prepared drawing a second
        time, squashing nothing but resampling everything, and a measured number stops being
        comparable with the rendered split it was meant to match. Checked once, at load, and again
        for a per-call ``prepare``, whose ``render.size`` can differ from the encoder's own.
        """
        expected = (prepare or self.prepare or default_prepare_config()).size
        size = getattr(getattr(processor, "image_processor", None), "size", None)
        if isinstance(size, dict):
            height, width = size.get("height"), size.get("width")
        else:
            height, width = getattr(size, "height", None), getattr(size, "width", None)
        if height is None or width is None:
            # A shortest-edge (CLIP, DINOv2) or patch-budget (SigLIP 2 NaFlex) processor has no
            # fixed square, so there is no resolution to change `size` to -- the fix is different.
            raise ValueError(
                f"{self.config.encoder}'s processor declares no fixed height x width (size: "
                f"{size!r}), but `prepare_drawing` hands it a {expected} x {expected} square; a "
                "checkpoint that resizes by shortest edge or patch budget needs this coupling "
                "rethought, not a new `size`"
            )
        if (height, width) != (expected, expected) and prepare is not None:
            raise ValueError(
                f"this call's `prepare` renders at {expected} x {expected}, but {self.config.encoder} "
                f"expects {width} x {height}; a variant must keep the checkpoint's resolution, or "
                "the processor resamples every prepared drawing a second time"
            )
        if (height, width) != (expected, expected):
            raise ValueError(
                f"{self.config.encoder} expects {width} x {height} input, but drawings are "
                f"prepared at {expected} x {expected}; change `size` in drawing_render.yaml to the "
                "checkpoint's resolution (and re-render the split) rather than resample twice"
            )

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        # Taken unconditionally rather than behind a `self._model is not None` fast path. The fast
        # path would be correct only if a reader that sees `_model` is guaranteed to see the
        # device and dtype written before it, which the GIL happens to give and the free-threaded
        # build does not. An uncontended lock costs nanoseconds against a forward pass.
        with self._lock:
            if self._model is not None:
                return
            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            # bf16 on CUDA, fp32 on CPU, where reduced precision is slow or unsupported. bf16
            # rather than fp16 because `AGENTS.md` names it for local GPU work and S6d will
            # fine-tune in it, so inference and training agree -- and because it keeps fp32's
            # exponent range, which is the half of the precision question that produces NaN rather
            # than drift. Measured, the weights are 1.398 GB in fp32 and 0.699 GB in either half
            # format; what that costs in VRAM alongside activations has not been measured, and is
            # S6d's to find out.
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(
                self.config.encoder, revision=self.config.encoder_revision
            )
            self._check_resolution(processor)
            # `dtype`, not `torch_dtype`: the latter is deprecated in transformers 5 and warns on
            # every load.
            model = AutoModel.from_pretrained(
                self.config.encoder, revision=self.config.encoder_revision, dtype=dtype
            ).to(device)
            model.eval()
            # Published last, and only once it is on the device and out of training mode. A
            # half-initialised encoder reachable from another thread is the failure this ordering
            # exists to prevent, not just the duplicate checkpoint.
            self.device = device
            self._dtype = dtype
            self._processor = processor
            self._model = model

    def _encode_prompts(self, prompts: Sequence[str]) -> Any:
        """The text features for `prompts`, computed once and reused while they stay the same."""
        import torch

        wanted = tuple(prompts)
        cached = self._text_cache
        if cached is not None and cached[0] == wanted:
            return cached[1]
        # `input_ids` are integers, so only the device matters here, not the dtype.
        inputs = self._processor(
            text=list(wanted),
            padding=self.config.text_padding,
            max_length=self.config.text_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            raw = pooled(self._model.get_text_features(**inputs))
            features = l2_normalised(raw, torch.finfo(raw.dtype).tiny)
        self._text_cache = (wanted, features)
        return features

    def encode(
        self,
        image: bytes,
        prompts: Sequence[str],
        *,
        prepare: PrepareConfig | None = None,
        box: object = None,
    ) -> Encoding:
        """Score ``image`` against ``prompts``.

        ``prepare`` overrides the encoder's own settings for this call and ``box`` crops to an
        annotated character first; both go straight to ``prepare_drawing``. They exist for
        evaluation (S6c2's variants), so a variant is a setting of the one preprocessing path
        rather than a second path built beside it. The product passes neither.
        """
        import io

        import torch
        from PIL import Image

        self.load()
        if prepare is not None:
            self._check_resolution(self._processor, prepare)
        text_features = self._encode_prompts(prompts)
        # RGB: the tower expects three channels, and a child's drawing arrives as anything from a
        # 1-bit PNG to an RGBA screenshot. Composited rather than converted -- see `flattened`,
        # where dropping alpha is the difference between a drawing and a black square.
        drawing = flattened(Image.open(io.BytesIO(image)), self.config.alpha_background)
        # Decode and composite stop here; everything after them is `prepare_drawing`, the one
        # inference preprocessing path (KAN-31), which the training loader applies to rendered
        # canvases too. It hands the processor a square at the processor's own resolution, so the
        # processor's resize -- which would otherwise squash a non-square upload -- has nothing
        # left to do.
        pixels = prepare_drawing(np.asarray(drawing), prepare or self.prepare, box=box)
        # The dtype as well as the device. The processor always returns float32 `pixel_values`, and
        # `BatchFeature.to(device)` moves them without casting, so on CUDA -- where the tower is in
        # reduced precision -- the first matmul raises "mat1 and mat2 must have the same dtype".
        # The CPU path never sees it, which is why a CPU smoke run cannot prove the GPU one.
        inputs = self._processor(images=pixels, return_tensors="pt").to(self.device, self._dtype)
        with torch.no_grad():
            raw = pooled(self._model.get_image_features(**inputs))
            normalised = l2_normalised(raw, torch.finfo(raw.dtype).tiny)
            # The same arithmetic `Siglip2Model.forward` does for `logits_per_image`: unit vectors,
            # dot product, then the learned scale and bias. Done here rather than by calling
            # `forward` so the 345 text prompts are encoded once instead of once per drawing.
            logits = (
                normalised @ text_features.T * self._model.logit_scale.exp() + self._model.logit_bias
            )
            confidences = torch.sigmoid(logits)[0]
        # `tolist()` once per vector, not `float()` once per element. Each `float()` on a CUDA
        # tensor is a device-to-host copy that synchronises the stream, so the old form made
        # 345 + 768 = 1113 of them per drawing for the shipped checkpoint -- on the warm path that
        # KAN-71 budgets, where the forward pass itself is 0.12 s. `tolist()` moves each vector in
        # one transfer. The numbers are identical: both widen the same elements to Python floats.
        return Encoding(
            confidences=tuple(confidences.tolist()),
            # The normalised vector, so two drawings of the same character land in the same place
            # whatever their contrast. This is the KAN-71 cache key, never the consistency signal:
            # KAN-48 measures consistency in DINOv2 + CLIP space, not this one.
            embedding=tuple(normalised[0].tolist()),
        )


class DrawingRecognizerModel:
    """The production drawing implementation. Satisfies ``interface.DrawingRecognizer``.

    Construction is free: no weights, no ``torch``. The encoder is built on the first ``analyse``
    call unless one is injected, which is how the tests run with no checkpoint and no network.
    """

    def __init__(self, config: DrawingConfig | None = None, encoder: Encoder | None = None) -> None:
        self.config = config or DrawingConfig.from_yaml()
        self._encoder = encoder
        #: Serialises the lazy build below. Locking `SiglipEncoder.load` alone would not be
        #: enough: two threads racing this property build two *different* encoders, and each one
        #: then loads its own checkpoint past its own lock. That is the duplicated VRAM, and it
        #: survives any amount of care inside `load`.
        self._encoder_lock = threading.Lock()
        # Read and checked here, not inside `analyse`. These are the failures a bad config causes,
        # and every one of them is cheaper to hit now than after `smoke.py` has pulled 1.6 GB of
        # weights off the network. Reading a few KB of text keeps construction effectively free.
        self._labels = self.config.labels()
        self._prompts = self.config.prompts()
        if not self._labels:
            raise ValueError(
                f"the vocabulary in {self.config.labels_file} is empty; "
                "there is nothing to score the drawing against"
            )
        if self.config.top_k < 1:
            # `0` passes the check below (`345 < 0` is false) and then dies on `candidates[0]`
            # inside `analyse` -- the same unreadable IndexError the empty-vocabulary guard above
            # exists to replace.
            raise ValueError(f"top_k must be at least 1, not {self.config.top_k}")
        if len(self._labels) < self.config.top_k:
            raise ValueError(
                f"{self.config.labels_file} has {len(self._labels)} labels but top_k is "
                f"{self.config.top_k}; drawing_top5_acc would be scored over fewer guesses than "
                "it reports"
            )
        if len(set(self._prompts)) != len(self._prompts):
            # Two labels sharing a prompt score identically, and the winner becomes whichever
            # sorts first. The usual cause is a `prompt_template` with no `{label}` in it, which
            # `str.format` accepts silently and which collapses all 345 prompts into one.
            raise ValueError(
                f"prompt_template {self.config.prompt_template!r} does not give every label a "
                "distinct prompt; a template without {label} makes every score identical"
            )

    @property
    def encoder(self) -> Encoder:
        with self._encoder_lock:
            if self._encoder is None:
                self._encoder = SiglipEncoder(self.config)
            return self._encoder

    def analyse(self, image: bytes) -> DrawingAnalysis:
        """Score the drawing against the whole vocabulary and return its one character.

        ``description`` stays empty and the structural fields keep their defaults: S6e fills them
        from different models. An empty string is honest here, while a label echoed into a sentence
        would read like a caption the story writer could use.
        """
        return DrawingAnalysis(characters=(self.character(image),))

    def character(self, image: bytes, box: Sequence[float] | None = None) -> Character:
        """Score one character -- the whole drawing, or the part inside ``box`` -- and rank it.

        ``box`` is ``(x, y, width, height)`` in the decoded drawing's pixels. It goes to
        ``prepare_drawing`` through the encoder, so a character cut from a drawing reaches the
        tower through the one preprocessing path (S6e's per-character sheets, KAN-31's rule).
        """
        if not image:
            raise ValueError("analyse needs the drawing's bytes; got none")

        encoding = self.encoder.encode(image, self._prompts, box=box)
        if len(encoding.confidences) != len(self._labels):
            raise ValueError(
                f"the encoder scored {len(encoding.confidences)} prompts for "
                f"{len(self._labels)} labels; confidences are positional, so a length mismatch "
                "would mislabel every candidate"
            )
        if not all(math.isfinite(confidence) for confidence in encoding.confidences):
            # NaN sorts arbitrarily and compares equal to itself by identity, so it survives
            # `Character.__post_init__` and comes back as a confidently wrong label rather than as
            # an error. Refusing it here is the difference between a crash and a bad number.
            raise ValueError(
                "the encoder returned a non-finite confidence; a NaN ranks arbitrarily and would "
                "be reported as a label rather than as a failure"
            )
        if not all(0.0 <= confidence <= 1.0 for confidence in encoding.confidences):
            # The `Encoder` seam promises sigmoid scores, and `Character.confidence` is described
            # as a probability in three places. Nothing else checks it: drop the `sigmoid` in
            # `SiglipEncoder.encode` and every ranking test still passes, while an unbounded logit
            # flows out under a name that says otherwise.
            raise ValueError(
                "the encoder returned a confidence outside 0-1; `Encoding.confidences` are "
                "sigmoid scores, and a raw logit here would be reported as a probability"
            )

        ranked = sorted(
            zip(self._labels, encoding.confidences, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        candidates = tuple(
            Candidate(label, round(confidence, CONFIDENCE_DECIMALS))
            for label, confidence in ranked[: self.config.top_k]
        )
        character = Character(
            label=candidates[0].label,
            confidence=candidates[0].confidence,
            embedding=tuple(round(value, EMBEDDING_DECIMALS) for value in encoding.embedding),
            candidates=candidates,
        )
        return character
