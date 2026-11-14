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
fine-tune is S6d; the measured ``drawing_top5_acc`` is S6c, which needs S6b's rendered split. Those
fields keep their contract defaults here rather than being guessed, because a guessed
``body_type`` sends KAN-42 down the wrong reposing branch silently.

**Zero-shot is the floor, not the answer.** `docs/briefs/drawing-recognition-model-candidates.md`
collects the sketch literature: plain CLIP scores ~43% top-1 on Quick, Draw!, and fine-tuning is
what closes the gap to the 0.85 top-5 gate. The point of this slice is the wiring and the number
S6d has to beat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import yaml

from primer.drawing.interface import Candidate, Character, DrawingAnalysis

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_PATH = CONFIG_DIR / "drawing_baseline.yaml"

#: How many decimals a confidence and an embedding component keep. Rounding is monotonic, so a
#: ranked list stays ranked; it exists so that `Character.identity_key`, which hashes the embedding,
#: is not changed by float noise that means nothing.
CONFIDENCE_DECIMALS = 4
EMBEDDING_DECIMALS = 6

#: The smallest norm `l2_normalised` will divide by. Guards the 0/0 that turns a whole encoding
#: into NaN; far below any real feature magnitude, so it never changes a genuine vector.
NORM_FLOOR = 1e-12


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


def l2_normalised(features: Any) -> Any:
    """``features`` scaled to unit length along the last axis.

    SigLIP's score is a dot product of unit vectors, so this is not cosmetic: skipping it leaves
    the logits scaled by each vector's magnitude and the sigmoid saturates.

    The norm is floored before dividing. ``Siglip2Model.forward`` does not bother, but a
    zero-magnitude row there is a NaN that propagates into every confidence, and a NaN ranks
    arbitrarily rather than loudly -- `analyse` would return a mislabelled character that
    `Character.__post_init__` does not reject, because NaN compares equal to itself by identity.
    An fp16 overflow on the CUDA path reaches the same place.
    """
    return features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(NORM_FLOOR)


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

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> DrawingConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**{f.name: raw[f.name] for f in fields(cls)})

    def labels(self) -> tuple[str, ...]:
        """The label vocabulary, in the order the file lists it.

        Comment and blank lines are skipped so the file can carry its own provenance. The order is
        the bucket listing order the ingestion counts in; keeping it means a label index means the
        same thing here and in `primer.data.sources.quickdraw`.
        """
        text = (CONFIG_DIR / self.labels_file).read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines()]
        return tuple(line for line in lines if line and not line.startswith("#"))

    def prompts(self) -> tuple[str, ...]:
        """One text prompt per label, in label order, as the checkpoint was trained to read them."""
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
    """Scores a drawing against text prompts. The real one wraps SigLIP 2; tests pass a fake."""

    def encode(self, image: bytes, prompts: Sequence[str]) -> Encoding: ...


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

    def __init__(self, config: DrawingConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._dtype: Any = None
        #: (prompts, their features) or None. One field rather than two so the pair is swapped in
        #: a single assignment: with two, a reader between them sees new features under the old
        #: prompts, which is stale scoring with no error anywhere.
        self._text_cache: tuple[tuple[str, ...], Any] | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp32 on CPU: fp16 arithmetic there is slow or unsupported. Measured, the weights are
        # 1.398 GB in fp32 and 0.699 GB in fp16; what that costs in VRAM alongside activations has
        # not been measured, and is S6d's to find out.
        self._dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            self.config.encoder, revision=self.config.encoder_revision
        )
        # `dtype`, not `torch_dtype`: the latter is deprecated in transformers 5 and warns on
        # every load.
        self._model = AutoModel.from_pretrained(
            self.config.encoder, revision=self.config.encoder_revision, dtype=self._dtype
        ).to(self.device)
        self._model.eval()

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
            features = l2_normalised(pooled(self._model.get_text_features(**inputs)))
        self._text_cache = (wanted, features)
        return features

    def encode(self, image: bytes, prompts: Sequence[str]) -> Encoding:
        import io

        import torch
        from PIL import Image

        self.load()
        text_features = self._encode_prompts(prompts)
        # RGB: the tower expects three channels, and a child's drawing arrives as anything from a
        # 1-bit PNG to an RGBA screenshot.
        drawing = Image.open(io.BytesIO(image)).convert("RGB")
        # The dtype as well as the device. The processor always returns float32 `pixel_values`, and
        # `BatchFeature.to(device)` moves them without casting, so on CUDA -- where the tower is
        # fp16 -- the first matmul raises "mat1 and mat2 must have the same dtype". The CPU path
        # never sees it, which is why a CPU smoke run cannot prove the GPU one.
        inputs = self._processor(images=drawing, return_tensors="pt").to(self.device, self._dtype)
        with torch.no_grad():
            normalised = l2_normalised(pooled(self._model.get_image_features(**inputs)))
            # The same arithmetic `Siglip2Model.forward` does for `logits_per_image`: unit vectors,
            # dot product, then the learned scale and bias. Done here rather than by calling
            # `forward` so the 345 text prompts are encoded once instead of once per drawing.
            logits = (
                normalised @ text_features.T * self._model.logit_scale.exp() + self._model.logit_bias
            )
            confidences = torch.sigmoid(logits)[0]
        return Encoding(
            confidences=tuple(float(value) for value in confidences),
            # The normalised vector, so two drawings of the same character land in the same place
            # whatever their contrast. This is the KAN-71 cache key, never the consistency signal:
            # KAN-48 measures consistency in DINOv2 + CLIP space, not this one.
            embedding=tuple(float(value) for value in normalised[0]),
        )


class DrawingRecognizerModel:
    """The production drawing implementation. Satisfies ``interface.DrawingRecognizer``.

    Construction is free: no weights, no ``torch``. The encoder is built on the first ``analyse``
    call unless one is injected, which is how the tests run with no checkpoint and no network.
    """

    def __init__(self, config: DrawingConfig | None = None, encoder: Encoder | None = None) -> None:
        self.config = config or DrawingConfig.from_yaml()
        self._encoder = encoder
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
        if self._encoder is None:
            self._encoder = SiglipEncoder(self.config)
        return self._encoder

    def analyse(self, image: bytes) -> DrawingAnalysis:
        """Score the drawing against the whole vocabulary and return its one character.

        ``description`` stays empty and the structural fields keep their defaults: S6e fills them
        from different models. An empty string is honest here, while a label echoed into a sentence
        would read like a caption the story writer could use.
        """
        if not image:
            raise ValueError("analyse needs the drawing's bytes; got none")

        encoding = self.encoder.encode(image, self._prompts)
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
        return DrawingAnalysis(characters=(character,))
