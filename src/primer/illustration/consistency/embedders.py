"""Pretrained image embedders the consistency score can run on. Weights load on first use.

Install with ``pip install -e ".[consistency]"``. Importing this module needs neither torch nor
transformers; they are imported the first time a model actually embeds something, so serving and the
tests can import the area without pulling a checkpoint.

KAN-38's recogniser embeds drawings with SigLIP 2, a different space from these, so its
``Character.embedding`` is a latency cache key, not an input to this score (KAN-42, KAN-48).
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from primer.illustration.consistency.scorer import ImageEmbedder, normalise

#: Candidate name -> (family, Hugging Face model id). `experiment.py` compares these.
MODELS: dict[str, tuple[str, str]] = {
    "clip-vit-b32": ("clip", "openai/clip-vit-base-patch32"),
    "clip-vit-l14": ("clip", "openai/clip-vit-large-patch14"),
    # DINOv2 is the other half of the KAN-48 benchmark: self-supervised, keyed on shape and texture
    # rather than on what a caption would say, which is closer to "is this the same character".
    "dinov2-small": ("dino", "facebook/dinov2-small"),
}


def decode(image: bytes) -> Any:
    """Encoded bytes -> an RGB PIL image."""
    from PIL import Image

    with Image.open(io.BytesIO(image)) as opened:
        return opened.convert("RGB")


class HuggingFaceEmbedder:
    """CLIP image tower or DINOv2 from Hugging Face. Satisfies ``scorer.ImageEmbedder``."""

    def __init__(self, name: str, revision: str = "main", batch_size: int = 32, device: str | None = None):
        if name not in MODELS:
            raise ValueError(f"unknown embedder {name!r}; choose from {sorted(MODELS)}")
        self.name = name
        self.family, self.model_id = MODELS[name]
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._resolved: str | None = None

    @property
    def resolved_revision(self) -> str | None:
        """The commit ``revision`` pointed at when the weights loaded, for the model card."""
        if self._model is not None and self._resolved is None:
            from huggingface_hub import try_to_load_from_cache

            # The local cache stores each download under snapshots/<commit>/, so no network is needed.
            cached = try_to_load_from_cache(self.model_id, "config.json", revision=self.revision)
            if isinstance(cached, str):
                self._resolved = Path(cached).parent.name
        return self._resolved

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_cls: Any = CLIPModel if self.family == "clip" else AutoModel
        self._model = model_cls.from_pretrained(self.model_id, revision=self.revision).to(self.device).eval()
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, revision=self.revision)

    def embed(self, images: Sequence[bytes]) -> np.ndarray:
        import torch

        if self._model is None:
            self._load()
        rows: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = [decode(image) for image in images[start:start + self.batch_size]]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                if self.family == "clip":
                    features = self._model.get_image_features(**inputs)
                    # transformers 4.x returns the projected tensor; 5.x wraps it as `pooler_output`.
                    features = getattr(features, "pooler_output", features)
                else:
                    features = self._model(**inputs).pooler_output
            rows.append(features.float().cpu().numpy())
        return normalise(np.concatenate(rows)) if rows else np.zeros((0, 0))


def load_embedder(name: str, **options: Any) -> ImageEmbedder:
    """The embedder for a candidate name in ``MODELS``."""
    return HuggingFaceEmbedder(name, **options)
