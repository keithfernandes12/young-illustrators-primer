"""The real illustration implementation. Weights load lazily, never at import time.

Importing this module must stay cheap: the serving layer imports every area, and a test that
imports it must not pull a checkpoint off disk or out of GCS. ``torch`` and ``diffusers`` are
imported only inside ``DiffusersPipeline``, and only when the first page is generated; they come
from the ``illustration`` extra (``pip install -e ".[illustration]"``), which CI does not install.

This is the pretrained baseline (KAN-42, plan slice S7a): Stable Diffusion 1.5 conditioned on the
child's sketch through ControlNet-Scribble, one image per page, written to a local folder. Not here
yet: IP-Adapter on the characters, the style LoRA, reposing, the consistency score, GCS output and
the GPU container -- see `docs/plans/mvp-cloud-deploy.md`.

Generated images are not safety-checked here. Diffusers' built-in checker is turned off because it
replaces a flagged image with a black one and reports nothing upward; the check belongs in serving,
after `illustrate()`, so its verdict reaches `Book.safety` (KAN-94). Until that lands, run this only
on synthetic inputs.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import yaml
from PIL import Image

from primer.drawing.interface import Character
from primer.illustration.interface import Illustration
from primer.illustration.sketch import load_drawing, prepare_sketch

CONFIG_PATH = Path(__file__).parent / "configs" / "illustration_baseline.yaml"


@dataclass(frozen=True)
class IllustrationConfig:
    """The generation settings, read from `configs/illustration_baseline.yaml`."""

    base_model: str
    base_revision: str
    controlnet: str
    controlnet_revision: str
    seed: int
    steps: int
    guidance: float
    controlnet_scale: float
    size: int
    negative_prompt: str

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> IllustrationConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**{f.name: raw[f.name] for f in fields(cls)})


@dataclass(frozen=True)
class GenerationRequest:
    """Everything one diffusion call needs. The control image is already white-on-black."""

    prompt: str
    negative_prompt: str
    control_image: Image.Image
    seed: int
    steps: int
    guidance: float
    controlnet_scale: float
    size: int


class Pipeline(Protocol):
    """One diffusion call. The real one wraps diffusers; tests pass a recording fake."""

    def __call__(self, request: GenerationRequest) -> Image.Image: ...


class DiffusersPipeline:
    """SD 1.5 + ControlNet-Scribble through diffusers, loaded on the first call."""

    def __init__(self, config: IllustrationConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

        if self.device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        # Always the fp16 files: they are the ones pinned, measured and cached, and the full-precision
        # files are another ~5 GB that an offline run cannot fetch. On CPU they are upcast to fp32,
        # because fp16 arithmetic on CPU is slow or unsupported.
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        variant = "fp16"
        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet,
            revision=self.config.controlnet_revision,
            variant=variant,
            torch_dtype=dtype,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.base_model,
            revision=self.config.base_revision,
            variant=variant,
            controlnet=controlnet,
            torch_dtype=dtype,
            # See the module docstring: a silent black page is worse than no check at all.
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def __call__(self, request: GenerationRequest) -> Image.Image:
        import torch

        self.load()
        # A CPU generator gives the same starting noise for a seed on every device. The image can
        # still differ slightly between devices, because fp16 arithmetic differs.
        generator = torch.Generator("cpu").manual_seed(request.seed)
        result = self._pipe(
            request.prompt,
            image=request.control_image,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            controlnet_conditioning_scale=request.controlnet_scale,
            generator=generator,
            height=request.size,
            width=request.size,
        )
        image: Image.Image = result.images[0]
        return image


class IllustratorModel:
    """The production illustration implementation. Satisfies ``interface.Illustrator``.

    **Not ready for** ``build_book`` **yet.** ``build_book`` calls ``consistency()`` on every book,
    and here it raises until the stage 6 consistency score exists. Passing this model to
    ``build_book`` would fail every book even though ``isinstance(model, Illustrator)`` holds. Until
    then, use ``illustrate()`` directly, as ``smoke.py`` does.

    Each page is conditioned on the child's sketch: the in-memory ``sketch`` when serving passes it,
    otherwise the drawing re-read from ``drawing_ref``. The image is written under ``output_dir``
    with a random name -- never one derived from the drawing -- and returned as a ``file://`` URI.
    """

    def __init__(
        self,
        output_dir: Path | str,
        config: IllustrationConfig | None = None,
        *,
        pipeline: Pipeline | None = None,
        readers: dict[str, Callable[[str], bytes]] | None = None,
        new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or IllustrationConfig.from_yaml()
        self._pipeline = pipeline
        self._readers = readers
        self._new_id = new_id

    def pipeline(self) -> Pipeline:
        """The diffusion pipeline, created on first use so construction stays cheap."""
        if self._pipeline is None:
            self._pipeline = DiffusersPipeline(self.config)
        return self._pipeline

    def prompt(self, page_text: str, style: str, subject: str | None) -> str:
        parts = [page_text.strip()]
        if subject:
            parts.append(f"featuring a {subject}")
        parts += ["children's storybook illustration", f"{style} style", "bright colors"]
        return ", ".join(part for part in parts if part)

    def illustrate(
        self,
        page_index: int,
        page_text: str,
        style: str,
        subject: str | None = None,
        drawing_ref: str | None = None,
        sketch: bytes | None = None,
        characters: tuple[Character, ...] = (),
    ) -> Illustration:
        """One page, conditioned on the sketch. ``characters`` is unused until IP-Adapter lands."""
        if sketch is None:
            if not drawing_ref:
                raise ValueError("illustrate() needs the sketch, or a drawing_ref to read it from")
            sketch = load_drawing(drawing_ref, self._readers)

        request = GenerationRequest(
            prompt=self.prompt(page_text, style, subject),
            negative_prompt=self.config.negative_prompt,
            control_image=prepare_sketch(sketch, self.config.size),
            seed=self.config.seed,
            steps=self.config.steps,
            guidance=self.config.guidance,
            controlnet_scale=self.config.controlnet_scale,
            size=self.config.size,
        )
        image = self.pipeline()(request)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = (self.output_dir / f"page-{page_index:03d}-{self._new_id()}.png").resolve()
        image.save(path, format="PNG")
        return Illustration(
            page_index=page_index,
            image_ref=path.as_uri(),
            alt_text=page_text[:60],
            subject=subject,
            conditioned_on_sketch=True,
            identity=None,  # no IP-Adapter yet, so no page is conditioned on a character identity
        )

    def consistency(self, illustrations: list[Illustration], characters: tuple[Character, ...] = ()) -> float:
        # Raising, not returning a placeholder: a made-up score would pass the gate silently.
        raise NotImplementedError(
            "the consistency score is stage 6 and not built yet; do not pass this model to build_book"
        )
