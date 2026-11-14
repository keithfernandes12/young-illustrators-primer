"""IllustratorModel: the sketch reaches the diffusion call, and a page reference comes back.

A recording pipeline stands in for diffusers, so nothing here loads weights or reaches the network.
"""

import subprocess
import sys
import types
from pathlib import Path
from urllib.parse import urlparse

import pytest
from PIL import Image

from primer.drawing.interface import Character
from primer.illustration.interface import Illustration, Illustrator
from primer.illustration.model import (
    DiffusersPipeline,
    GenerationRequest,
    IllustrationConfig,
    IllustratorModel,
)
from primer.illustration.sketch import prepare_sketch


class RecordingPipeline:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def __call__(self, request: GenerationRequest) -> Image.Image:
        self.requests.append(request)
        return Image.new("RGB", (request.size, request.size), (200, 220, 255))


@pytest.fixture
def pipeline() -> RecordingPipeline:
    return RecordingPipeline()


@pytest.fixture
def model(tmp_path, pipeline) -> IllustratorModel:
    return IllustratorModel(tmp_path / "pages", pipeline=pipeline, new_id=lambda: "fixed")


def _control_pixels(request: GenerationRequest) -> bytes:
    return request.control_image.convert("L").tobytes()


def test_the_model_satisfies_the_illustrator_contract(model):
    assert isinstance(model, Illustrator)


def test_the_in_memory_sketch_conditions_the_generation(model, pipeline, child_style_sketch):
    page = model.illustrate(0, "a house on a cloud", "crayon", "house", sketch=child_style_sketch)

    [request] = pipeline.requests
    expected = prepare_sketch(child_style_sketch, model.config.size)
    assert _control_pixels(request) == expected.convert("L").tobytes()
    assert page.conditioned_on_sketch


def test_without_bytes_the_sketch_is_read_from_the_drawing_ref(tmp_path, model, pipeline, child_style_sketch):
    upload = tmp_path / "upload.png"
    upload.write_bytes(child_style_sketch)

    model.illustrate(0, "a house on a cloud", "crayon", "house", drawing_ref=upload.as_uri())

    [request] = pipeline.requests
    assert _control_pixels(request) == prepare_sketch(child_style_sketch, 512).convert("L").tobytes()


def test_the_in_memory_sketch_wins_over_re_reading_the_ref(tmp_path, pipeline, child_style_sketch):
    """`drawing_ref` is the durable handle for the same image; it is not read when bytes are here."""

    def must_not_read(ref: str) -> bytes:
        raise AssertionError("the drawing was re-read although the bytes were passed")

    model = IllustratorModel(
        tmp_path, pipeline=pipeline, readers={"gs": must_not_read, "file": must_not_read}
    )
    model.illustrate(0, "p", "crayon", drawing_ref="gs://drawings/d1.png", sketch=child_style_sketch)
    assert len(pipeline.requests) == 1


def test_a_page_with_neither_sketch_nor_ref_is_refused(model, pipeline):
    with pytest.raises(ValueError, match="sketch"):
        model.illustrate(0, "a page", "crayon")
    assert pipeline.requests == [], "nothing is generated without the child's drawing"


def test_the_page_reference_points_at_the_written_image(tmp_path, model, child_style_sketch):
    page = model.illustrate(3, "a house on a cloud", "crayon", "house", sketch=child_style_sketch)

    assert isinstance(page, Illustration)
    assert page.page_index == 3
    parsed = urlparse(page.image_ref)
    assert parsed.scheme == "file"
    written = Path(parsed.path)
    assert written.parent == (tmp_path / "pages").resolve()
    assert written.name == "page-003-fixed.png"
    with Image.open(written) as image:
        assert image.size == (512, 512)


def test_the_page_is_not_named_after_the_drawing(tmp_path, model, child_style_sketch):
    """A child's drawing must not leak into an object name, even through its file name."""
    upload = tmp_path / "zip-the-dragon-by-maya.png"
    upload.write_bytes(child_style_sketch)

    page = model.illustrate(0, "p", "crayon", drawing_ref=upload.as_uri())
    assert "zip" not in page.image_ref and "maya" not in page.image_ref


def test_the_baseline_settings_reach_the_diffusion_call(model, pipeline, child_style_sketch):
    model.illustrate(0, "a house on a cloud", "soft watercolor", "house", sketch=child_style_sketch)

    [request] = pipeline.requests
    config = IllustrationConfig.from_yaml()
    assert (request.seed, request.steps, request.guidance, request.controlnet_scale, request.size) == (
        config.seed,
        config.steps,
        config.guidance,
        config.controlnet_scale,
        config.size,
    )
    assert "a house on a cloud" in request.prompt
    assert "featuring a house" in request.prompt
    assert "soft watercolor style" in request.prompt
    assert request.negative_prompt == config.negative_prompt


def test_a_page_without_a_recognised_subject_still_gets_a_prompt(model, pipeline, child_style_sketch):
    model.illustrate(0, "a sunny day", "crayon", None, sketch=child_style_sketch)
    assert "None" not in pipeline.requests[0].prompt


def test_no_identity_is_claimed_without_ip_adapter(model, child_style_sketch):
    """Characters are accepted but not conditioned on yet, so the page must not say otherwise."""
    cat = Character(label="cat", confidence=1.0, embedding=(0.1, 0.2))
    page = model.illustrate(0, "p", "crayon", "cat", sketch=child_style_sketch, characters=(cat,))
    assert page.identity is None


def test_the_config_pins_the_measured_model_revisions():
    config = IllustrationConfig.from_yaml()
    assert config.base_model == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert config.controlnet == "lllyasviel/control_v11p_sd15_scribble"
    assert len(config.base_revision) == 40 and len(config.controlnet_revision) == 40


def test_importing_the_model_loads_no_ml_framework():
    """Serving imports every area; torch and diffusers must wait until a page is generated.

    The frameworks are blocked outright rather than looked for afterwards: CI does not install them,
    so "not in sys.modules" would hold there even if model.py imported torch at the top.
    """
    code = (
        "import sys\n"
        "for name in ('torch', 'diffusers', 'transformers'):\n"
        "    sys.modules[name] = None  # any import of it now raises ImportError\n"
        "import primer.illustration.model, primer.illustration.smoke\n"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"importing the area pulled in an ML framework:\n{result.stderr}"


class _Recorder:
    """Stands in for a diffusers class: records `from_pretrained` calls, returns a fake pipeline."""

    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls

    def from_pretrained(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return _FakeDiffusersPipe()

    def from_config(self, config):
        return "scheduler"


class _FakeDiffusersPipe:
    scheduler = types.SimpleNamespace(config={})

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, **kwargs):
        pass


@pytest.fixture
def stub_frameworks(monkeypatch):
    """Fake `torch` and `diffusers` modules, so `DiffusersPipeline.load()` runs with no weights."""
    calls: list[tuple[str, dict]] = []
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "float16", "float32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    diffusers = types.ModuleType("diffusers")
    recorder = _Recorder(calls)
    diffusers.ControlNetModel = recorder
    diffusers.StableDiffusionControlNetPipeline = recorder
    diffusers.UniPCMultistepScheduler = recorder
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return calls


@pytest.mark.parametrize("device,dtype", [("cpu", "float32"), ("mps", "float16"), ("cuda", "float16")])
def test_every_device_loads_the_pinned_fp16_files(stub_frameworks, device, dtype):
    """Only the fp16 files are pinned and cached; a CPU run asking for fp32 files fails offline."""
    config = IllustrationConfig.from_yaml()
    DiffusersPipeline(config, device=device).load()

    controlnet, base = stub_frameworks
    assert controlnet[0] == config.controlnet and base[0] == config.base_model
    for _, kwargs in (controlnet, base):
        assert kwargs["variant"] == "fp16"
        assert kwargs["torch_dtype"] == dtype
    assert controlnet[1]["revision"] == config.controlnet_revision
    assert base[1]["revision"] == config.base_revision
    assert base[1]["safety_checker"] is None, "see model.py: the check belongs in serving (KAN-94)"


def test_without_a_gpu_the_pipeline_falls_back_to_cpu(stub_frameworks):
    pipeline = DiffusersPipeline(IllustrationConfig.from_yaml())
    pipeline.load()
    assert pipeline.device == "cpu"


def test_build_book_cannot_use_the_model_until_consistency_exists(model):
    """build_book calls consistency() on every book; until stage 6 lands it must fail loudly."""
    with pytest.raises(NotImplementedError, match="build_book"):
        model.consistency([])
