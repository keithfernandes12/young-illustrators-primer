"""`SiglipEncoder` itself: the arithmetic and the text cache, against a stub tower.

Everything else in `tests/drawing/` injects an `Encoder` and so never runs this class -- which is
how the `pooled()` bug reached a real run (`get_text_features` returns a model output, not a
tensor) and how the CUDA dtype bug is still only reasoned about rather than executed. These tests
close the arithmetic half of that gap.

They need `torch` for the tensor operations and are skipped where it is absent, which is every CI
run: `torch` is in the `drawing` extra, deliberately outside `dependencies`. Nothing here reaches
the network or loads a checkpoint -- the "model" is forty lines of `nn.Module` with fixed weights,
so the test is hermetic wherever it does run.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from primer.drawing.model import SiglipEncoder

torch = pytest.importorskip("torch", reason="torch comes from the `drawing` extra; CI omits it")
Image = pytest.importorskip("PIL.Image", reason="pillow comes from the `drawing` extra too")
# `load()`'s test stubs the factories on this module, so it has to be importable here too --
# without this the file errors rather than skipping wherever only part of the extra is installed.
pytest.importorskip("transformers", reason="transformers comes from the `drawing` extra too")

DIMENSIONS = 4


@pytest.fixture
def png() -> bytes:
    """A real, decodable image, because `encode` opens the bytes before encoding them.

    The fake-encoder tests elsewhere can use arbitrary bytes; these run the real `encode`, which
    does `Image.open(...).convert("RGB")`. Generated rather than committed, and an eight-pixel
    square rather than a drawing: the stub tower ignores the pixels entirely.
    """
    import io

    buffer = io.BytesIO()
    Image.new("L", (8, 8), 255).save(buffer, format="PNG")
    return buffer.getvalue()


class Pooling:
    """The `BaseModelOutputWithPooling` shape `transformers` 5.x returns from both feature calls."""

    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class StubTower(torch.nn.Module):
    """Returns fixed features so the expected confidences can be computed by hand.

    Text features are one-hot per prompt, image features are a weighted mix, so the dot product of
    image with prompt *i* is simply ``weights[i]`` -- which makes the ranking predictable without
    reimplementing the arithmetic in the test.

    ``logit_scale`` and ``logit_bias`` are deliberately **not** the identities. With `exp(0)` and
    `0.0` a test cannot tell multiplying by the scale from dividing by it, or adding the bias from
    subtracting it, because both give the same numbers. `log(2)` and `-1.0` separate all four.

    It records the dtype of the pixels it was handed rather than asserting on it, so the assertion
    lives in the test that cares.
    """

    def __init__(self, weights, dtype=torch.float32):
        super().__init__()
        self.weights = weights
        self.dtype_out = dtype
        self.logit_scale = torch.nn.Parameter(torch.tensor([math.log(2.0)]))
        self.logit_bias = torch.nn.Parameter(torch.tensor([-1.0]))
        self.text_calls = 0
        self.seen_dtype: Any = None
        self.evaluated = False

    def get_text_features(self, **inputs):
        self.text_calls += 1
        count = inputs["input_ids"].shape[0]
        # Five units long, not one. With unit rows the text-side L2 normalisation is a no-op and
        # deleting it changes nothing in this file -- while against the real checkpoint the text
        # norms are 18-26, so skipping it scales every logit by about twenty and all 345 sigmoids
        # saturate to 1.0, making the winner whichever label sorts first.
        return Pooling(torch.eye(count, DIMENSIONS, dtype=self.dtype_out) * 5.0)

    def get_image_features(self, **inputs):
        self.seen_dtype = inputs["pixel_values"].dtype
        return Pooling(torch.tensor([self.weights], dtype=self.dtype_out))

    def to(self, *args, **kwargs):
        # Overrides `nn.Module.to` so that `load()` can be tested against a CUDA device on a
        # CPU-only torch: the real one would raise before the dtype could be asserted.
        self.moved_to = args
        return self

    def eval(self):
        self.evaluated = True
        return self


class StubProcessor:
    """Turns prompts into row counts and any bytes into a pixel tensor. No tokeniser, no PIL."""

    def __init__(self):
        self.text_kwargs: dict = {}
        self.image_mode: str | None = None

    def __call__(self, text=None, images=None, **kwargs):
        if text is not None:
            self.text_kwargs = kwargs
            return Batch({"input_ids": torch.zeros(len(text), 3, dtype=torch.long)})
        self.image_mode = getattr(images, "mode", None)
        return Batch({"pixel_values": torch.zeros(1, 3, 8, 8)})


class Batch(dict):
    """Enough of `BatchFeature` for the encoder: `.to(device)` and `.to(device, dtype)`."""

    def to(self, device, dtype=None):
        moved = {
            key: (value.to(dtype) if dtype is not None and value.is_floating_point() else value)
            for key, value in self.items()
        }
        return Batch(moved)

    def keys(self):  # so `**inputs` works
        return super().keys()


def stubbed(config, dtype):
    """A `SiglipEncoder` with its tower already stubbed, so `load()` is a no-op."""
    built = SiglipEncoder(config, device="cpu")
    built._model = StubTower(weights=[0.0, 2.0, 0.0, -2.0], dtype=dtype)
    built._processor = StubProcessor()
    built._dtype = dtype
    return built


@pytest.fixture
def encoder(config):
    return stubbed(config, torch.float32)


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["float16", "bfloat16"])
def half(config, request):
    """A tower in reduced precision. Both of them, deliberately.

    `load()` picks **bfloat16** on CUDA, so that is the dtype the production path runs in. float16
    is kept because it is the dtype where a fixed norm floor breaks -- bf16 keeps fp32's exponent
    range and represents 1e-12 exactly, so a bf16-only test would not catch that regression.
    Every reduced-precision bug in this file's history lived in one or the other.
    """
    return stubbed(config, request.param)


PROMPTS = ("a cat.", "a dog.", "a sun.", "a tree.")


def test_encode_returns_a_sigmoid_score_per_prompt(encoder, png):
    """The arithmetic `Siglip2Model.forward` does, checked against hand-computed values.

    With a unit-norm one-hot per prompt, the dot product for prompt *i* is the image feature's
    *i*th component; the logit is that times `exp(logit_scale)` plus `logit_bias`, and the
    confidence is `sigmoid` of the logit. The stub's scale of 2 and bias of -1 mean this fails if
    either is applied the wrong way round.
    """
    encoding = encoder.encode(png, PROMPTS)

    assert len(encoding.confidences) == len(PROMPTS)
    # image features (0, 2, 0, -2), normalised to length 1 -> (0, .707, 0, -.707)
    expected = [
        pytest.approx(float(torch.sigmoid(torch.tensor(value) * 2.0 - 1.0)), abs=1e-5)
        for value in (0.0, 2 / 8**0.5, 0.0, -2 / 8**0.5)
    ]
    assert list(encoding.confidences) == expected
    assert encoding.confidences[1] > encoding.confidences[0] > encoding.confidences[3]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_a_text_row_with_no_features_does_not_poison_every_other_score(config, png, dtype):
    """Six prompts over four feature dimensions, so the stub's last two rows are zero vectors.

    A zero text row divides 0 by 0 without the norm floor, and one NaN anywhere in the text matrix
    makes *every* confidence NaN rather than only that prompt's. The four-prompt tests cannot
    reach this: `PROMPTS` is exactly `DIMENSIONS` long, so no row is ever zero.

    Run in reduced precision as well as fp32 because the floor on the **text** side is otherwise
    unpinned: at fp32 a hard-coded 1e-12 is a perfectly good floor, so only the fp16 case fails
    when someone puts one back. The text side is the one where a single NaN ruins every
    confidence rather than one encoding.
    """
    built = stubbed(config, dtype)

    encoding = built.encode(png, (*PROMPTS, "a boat.", "a kite."))

    assert len(encoding.confidences) == 6
    assert all(math.isfinite(value) for value in encoding.confidences)
    assert encoding.confidences[1] == max(encoding.confidences), "the real prompts still rank"


@pytest.mark.parametrize(("device", "dtype"), [("cuda", "bfloat16"), ("cpu", "float32")])
def test_load_selects_the_dtype_that_goes_with_the_device(config, monkeypatch, device, dtype):
    """The dtype is a decision `AGENTS.md` names, and nothing asserted it.

    Before this, setting it to float16 -- or to plain float32, giving up reduced precision on CUDA
    entirely -- left the whole suite green. This is also the only coverage `load()` has at all:
    every other test here assigns `_model` and `_processor` directly and never runs it.
    """
    import transformers

    tower, passed = StubTower(weights=[0.0] * DIMENSIONS), {}

    def fake_model(name, **kwargs):
        passed.update(kwargs)
        passed["name"] = name
        return tower

    def fake_processor(name, **kwargs):
        passed["processor"] = {"name": name, **kwargs}
        return StubProcessor()

    # The factory methods rather than the `transformers` attributes: `load()` does
    # `from transformers import AutoModel`, and that name resolves through the package's lazy
    # module, so replacing the attribute does not reliably reach what `load()` ends up calling.
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_processor)
    built = SiglipEncoder(config, device=device)

    built.load()

    assert built._dtype is getattr(torch, dtype)
    assert passed["dtype"] is getattr(torch, dtype), "the tower is loaded in it, not just recorded"
    assert passed["revision"] == config.encoder_revision, "the pinned weights, not the branch head"
    assert passed["name"] == config.encoder
    # The processor is pinned too, and for the same reason: it carries the image size, the
    # normalisation mean and standard deviation and the tokenizer, so a `preprocessor_config.json`
    # moving under pinned weights changes every score just as surely as new weights would.
    assert passed["processor"] == {"name": config.encoder, "revision": config.encoder_revision}
    assert tower.moved_to == (device,)
    assert tower.evaluated, "dropout and the like off; the tower is only ever used for inference"


def test_the_pixels_are_cast_to_the_towers_dtype(half, png):
    """Without the cast, the first matmul on CUDA raises "mat1 and mat2 must have the same dtype".

    The float32 fixture cannot show this: there `.to(device)` and `.to(device, dtype)` do the same
    thing, so the old assertion passed with the fix reverted.
    """
    half.encode(png, PROMPTS)

    assert half._model.seen_dtype == half._dtype


def test_a_zero_feature_vector_is_not_nan_in_reduced_precision(half, png):
    """The norm floor has to come from the dtype, not from a constant.

    float16's smallest normal is about 6e-5, so `clamp_min(1e-12)` on an fp16 norm clamps to
    exactly 0.0 and the 0/0 still happens. bfloat16 keeps fp32's exponent range and would not
    show it, which is why the fixture covers both rather than only the production dtype.
    """
    half._model.weights = [0.0, 0.0, 0.0, 0.0]

    encoding = half.encode(png, PROMPTS)

    assert all(math.isfinite(value) for value in encoding.confidences)
    assert all(math.isfinite(value) for value in encoding.embedding)


def test_the_drawing_is_converted_to_three_channels(encoder, png):
    """The tower expects RGB, and a child's drawing arrives as anything from 1-bit to RGBA."""
    encoder.encode(png, PROMPTS)

    assert encoder._processor.image_mode == "RGB", "the 8-bit greyscale fixture was converted"


def test_encode_returns_a_unit_length_embedding(encoder, png):
    """The KAN-71 cache key is the normalised image feature, not the raw one."""
    embedding = encoder.encode(png, PROMPTS).embedding

    assert len(embedding) == DIMENSIONS
    assert sum(value**2 for value in embedding) == pytest.approx(1.0, abs=1e-5)


def test_the_prompts_are_encoded_once_across_drawings(encoder, png):
    """The whole reason the text side is cached: 345 prompts per drawing is the latency budget."""
    for _ in range(3):
        encoder.encode(png, PROMPTS)
    assert encoder._model.text_calls == 1


def test_changing_the_prompts_re_encodes_them(encoder, png):
    """A cache keyed on anything looser -- a length, a count -- would score the wrong text."""
    encoder.encode(png, PROMPTS)
    encoder.encode(png, ("a cat.", "a dog.", "a sun.", "a boat."))
    assert encoder._model.text_calls == 2, "same length, different prompts, must be a miss"

    encoder.encode(png, ("a dog.", "a cat.", "a sun.", "a boat."))
    assert encoder._model.text_calls == 3, "same prompts reordered must be a miss too"


def test_the_cache_is_swapped_in_one_assignment(encoder, png):
    """Prompts and their features move together, so no reader can pair one with the other's.

    With two fields there is a window in which `_prompts` still matches a caller's list while
    `_text_features` has already been replaced -- stale scoring with no error anywhere.
    """
    encoder.encode(png, PROMPTS)
    cached = encoder._text_cache

    assert cached is not None
    assert cached[0] == PROMPTS, "the key is the prompts the features were built from"
    assert not hasattr(encoder, "_text_features"), "no second field to drift from the first"


def test_the_processor_is_given_the_padding_the_checkpoint_was_trained_with(encoder, png):
    """`padding='max_length'` and `max_length=64`; the card says anything else degrades retrieval."""
    encoder.encode(png, PROMPTS)

    assert encoder._processor.text_kwargs["padding"] == "max_length"
    assert encoder._processor.text_kwargs["max_length"] == 64
    assert encoder._processor.text_kwargs["truncation"] is True
