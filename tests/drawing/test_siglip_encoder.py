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

import io
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from primer.drawing import model
from primer.drawing.model import SiglipEncoder, flattened
from primer.drawing.preprocess import default_prepare_config

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


    buffer = io.BytesIO()
    Image.new("L", (8, 8), 255).save(buffer, format="PNG")
    return buffer.getvalue()


def as_png(image) -> bytes:
    """`image` through a real PNG round trip, the way a drawing actually arrives.

    Saved and reopened rather than handed over in memory, because that is where transparency
    changes shape: a palette image keeps it in ``info`` rather than in a channel, and only the
    reopened file has the ``info`` the production path sees.
    """


    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def ink_on_transparent(mode: str):
    """A black stroke on a fully transparent canvas -- what a canvas app exports.

    The trap this fixture exists for: the ink and the background have the *same* RGB value and
    differ only in alpha, so dropping alpha loses the drawing completely rather than partly.
    """
    if mode == "RGBA":
        image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        image.putpixel((1, 1), (0, 0, 0, 255))
    elif mode == "LA":
        image = Image.new("LA", (4, 4), (0, 0))
        image.putpixel((1, 1), (0, 255))
    else:
        image = Image.new("P", (4, 4), 1)
        image.putpalette([0, 0, 0] * 256)
        image.putpixel((1, 1), 2)
        image.info["transparency"] = 1
    return image


@pytest.fixture
def transparent_png() -> bytes:
    return as_png(ink_on_transparent("RGBA"))


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


class CountingTensor(torch.Tensor):
    """A tensor that counts every crossing from its own memory into Python.

    Both ways across are counted, so the test measures the number of transfers rather than the
    shape of the call. Torch propagates the subclass through the whole arithmetic chain -- matmul,
    scale, bias, sigmoid, indexing -- so tagging the image features is enough to count what comes
    out the far end.
    """

    crossings = 0

    def __float__(self):
        CountingTensor.crossings += 1
        return super().__float__()

    def tolist(self):
        CountingTensor.crossings += 1
        return super().tolist()


class CountingTower(StubTower):
    """A `StubTower` whose image features count the transfers they cause."""

    def get_image_features(self, **inputs):
        features = super().get_image_features(**inputs).pooler_output
        return Pooling(features.as_subclass(CountingTensor))


class StubProcessor:
    """Turns prompts into row counts and any bytes into a pixel tensor. No tokeniser, no PIL."""

    def __init__(self, size: Any = None):
        self.text_kwargs: dict = {}
        #: The input resolution `load()` checks against `prepare_drawing`'s output. The real
        #: processor's is a `SizeDict` read by attribute; a plain dict is accepted too.
        self.image_processor = SimpleNamespace(size=size or SimpleNamespace(height=224, width=224))
        #: What reached the processor: `prepare_drawing`'s array. The pixels, not just the shape,
        #: because they are the only place alpha compositing is observable -- a dropped alpha
        #: channel leaves three channels and every pixel black, so a shape assertion passes while
        #: the drawing is gone.
        self.image: Any = None

    def __call__(self, text=None, images=None, **kwargs):
        if text is not None:
            self.text_kwargs = kwargs
            return Batch({"input_ids": torch.zeros(len(text), 3, dtype=torch.long)})
        self.image = images
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


def stubbed(config, dtype, prepare=None):
    """A `SiglipEncoder` with its tower already stubbed, so `load()` is a no-op."""
    built = SiglipEncoder(config, device="cpu", prepare=prepare)
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


def loading_with(monkeypatch, config, size) -> SiglipEncoder:
    """A `SiglipEncoder` whose `load()` gets a stub tower and a processor declaring `size`."""
    import transformers

    processor = StubProcessor()
    processor.image_processor = SimpleNamespace(size=size)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda name, **kwargs: StubTower([0.0] * 4))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda name, **kwargs: processor)
    return SiglipEncoder(config, device="cpu")


@pytest.mark.parametrize(
    "size",
    [SimpleNamespace(height=256, width=256), {"height": 256, "width": 256}],
    ids=["size-dict", "plain-dict"],
)
def test_load_refuses_a_processor_whose_resolution_is_not_the_prepared_one(config, monkeypatch, size):
    """`prepare_drawing` is the only resampling only while the two sizes agree; nothing else checks.

    `drawing_render.yaml` marks its `size` provisional, so the checkpoint and the render settings
    can part. When they do, the processor would resize every prepared drawing a second time.
    """
    built = loading_with(monkeypatch, config, size)

    with pytest.raises(ValueError, match="prepared at 224 x 224"):
        built.load()
    assert built._model is None, "nothing is published from a refused load"


@pytest.mark.parametrize(
    "size",
    [SimpleNamespace(height=224, width=224), {"height": 224, "width": 224}],
    ids=["size-dict", "plain-dict"],
)
def test_load_accepts_a_processor_at_the_prepared_resolution(config, monkeypatch, size):
    """Both shapes a processor's `size` comes in -- the real one is a `SizeDict`, read by attribute."""
    built = loading_with(monkeypatch, config, size)

    built.load()

    assert built._model is not None


@pytest.mark.parametrize("size", [None, {"shortest_edge": 224}, SimpleNamespace(max_num_patches=256)])
def test_a_processor_with_no_fixed_square_is_refused_for_what_it_is(config, monkeypatch, size):
    """Shortest-edge and patch-budget processors have no resolution to set `size` to."""
    built = loading_with(monkeypatch, config, size)

    with pytest.raises(ValueError, match="no fixed height x width"):
        built.load()


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

    seen = encoder._processor.image
    assert seen.shape == (224, 224, 3), "the 8-bit greyscale fixture was converted"
    assert seen.dtype == np.uint8


def test_the_tower_sees_the_drawing_through_prepare_drawing(encoder, png, monkeypatch):
    """KAN-31: decode and composite at the call site, then the one preprocessing path."""
    calls = []
    real = model.prepare_drawing

    def recording(image, config=None, **kwargs):
        calls.append((image.shape, config))
        return real(image, config, **kwargs)

    monkeypatch.setattr(model, "prepare_drawing", recording)
    encoder.encode(png, PROMPTS)

    assert calls == [((8, 8, 3), None)], "the composited RGB array, with the shipped settings"


def test_a_wide_drawing_reaches_the_tower_padded_rather_than_squashed(encoder):
    """The processor resizes straight to a square; `prepare_drawing` gets there first."""
    wide = Image.new("RGB", (8, 4), (255, 255, 255))
    wide.putpixel((3, 1), (0, 0, 0))
    encoder.encode(as_png(wide), PROMPTS)

    # Each source pixel is a 28px block, the drawing's four rows sit in rows 56-167, and the mark
    # (source row 1) in rows 84-111. Squashed instead, source row 1 would be rows 56-111.
    seen = encoder._processor.image
    assert seen.shape == (224, 224, 3)
    assert (seen[:56] == 255).all() and (seen[-56:] == 255).all(), "paper above and below"
    assert seen[100, 100].tolist() == [0, 0, 0], "the mark"
    assert seen[60, 100].tolist() == [255, 255, 255], "where the squash would have put the mark"


def test_the_encoder_prepares_with_the_settings_it_was_given(config):
    """How S6c2 runs its variants: a `PrepareConfig` handed to the encoder reaches every drawing."""
    greyscale = replace(default_prepare_config(), greyscale=True)
    built = stubbed(config, torch.float32, prepare=greyscale)  # through the constructor, as S6c2 will
    red = Image.new("RGB", (8, 8), (255, 0, 0))

    built.encode(as_png(red), PROMPTS)

    assert (built._processor.image == 76).all(), "BT.601 luma of pure red, on all three channels"


def test_a_call_can_override_the_settings_and_crop_to_a_box(encoder):
    """S6c2's variants: per-call settings and the annotated box, through the one path."""
    page = Image.new("RGB", (8, 8), (255, 255, 255))
    for x in range(2, 4):
        for y in range(2, 4):
            page.putpixel((x, y), (255, 0, 0))

    encoder.encode(as_png(page), PROMPTS, prepare=replace(default_prepare_config(), greyscale=True), box=(2, 2, 2, 2))

    assert (encoder._processor.image == 76).all(), "greyscaled, and cropped to the red square alone"
    encoder.encode(as_png(page), PROMPTS)
    assert encoder._processor.image[0, 0].tolist() == [255, 255, 255], "the override did not stick"


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


# ------------------------------------------------------------------ transparency is composited


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_transparency_is_composited_onto_the_background(mode):
    """`convert("RGB")` does not composite alpha, it drops it -- and the drawing with it.

    Measured on this fixture before the fix: all three modes converted to a uniform `(0, 0, 0)`
    square, ink and canvas alike, because a sketch is black on transparent-black. Nothing raised;
    the tower scored a black square and returned a confident label for it.

    All three shapes are here because the mode alone does not say whether an image is transparent:
    a palette PNG reports `"P"` and carries its transparency in `info`.
    """
    image = Image.open(io.BytesIO(as_png(ink_on_transparent(mode))))

    result = flattened(image, background=255)

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255), "the transparent canvas became background"
    assert result.getpixel((1, 1)) == (0, 0, 0), "the ink survived the composite"


def test_the_composite_uses_the_configured_background():
    """The background is a setting because it changes every score, so it has to be read."""
    image = Image.open(io.BytesIO(as_png(ink_on_transparent("RGBA"))))

    assert flattened(image, background=0).getpixel((0, 0)) == (0, 0, 0)
    assert flattened(image, background=128).getpixel((0, 0)) == (128, 128, 128)


def test_an_opaque_drawing_is_only_converted():
    """No canvas, no composite: the common path stays a plain conversion."""
    grey = Image.open(io.BytesIO(as_png(Image.new("L", (4, 4), 200))))

    result = flattened(grey, background=255)

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (200, 200, 200), "the pixels are untouched"


def test_encode_composites_before_the_tower_sees_the_drawing(encoder, transparent_png):
    """The end-to-end form of the above: what reaches the processor is the composited image.

    `test_the_drawing_is_converted_to_three_channels` cannot catch this -- a dropped alpha channel
    leaves the mode "RGB" and passes, while every pixel is black.
    """
    encoder.encode(transparent_png, PROMPTS)
    seen = encoder._processor.image

    # The 4x4 fixture reaches the tower enlarged to 224x224, so each source pixel is a 56px block:
    # source (1, 1), the ink, is rows and columns 56-111.
    assert seen.shape == (224, 224, 3)
    assert seen[0, 0].tolist() == [255, 255, 255]
    assert seen[80, 80].tolist() == [0, 0, 0]


# ------------------------------------------------------------------ one transfer per vector


def test_each_result_vector_crosses_to_the_host_once(config, png):
    """`float()` per element is a device-to-host copy per element, and each one synchronises.

    For the shipped checkpoint that is 345 confidences + 768 embedding components = 1113 transfers
    per drawing, on the warm path KAN-71 budgets. `tolist()` moves each vector in one. The values
    are unchanged, which is what the rest of this file pins; this pins the cost.
    """
    built = stubbed(config, torch.float32)
    built._model = CountingTower(weights=[0.0, 2.0, 0.0, -2.0])
    CountingTensor.crossings = 0

    encoding = built.encode(png, PROMPTS)

    assert len(encoding.confidences) == len(PROMPTS)
    assert len(encoding.embedding) == DIMENSIONS
    assert CountingTensor.crossings == 2, "one transfer per vector, not one per element"


# ------------------------------------------------------------------ loading is serialised


def test_concurrent_first_calls_load_one_checkpoint(config, monkeypatch):
    """Two requests arriving on a cold start is the normal case, not a rare interleaving.

    The serving layer runs sync endpoints in a threadpool, so an unguarded `_model is None` lets
    both callers past and loads two copies of the weights onto the same device -- 1.4 GB of the
    8 GB budget, spent on a duplicate, at exactly the moment the first request is already slow.
    """
    import threading
    import time

    import transformers

    loads = []

    def fake_model(name, **kwargs):
        loads.append(name)
        # Long enough that every thread is inside the unguarded window: without the lock this
        # fails with four loads, not flakily but every run.
        time.sleep(0.05)
        return StubTower(weights=[0.0] * DIMENSIONS)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda name, **kwargs: StubProcessor()
    )
    built = SiglipEncoder(config, device="cpu")
    threads = [threading.Thread(target=built.load) for _ in range(4)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert loads == [config.encoder], "the checkpoint is fetched once"
    assert built._model is not None


def test_concurrent_first_calls_build_one_encoder(config, monkeypatch):
    """Serialising `load` is not enough on its own.

    Two threads racing the `encoder` property build two *different* `SiglipEncoder` objects, and
    each then loads its own checkpoint past its own lock. The duplicated VRAM is reachable through
    this property no matter how careful `load` is, which is why both are locked.
    """
    import threading
    import time

    from primer.drawing import model as model_module

    built = []

    def slow_encoder(cfg):
        built.append(cfg)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(model_module, "SiglipEncoder", slow_encoder)
    recogniser = model_module.DrawingRecognizerModel(config)
    seen: list[Any] = []
    threads = [
        threading.Thread(target=lambda: seen.append(recogniser.encoder)) for _ in range(4)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(built) == 1, "one encoder, so one checkpoint"
    assert len({id(value) for value in seen}) == 1, "and every caller got that one"
