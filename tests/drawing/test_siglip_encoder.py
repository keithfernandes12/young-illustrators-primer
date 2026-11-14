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

import pytest

from primer.drawing.model import SiglipEncoder

torch = pytest.importorskip("torch", reason="torch comes from the `drawing` extra; CI omits it")
Image = pytest.importorskip("PIL.Image", reason="pillow comes from the `drawing` extra too")

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
    """

    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.logit_scale = torch.nn.Parameter(torch.tensor([0.0]))  # exp(0) == 1
        self.logit_bias = torch.nn.Parameter(torch.tensor([0.0]))
        self.text_calls = 0

    def get_text_features(self, **inputs):
        self.text_calls += 1
        count = inputs["input_ids"].shape[0]
        return Pooling(torch.eye(count, DIMENSIONS))

    def get_image_features(self, **inputs):
        assert inputs["pixel_values"].dtype == torch.float32, "the encoder casts to the tower dtype"
        return Pooling(torch.tensor([self.weights], dtype=torch.float32))

    def eval(self):
        return self


class StubProcessor:
    """Turns prompts into row counts and any bytes into a pixel tensor. No tokeniser, no PIL."""

    def __init__(self):
        self.text_kwargs: dict = {}

    def __call__(self, text=None, images=None, **kwargs):
        if text is not None:
            self.text_kwargs = kwargs
            return Batch({"input_ids": torch.zeros(len(text), 3, dtype=torch.long)})
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


@pytest.fixture
def encoder(config):
    """A `SiglipEncoder` with its tower already stubbed, so `load()` is a no-op."""
    built = SiglipEncoder(config, device="cpu")
    built._model = StubTower(weights=[0.0, 2.0, 0.0, -2.0])
    built._processor = StubProcessor()
    built._dtype = torch.float32
    return built


PROMPTS = ("a cat.", "a dog.", "a sun.", "a tree.")


def test_encode_returns_a_sigmoid_score_per_prompt(encoder, png):
    """The arithmetic `Siglip2Model.forward` does, checked against hand-computed values.

    With a unit-norm one-hot per prompt and `logit_scale` of exp(0), the logit for prompt *i* is
    the image feature's *i*th component, so the confidence is `sigmoid` of that.
    """
    encoding = encoder.encode(png, PROMPTS)

    assert len(encoding.confidences) == len(PROMPTS)
    # image features (0, 2, 0, -2), normalised to length 1 -> (0, .707, 0, -.707)
    expected = [
        pytest.approx(float(torch.sigmoid(torch.tensor(value))), abs=1e-5)
        for value in (0.0, 2 / 8**0.5, 0.0, -2 / 8**0.5)
    ]
    assert list(encoding.confidences) == expected
    assert encoding.confidences[1] > encoding.confidences[0] > encoding.confidences[3]


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
