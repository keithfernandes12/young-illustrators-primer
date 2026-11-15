"""S6c2's preprocessing-stability measurement in `evaluate.py`, against a fake encoder.

No weights, no bucket, no Pillow: the encoder is a recording fake whose scores depend on which
variant it was asked for, so every statistic can be checked by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from primer.drawing import evaluate
from primer.drawing.evaluate import ADJACENT, VARIANTS, Scored, pair_stats, scored, variant_stats


def ranked(order: tuple[int, ...], top: float = 0.5, second: float = 0.1) -> Scored:
    return Scored(ranking=order, top1_confidence=top, margin=top - second)


# ---------------------------------------------------------------------------------------- variants


def test_the_variants_follow_the_plans_table_one_operation_at_a_time():
    settings = evaluate.variant_settings()

    assert tuple(settings) == VARIANTS
    assert ADJACENT == (
        ("as_published", "greyscale"),
        ("greyscale", "binarised"),
        ("binarised", "cropped_binarised"),
    )
    # Each adjacent pair differs by exactly one of: greyscale, binarise, crop.
    flags = {name: (p.greyscale, p.binarise, crops) for name, (p, crops) in settings.items()}
    for before, after in ADJACENT:
        assert sum(a != b for a, b in zip(flags[before], flags[after], strict=True)) == 1


def test_only_the_last_variant_crops():
    assert [crops for _, crops in evaluate.variant_settings().values()] == [False, False, False, True]


# -------------------------------------------------------------------------------------- statistics


def test_scored_ranks_best_first_and_measures_the_margin():
    result = scored([0.1, 0.7, 0.3, 0.7])

    assert result.ranking == (1, 3, 2, 0), "ties keep vocabulary order"
    assert result.top1_confidence == 0.7 and result.margin == 0.0


def test_scored_needs_two_labels():
    with pytest.raises(ValueError, match="two labels"):
        scored([0.9])


def test_pair_stats_counts_changed_winners_and_surviving_top_fives():
    before = [ranked((0, 1, 2, 3, 4, 5)), ranked((0, 1, 2, 3, 4, 5))]
    after = [ranked((0, 1, 2, 3, 4, 5)), ranked((5, 1, 2, 3, 4, 0))]

    stats = pair_stats(before, after)

    assert stats["top1_change_rate"] == 0.5
    assert stats["top5_overlap"] == pytest.approx((5 / 5 + 4 / 5) / 2)


def test_the_confident_change_rate_ignores_drawings_with_no_preference():
    """Flips on near-ties are noise; this rate is what separates them from an effect."""
    before = [ranked((0, 1, 2, 3, 4), 0.50, 0.49), ranked((0, 1, 2, 3, 4), 0.9, 0.1)]
    after = [ranked((1, 0, 2, 3, 4)), ranked((0, 1, 2, 3, 4))]

    stats = pair_stats(before, after)

    assert stats["top1_change_rate"] == 0.5
    assert stats["top1_change_rate_confident"] == 0.0, "only the near-tie flipped"


def test_overlap_is_out_of_the_labels_there_are_when_there_are_fewer_than_five():
    """An unchanged ranking shares everything it has; dividing by five called three labels 0.6."""
    same = [ranked((0, 1, 2)), ranked((2, 0, 1))]

    assert pair_stats(same, same)["top5_overlap"] == 1.0
    two = pair_stats([ranked((0, 1))], [ranked((1, 0))])
    assert two["top5_overlap"] == 1.0, "the same two labels, in the other order, is complete overlap"


def test_overlap_still_counts_the_top_five_when_there_are_more_labels():
    before = [ranked((0, 1, 2, 3, 4, 5, 6))]
    after = [ranked((0, 1, 2, 5, 6, 3, 4))]  # top five: {0,1,2,5,6} against {0,1,2,3,4}

    assert pair_stats(before, after)["top5_overlap"] == pytest.approx(3 / 5)


def test_flips_that_are_only_near_ties_leave_the_confident_rate_below_the_overall_one():
    """The reading the docstring gives: near-tie flips pile up below the median margin.

    The opposite -- a confident rate as high as the overall one -- means confident answers moved too.
    """
    near_ties = [ranked((0, 1, 2, 3, 4), 0.50, 0.49)] * 3 + [ranked((0, 1, 2, 3, 4), 0.9, 0.1)] * 3
    flipped = [ranked((1, 0, 2, 3, 4))] * 3 + [ranked((0, 1, 2, 3, 4))] * 3
    everywhere = [ranked((1, 0, 2, 3, 4))] * 6

    only_near_ties = pair_stats(near_ties, flipped)
    all_moved = pair_stats(near_ties, everywhere)

    assert only_near_ties["top1_change_rate_confident"] < only_near_ties["top1_change_rate"]
    assert all_moved["top1_change_rate_confident"] == all_moved["top1_change_rate"] == 1.0


def test_pair_stats_refuses_mismatched_sets():
    with pytest.raises(ValueError, match="same, non-empty"):
        pair_stats([ranked((0, 1))], [])


def test_variant_stats_report_confidence_margin_and_concentration():
    scores = [ranked((0, 1), 0.6, 0.2), ranked((0, 1), 0.4, 0.3), ranked((1, 0), 0.2, 0.1)]

    stats = variant_stats(scores)

    assert stats["top1_confidence_median"] == pytest.approx(0.4)
    assert stats["margin_median"] == pytest.approx(0.1)
    assert stats["distinct_top1"] == 2
    assert stats["top1_label_share"] == pytest.approx(2 / 3), "the collapse detector S6d reads"
    assert stats["top1_label_index"] == 0, "and which label it is"


# ---------------------------------------------------------------------------------- the whole run


@dataclass
class Encoding:
    confidences: tuple[float, ...]


class FakeEncoder:
    """Scores depend on the variant asked for, and every call is recorded."""

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, bool, bool, object]] = []

    def encode(self, image, prompts, *, prepare=None, box=None):
        self.calls.append((image, prepare.greyscale, prepare.binarise, box))
        if box is not None:
            return Encoding((0.1, 0.2, 0.9))  # cropping moves the answer
        return Encoding((0.8, 0.3, 0.1))


def test_every_drawing_is_scored_under_every_variant_and_only_the_crop_gets_the_box():
    encoder = FakeEncoder()
    drawings = [(b"one", (1, 2, 3, 4)), (b"two", (5, 6, 7, 8))]

    summary = evaluate.measure_stability(drawings, encoder, prompts=("a", "b", "c"))

    assert summary["drawings"] == 2
    assert len(encoder.calls) == 8
    boxes = [box for *_, box in encoder.calls]
    assert boxes == [None, None, None, (1, 2, 3, 4), None, None, None, (5, 6, 7, 8)]
    assert summary["pairs"]["greyscale->binarised"]["top1_change_rate"] == 0.0
    assert summary["pairs"]["binarised->cropped_binarised"]["top1_change_rate"] == 1.0
    assert summary["rank_statistic"] == "top-5 set overlap"


# -------------------------------------------------------------------------------------- the sample


def test_the_frozen_sample_is_a_thousand_distinct_ids_and_says_how_it_was_drawn():
    ids = evaluate.load_sample()
    header = evaluate.SAMPLE_PATH.read_text(encoding="utf-8")

    assert len(ids) == 1000 and len(set(ids)) == 1000
    assert "20261115" in header and "UNSTRATIFIED" in header and "v1.1" in header


def test_a_sample_listing_a_record_twice_is_refused(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("# header\nabc\nabc\n", encoding="utf-8")

    with pytest.raises(ValueError, match="twice"):
        evaluate.load_sample(path)


def test_a_run_records_what_its_numbers_were_taken_under():
    from primer.drawing.model import DrawingConfig

    config = DrawingConfig.from_yaml()
    record = evaluate.describe_run(config, "cpu", "float32")

    assert record["device"] == "cpu" and record["dtype"] == "float32"
    assert record["scoring"]["prompt_template"] == config.prompt_template
    assert record["scoring"]["text_max_length"] == config.text_max_length
    assert "config_dir" not in record["scoring"], "a path on one machine says nothing about the score"
    assert record["prompt_count"] == len(config.labels()) == len(config.prompts())
    assert len(record["prompts_sha256"]) == 64
    assert set(record["preprocessing"]) == set(VARIANTS)
    assert record["preprocessing"]["greyscale"]["greyscale"] is True
    assert record["preprocessing"]["binarised"]["binarise"] is True
    assert [name for name, v in record["preprocessing"].items() if v["crops_to_box"]] == [VARIANTS[-1]]
    assert record["preprocessing"]["as_published"]["render"]["size"] == 224
    import json

    json.dumps(record)  # it is written to a file, so it must serialise


def test_the_recorded_prompt_hash_moves_with_the_prompt_template():
    from dataclasses import replace

    from primer.drawing.model import DrawingConfig

    config = DrawingConfig.from_yaml()
    other = replace(config, prompt_template="a doodle of a {label}")

    assert (
        evaluate.describe_run(config, "cpu", "float32")["prompts_sha256"]
        != evaluate.describe_run(other, "cpu", "float32")["prompts_sha256"]
    )


def test_the_stability_cli_can_pin_the_device():
    parser = evaluate.stability_parser()

    assert parser.parse_args(["--version", "v", "--out", "o"]).device is None
    assert parser.parse_args(["--version", "v", "--out", "o", "--device", "cpu"]).device == "cpu"
    assert parser.parse_args(["--version", "v", "--out", "o", "--device", "cuda"]).device == "cuda"
    with pytest.raises(SystemExit):
        parser.parse_args(["--version", "v", "--out", "o", "--device", "tpu"])


@pytest.mark.parametrize("device", ["cpu", "cuda", None])
def test_the_stability_cli_passes_the_device_on_and_records_what_ran(monkeypatch, tmp_path, device):
    """End to end on fakes: the store, the data read and the encoder are stand-ins."""
    import json
    from types import SimpleNamespace

    import pandas as pd

    from primer.data import catalog, storage
    from primer.drawing import model

    ids = evaluate.load_sample()
    frame = pd.DataFrame(
        {
            "record_id": ids,
            "image_uri": [f"gs://bucket/images/{i}.png" for i in ids],
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_width": 10,
            "bbox_height": 10,
            "license": "CC-BY-4.0",
            "attribution": "someone",
        }
    )
    seen: dict[str, object] = {}

    class FakeStore:
        uri = "gs://bucket"

        def read_bytes(self, relative):
            return b"png"

    class FakeEncoder:
        def __init__(self, config, device=None):
            seen["device"] = device
            self.device = device or "auto-picked"
            self.dtype_name = "bfloat16" if device == "cuda" else "float32"

        def encode(self, image, prompts, *, prepare=None, box=None):
            return SimpleNamespace(confidences=[(i * 37 % len(prompts)) / len(prompts) for i in range(len(prompts))])

    monkeypatch.setattr(catalog, "read_dataset", lambda *args, **kwargs: frame)
    monkeypatch.setattr(storage, "Store", FakeStore)
    monkeypatch.setattr(model, "SiglipEncoder", FakeEncoder)
    out = tmp_path / "stability.json"
    argv = ["stability", "--version", "v1.1", "--out", str(out)] + (["--device", device] if device else [])

    assert evaluate.main(argv) == 0

    result = json.loads(out.read_text(encoding="utf-8"))
    assert seen["device"] == device, "the chosen device reaches the encoder"
    assert result["device"] == (device or "auto-picked")
    assert result["dtype"] == ("bfloat16" if device == "cuda" else "float32")
    assert result["drawings"] == 1000 and result["prompt_count"] == 345
    assert result["scoring"]["prompt_template"] and result["preprocessing"]["binarised"]["binarise"] is True


def test_the_gate_cli_is_still_s6cs_stub():
    """S6c fills `--model --split --out`; this slice only adds the `stability` subcommand."""
    with pytest.raises(NotImplementedError):
        evaluate.main(["--model", "m", "--out", "o.json"])
