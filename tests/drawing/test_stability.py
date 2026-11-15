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


def test_the_gate_cli_is_still_s6cs_stub():
    """S6c fills `--model --split --out`; this slice only adds the `stability` subcommand."""
    with pytest.raises(NotImplementedError):
        evaluate.main(["--model", "m", "--out", "o.json"])
