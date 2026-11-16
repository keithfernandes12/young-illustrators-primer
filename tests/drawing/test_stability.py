"""S6c2's preprocessing-stability measurement in `evaluate.py`, against a fake encoder.

No weights, no bucket, no Pillow: the encoder is a recording fake whose scores depend on which
variant it was asked for, so every statistic can be checked by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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


def test_the_label_share_is_the_commonest_label_over_all_drawings_not_distinct_over_total():
    """Distinct labels / drawings would be 3/6 here; the collapse detector is the share, 4/6."""
    scores = [ranked((0, 1, 2))] * 4 + [ranked((1, 0, 2)), ranked((2, 0, 1))]

    stats = variant_stats(scores)

    assert stats["distinct_top1"] == 3
    assert stats["top1_label_share"] == pytest.approx(4 / 6)
    assert stats["top1_label_share"] != pytest.approx(stats["distinct_top1"] / len(scores))


def test_the_confident_half_is_cut_at_the_median_margin_not_the_mean():
    """Margins 0.1, 0.1, 0.9: the median is 0.1, so all three qualify; the mean (0.37) would keep one."""
    before = [ranked((0, 1, 2), 0.2, 0.1), ranked((0, 1, 2), 0.2, 0.1), ranked((0, 1, 2), 1.0, 0.1)]
    after = [ranked((1, 0, 2)), ranked((1, 0, 2)), ranked((0, 1, 2))]

    stats = pair_stats(before, after)

    assert stats["confident_drawings"] == 3
    assert stats["top1_change_rate_confident"] == pytest.approx(2 / 3)


def test_when_the_median_margin_is_zero_the_confident_subset_is_everyone_and_says_so():
    """A saturated model: the confident rate equals the overall one by construction. The count shows it."""
    before = [ranked((0, 1, 2), 1.0, 1.0)] * 4
    after = [ranked((1, 0, 2))] * 2 + [ranked((0, 1, 2))] * 2

    stats = pair_stats(before, after)

    assert stats["confident_drawings"] == 4
    assert stats["top1_change_rate_confident"] == stats["top1_change_rate"] == 0.5


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.1])
def test_scores_that_are_not_sigmoid_outputs_are_refused_not_ranked(bad):
    with pytest.raises(ValueError, match="0-1"):
        scored([0.5, bad, 0.2])


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


BOX = (11, 22, 333, 444)  # four distinct values, so a swapped pair cannot hide
LICENCE = "CC-BY-4.0 (test)"
CREDIT = "Somebody, Some Dataset (test)"


class CliWorld:
    """A fake store, data read and encoder, wired into `evaluate._stability` by monkeypatch."""

    def __init__(self, monkeypatch, *, uri_prefix="gs://bucket", frame_edit=None, winner=5):
        import hashlib

        import pandas as pd

        from primer.data import catalog, storage
        from primer.drawing import model

        self.boxes: list[object] = []
        self.devices: list[object] = []
        ids = evaluate.load_sample()
        self.frame = pd.DataFrame(
            {
                "record_id": ids,
                "image_uri": [f"{uri_prefix}/images/{i}.png" for i in ids],
                "image_sha256": hashlib.sha256(b"png").hexdigest(),
                "bbox_x": BOX[0],
                "bbox_y": BOX[1],
                "bbox_width": BOX[2],
                "bbox_height": BOX[3],
                "license": LICENCE,
                "attribution": CREDIT,
            }
        )
        if frame_edit:
            self.frame = frame_edit(self.frame)
        world = self

        class FakeStore:
            uri = "gs://bucket"

            def read_bytes(self, relative):
                return b"png"

        class FakeEncoder:
            def __init__(self, config, device=None):
                world.devices.append(device)
                self.device = device or "auto-picked"
                self.dtype_name = "bfloat16" if device == "cuda" else "float32"

            def encode(self, image, prompts, *, prepare=None, box=None):
                world.boxes.append(box)
                size = len(prompts)
                scores = [0.1] * size
                scores[winner] = 0.9  # a winner that is not label 0
                return SimpleNamespace(confidences=scores)

        monkeypatch.setattr(catalog, "read_dataset", lambda *args, **kwargs: self.frame)
        monkeypatch.setattr(storage, "Store", FakeStore)
        monkeypatch.setattr(model, "SiglipEncoder", FakeEncoder)


def run_cli(tmp_path, *extra):
    out = tmp_path / "stability.json"
    code = evaluate.main(["stability", "--version", "v1.1", "--out", str(out), *extra])
    return code, out


@pytest.mark.parametrize("device", ["cpu", "cuda", None])
def test_the_stability_cli_passes_the_device_on_and_records_what_ran(monkeypatch, tmp_path, device):
    import json

    world = CliWorld(monkeypatch)

    code, out = run_cli(tmp_path, *(["--device", device] if device else []))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and world.devices == [device], "the chosen device reaches the encoder"
    assert result["device"] == (device or "auto-picked")
    assert result["dtype"] == ("bfloat16" if device == "cuda" else "float32")
    assert result["drawings"] == 1000 and result["prompt_count"] == 345
    assert result["scoring"]["prompt_template"] and result["preprocessing"]["binarised"]["binarise"] is True


def test_the_stability_cli_carries_the_box_the_licence_and_the_label_through(monkeypatch, tmp_path):
    import json

    from primer.drawing.model import DrawingConfig

    world = CliWorld(monkeypatch, winner=5)

    _, out = run_cli(tmp_path)

    result = json.loads(out.read_text(encoding="utf-8"))
    assert world.boxes[:4] == [None, None, None, BOX], "(x, y, width, height), and only the last variant crops"
    assert result["license"] == [LICENCE] and result["attribution"] == [CREDIT]
    labels = DrawingConfig.from_yaml().labels()
    assert labels[5] != labels[0]
    assert {stats["top1_label"] for stats in result["variants"].values()} == {labels[5]}


def test_an_image_outside_the_store_is_refused(monkeypatch, tmp_path):
    CliWorld(monkeypatch, uri_prefix="gs://somewhere-else")

    with pytest.raises(ValueError, match="is not in gs://bucket"):
        run_cli(tmp_path)


def test_bytes_that_are_not_the_recorded_bytes_are_refused(monkeypatch, tmp_path):
    CliWorld(monkeypatch, frame_edit=lambda frame: frame.assign(image_sha256="0" * 64))

    with pytest.raises(ValueError, match="do not match the recorded image_sha256"):
        run_cli(tmp_path)


def test_a_record_id_that_repeats_is_refused_rather_than_scored_twice(monkeypatch, tmp_path):
    import pandas as pd

    CliWorld(monkeypatch, frame_edit=lambda frame: pd.concat([frame, frame.iloc[:1]], ignore_index=True))

    with pytest.raises(ValueError, match="matched 1001 rows"):
        run_cli(tmp_path)


def test_an_unusable_output_folder_is_refused_before_any_scoring(monkeypatch, tmp_path):
    world = CliWorld(monkeypatch)

    with pytest.raises(SystemExit):
        evaluate.main(["stability", "--version", "v1.1", "--out", str(tmp_path / "missing" / "s.json")])
    assert world.boxes == [], "nothing was scored first"


def test_anything_but_stability_goes_to_s6cs_accuracy_cli(monkeypatch):
    """S6c filled `--model --split --out`; the `stability` subcommand stays beside it."""
    seen: list[list[str]] = []
    monkeypatch.setattr(evaluate, "_evaluate_accuracy", lambda argv: seen.append(argv) or 0)

    assert evaluate.main(["--model", "m", "--version", "v1.1", "--out", "o.json"]) == 0
    assert seen == [["--model", "m", "--version", "v1.1", "--out", "o.json"]]



def test_the_stability_summary_records_the_commit_it_ran_at(monkeypatch, tmp_path):
    import json

    CliWorld(monkeypatch)
    monkeypatch.setattr(evaluate, "_git_state", lambda: ("abc1234", False))

    code, out = run_cli(tmp_path)

    result = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and result["git_sha"] == "abc1234" and result["git_dirty"] is False


def test_stability_mlflow_logs_the_written_summary(monkeypatch, tmp_path):
    import json

    from primer.drawing import tracking

    CliWorld(monkeypatch)
    calls = []
    monkeypatch.setattr(tracking, "log_run", lambda **kwargs: calls.append(kwargs) or "run-1")

    code, out = run_cli(tmp_path, "--mlflow", "--owner", "m1")

    assert code == 0 and len(calls) == 1
    assert calls[0]["kind"] == "stability" and calls[0]["metrics_path"] == out
    assert calls[0]["result"] == json.loads(out.read_text(encoding="utf-8"))


def test_stability_mlflow_without_an_owner_is_refused_before_any_read(monkeypatch, tmp_path):
    world = CliWorld(monkeypatch)

    with pytest.raises(SystemExit):
        run_cli(tmp_path, "--mlflow")
    assert world.devices == [], "the encoder was never built"
