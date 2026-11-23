"""S6f's sensitivity analysis in `drawing/sensitivity.py`, against a fake scorer.

No weights, no bucket, no Pillow, no MLflow: the "recogniser" scores a canvas by how much ink it
holds, so every perturbation has an effect that can be predicted by hand.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from primer.drawing import sensitivity
from primer.drawing.evaluate import scored
from primer.drawing.render import default_config, render_strokes
from primer.drawing.sensitivity import (
    Drawing,
    erased,
    erasure_only,
    ink_mass,
    measure,
    occlusion,
    paired_delta,
)
from primer.drawing.tracking import sensitivity_metrics

RENDER = default_config()
PROMPTS = tuple(f"p{i}" for i in range(8))
#: A full canvas of ink, for scaling the fake's score into 0-1.
FULL = float(RENDER.background * RENDER.size * RENDER.size)

# Three strokes: a long diagonal, a horizontal line and a short tick.
THREE = [[[0, 255], [0, 255]], [[0, 255], [128, 128]], [[20, 40], [200, 220]]]
ONE = [[[0, 255], [255, 0]]]


class InkScorer:
    """Label 0 scores by ink mass, so less ink pushes it down the ranking; the rest are fixed.

    Labels 1-7 sit at 0.50, 0.49, ... so label 0 is in the top five exactly when its score beats
    0.46 (label 5's), which happens when the canvas holds enough ink.
    """

    def __init__(self, gain: float = 60.0) -> None:
        self.gain = gain
        self.calls: list[tuple[np.ndarray, tuple[str, ...]]] = []

    def __call__(self, canvas: np.ndarray, prompts: Sequence[str]) -> list[float]:
        self.calls.append((canvas, tuple(prompts)))
        first = min(0.99, ink_mass(canvas, RENDER) / FULL * self.gain)
        return [first] + [0.5 - 0.01 * i for i in range(len(prompts) - 1)]


# ------------------------------------------------------------------------------------- statistics


def test_identical_hits_give_a_zero_delta_with_a_zero_interval():
    figures = paired_delta([True, False, True], [True, False, True], seed=0)

    assert figures["top5_delta"] == 0.0
    assert figures["top5_delta_ci_low"] == figures["top5_delta_ci_high"] == 0.0


def test_the_delta_is_paired_and_its_interval_contains_it():
    before = [True] * 50 + [False] * 50
    after = [True] * 30 + [False] * 70  # 20 hits lost, every one of them from `before`'s hits

    figures = paired_delta(before, after, seed=1)

    assert figures["top5_acc_before"] == 0.5 and figures["top5_acc_after"] == 0.3
    assert figures["top5_delta"] == -0.2
    assert figures["top5_delta_ci_low"] <= -0.2 <= figures["top5_delta_ci_high"] < 0


def test_the_same_seed_gives_the_same_interval():
    before, after = [True, False] * 20, [False, True, True, False] * 10

    assert paired_delta(before, after, seed=7) == paired_delta(before, after, seed=7)


def test_unequal_or_empty_sides_are_refused():
    with pytest.raises(ValueError, match="same, non-empty"):
        paired_delta([True], [True, False], seed=0)
    with pytest.raises(ValueError, match="same, non-empty"):
        paired_delta([], [], seed=0)


# ------------------------------------------------------------------------------------- erasure


def test_erasure_only_switches_every_other_operation_off():
    config = erasure_only()

    assert config.erasure_probability == 1.0
    assert config.wobble_probability == config.fill_probability == 0.0
    assert config.ink_colour_probability == config.lighting_probability == 0.0
    assert config.paper_texture == config.paper_grain == 0.0


def test_erasure_returns_a_grey_canvas_and_the_ink_it_removed():
    canvas = render_strokes(THREE, RENDER)

    image, removed = erased(canvas, 3, erasure_only())

    assert image.shape == canvas.shape and image.dtype == np.uint8
    assert 0.0 <= removed <= 1.0
    assert removed == pytest.approx(1 - ink_mass(image, RENDER) / ink_mass(canvas, RENDER))


def test_a_colourful_augmentation_is_refused_rather_than_measured():
    colourful = replace(erasure_only(), ink_colour_probability=1.0)

    with pytest.raises(ValueError, match="colour"):
        erased(render_strokes(THREE, RENDER), 0, colourful)


# ------------------------------------------------------------------------------------- occlusion


def test_occlusion_scores_only_inked_patches_and_finds_the_one_that_matters():
    canvas = np.full((RENDER.size, RENDER.size), RENDER.background, dtype=np.uint8)
    canvas[40:50, 40:50] = RENDER.ink  # all the ink in one 32px patch: row 1, column 1

    result = occlusion(canvas, 0, InkScorer(gain=2000.0), PROMPTS, RENDER)

    assert result["inked_patches"] == 1
    assert result["grid"][1][1] is not None and result["grid"][1][1] > 0
    assert result["grid"][0][0] is None, "a blank patch is not scored: its drop is 0 by construction"
    assert result["top3_share"] == 1.0
    assert result["max_drop"] == result["grid"][1][1]


def test_top3_share_sorts_the_drops_and_leaves_negative_ones_out():
    canvas = np.full((RENDER.size, RENDER.size), RENDER.background, dtype=np.uint8)
    # Row-major patches: (0,0) a block whose covering RAISES confidence, then positives of 16, 100,
    # 36 and 64 ink pixels. In raster order the first three positives are 16, 100, 36, so a missing
    # sort reads 152/216, not 200/216; counting the negative drop shrinks the denominator.
    canvas[5:15, 5:15] = RENDER.ink
    canvas[5:9, 40:44] = RENDER.ink
    canvas[5:15, 70:80] = RENDER.ink
    canvas[40:46, 5:11] = RENDER.ink
    canvas[40:48, 40:48] = RENDER.ink
    spoiler = (slice(5, 15), slice(5, 15))

    def scorer(image: np.ndarray, prompts: Sequence[str]) -> list[float]:
        rest = image.copy()
        rest[spoiler] = RENDER.background
        # The ink of the spoiler block suppresses the label; covering it lifts the score by 0.3.
        score = 0.1 + 100.0 * ink_mass(rest, RENDER) / FULL
        return [score + (0.0 if (image[spoiler] < RENDER.background).any() else 0.3)]

    result = occlusion(canvas, 0, scorer, PROMPTS, RENDER)

    assert result["inked_patches"] == 5
    assert result["grid"][0][0] < 0, "covering the spoiler raises confidence: a negative drop"
    assert result["top3_share"] == pytest.approx(200 / 216, abs=1e-4)


def test_top3_share_is_undefined_when_no_patch_lowers_the_confidence():
    constant = lambda image, prompts: [0.9] + [0.1] * (len(prompts) - 1)

    assert occlusion(render_strokes(THREE, RENDER), 0, constant, PROMPTS, RENDER)["top3_share"] is None

    result = measure(drawings(), constant, PROMPTS, prompt_sets={}, seed=0, occlude=5)
    summary = result["occlusion"]

    assert summary["drawings"] == 3
    assert summary["top3_share_undefined"] == 3, "counted, not silently dropped"
    assert summary["top3_share_median"] is None, "and a 0.0 never enters the median"
    assert summary["inked_patches_median"] > 0


def test_the_occlusion_summary_reports_the_median_inked_patches_and_undefined_shares():
    result = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={}, seed=0, occlude=5)
    summary = result["occlusion"]
    counts = [m["inked_patches"] for m in summary["maps"].values()]

    assert summary["inked_patches_median"] == statistics.median(counts) > 0
    assert summary["top3_share_undefined"] == sum(m["top3_share"] is None for m in summary["maps"].values())
    metrics = sensitivity_metrics(result)
    assert metrics["sensitivity/occlusion/inked_patches_median"] == summary["inked_patches_median"]
    assert metrics["sensitivity/occlusion/top3_share_undefined"] == summary["top3_share_undefined"]


def test_the_median_inked_patch_count_is_the_median_not_the_maximum():
    constant = lambda image, prompts: [0.9] + [0.1] * (len(prompts) - 1)
    short, medium, full = (
        Drawing(name, [[[0, end], [128, 128]]], 0) for name, end in (("s", 60), ("m", 130), ("f", 255))
    )

    summary = measure([short, medium, full], constant, PROMPTS, prompt_sets={}, occlude=5)["occlusion"]
    counts = [summary["maps"][name]["inked_patches"] for name in ("s", "m", "f")]

    assert counts[0] < counts[1] < counts[2], "three different spreads, so median != max"
    assert summary["inked_patches_median"] == counts[1]


def test_only_the_first_occlude_correct_drawings_are_occluded():
    constant = lambda image, prompts: [0.9] + [0.1] * (len(prompts) - 1)

    summary = measure(drawings(), constant, PROMPTS, prompt_sets={}, occlude=1)["occlusion"]

    assert summary["drawings"] == len(summary["maps"]) == 1


def test_a_blank_canvas_has_no_max_drop_rather_than_a_zero():
    blank = np.full((RENDER.size, RENDER.size), RENDER.background, dtype=np.uint8)
    constant = lambda image, prompts: [0.9] + [0.1] * (len(prompts) - 1)

    result = occlusion(blank, 0, constant, PROMPTS, RENDER)

    assert result["inked_patches"] == 0 and result["max_drop"] is None and result["top3_share"] is None


def test_a_hit_is_the_fifth_place_but_not_the_sixth_and_matches_the_gated_metric():
    from primer.drawing import evaluate

    confidences = [0.9 - 0.1 * i for i in range(8)]
    ranked = scored(confidences)

    assert sensitivity.TOP_K == evaluate.TOP_K == 5
    assert sensitivity.hit(ranked, 4) and not sensitivity.hit(ranked, 5)


def test_occlusion_paints_paper_white_not_some_other_grey():
    canvas = render_strokes(THREE, RENDER)
    scorer = InkScorer()

    occlusion(canvas, 0, scorer, PROMPTS, RENDER)

    values = set(np.unique(np.concatenate([c.ravel() for c, _ in scorer.calls])))
    assert values <= set(np.unique(canvas)), "an occluder adds no level the drawing did not have"


# ------------------------------------------------------------------------------------- measure


def drawings() -> list[Drawing]:
    return [Drawing("a", THREE, 0), Drawing("b", ONE, 0), Drawing("c", THREE, 3)]


def test_each_perturbation_is_compared_over_the_drawings_it_applies_to():
    result = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={}, seed=0, occlude=0)
    figures = result["perturbations"]

    assert figures["drop_last_1"]["drawings"] == 2, "a one-stroke drawing cannot lose its last stroke"
    assert figures["drop_last_2"]["drawings"] == 2
    assert all(figures[f"line_width_{w}"]["drawings"] == 3 for w in sensitivity.LINE_WIDTHS)
    assert set(figures) == {
        "drop_last_1",
        "drop_last_2",
        "line_width_1",
        "line_width_5",
        "line_width_7",
        "erasure",
    }


def test_dropping_strokes_removes_ink_and_costs_the_hit_it_depended_on():
    # Tune the fake so label 0 scores 0.455 once the last stroke is gone: just under label 5's 0.46,
    # so out of the top five. With the tick it scores 0.455 * full / minus_one, which must clear
    # 0.47 for the hit to exist before the drop and not after.
    full = ink_mass(render_strokes(THREE, RENDER), RENDER)
    minus_one = ink_mass(render_strokes(THREE[:-1], RENDER), RENDER)
    minus_two = ink_mass(render_strokes(THREE[:-2], RENDER), RENDER)
    assert full > 1.04 * minus_one > minus_two, "each stroke must carry real ink"
    scorer = InkScorer(gain=0.455 * FULL / minus_one)

    result = measure(drawings(), scorer, PROMPTS, prompt_sets={}, seed=0, occlude=0)
    figures = result["perturbations"]

    # Drawing "a" loses its hit; "c" (label 3) is in the top five whatever the ink; "b" has one stroke.
    assert figures["drop_last_1"]["top5_delta"] == -0.5
    assert figures["drop_last_2"]["top5_delta"] == -0.5
    assert figures["drop_last_1"]["top5_acc_before"] == 1.0
    assert figures["drop_last_1"]["top5_acc_after"] == 0.5


def test_k_strokes_are_dropped_from_the_end_and_each_k_scores_its_own_canvas():
    scorer = InkScorer()

    measure([Drawing("a", THREE, 0)], scorer, PROMPTS, prompt_sets={}, seed=0, occlude=0)

    scored_canvases = [canvas for canvas, _ in scorer.calls]

    def was_scored(strokes) -> bool:
        expected = render_strokes(strokes, RENDER)
        return any(np.array_equal(canvas, expected) for canvas in scored_canvases)

    assert was_scored(THREE[:-1]), "drop_last_1 scores the drawing without its last stroke"
    assert was_scored(THREE[:-2]), "drop_last_2 scores it without its last two, not just one"
    # Dropping from the front would score these instead, and the two drawings are different ones.
    assert not np.array_equal(render_strokes(THREE[:-1], RENDER), render_strokes(THREE[1:], RENDER))
    assert not np.array_equal(render_strokes(THREE[:-2], RENDER), render_strokes(THREE[2:], RENDER))
    assert not np.array_equal(render_strokes(THREE[:-1], RENDER), render_strokes(THREE[:-2], RENDER))


def test_a_drawing_that_falls_out_of_the_top_five_only_at_two_strokes_is_counted_only_for_two():
    full = ink_mass(render_strokes(THREE, RENDER), RENDER)
    minus_one = ink_mass(render_strokes(THREE[:-1], RENDER), RENDER)
    minus_two = ink_mass(render_strokes(THREE[:-2], RENDER), RENDER)
    assert full > minus_one > minus_two
    # Label 0 scores 0.465 with two strokes left (rank 5, a hit) and below 0.46 with one.
    scorer = InkScorer(gain=0.465 * FULL / minus_one)

    figures = measure([Drawing("a", THREE, 0)], scorer, PROMPTS, prompt_sets={}, seed=0, occlude=0)[
        "perturbations"
    ]

    assert figures["drop_last_1"]["top5_delta"] == 0.0
    assert figures["drop_last_2"]["top5_delta"] == -1.0


def test_full_erasure_removes_the_ink_and_costs_the_hit():
    # Many whole-strength rectangles, each up to the full canvas: nearly all the ink goes.
    everything = replace(
        erasure_only(),
        erasure_patches=60,
        erasure_size=1.0,
        erasure_strength_min=1.0,
        erasure_strength_max=1.0,
    )

    result = measure(
        drawings(), InkScorer(), PROMPTS, prompt_sets={}, seed=0, occlude=0, augmentation=everything
    )
    erasure = result["perturbations"]["erasure"]

    assert erasure["ink_removed_median"] > 0.9
    assert erasure["drawings"] == 3 and erasure["below_floor"] == 0
    # "a" and "b" lose label 0's top-five place with almost no ink; "c" (label 3) does not.
    assert erasure["top5_delta"] < 0
    assert erasure["top5_acc_after"] < erasure["top5_acc_before"]


def test_the_erasure_floor_is_inclusive_and_counts_the_misses(monkeypatch):
    fractions = {0: 0.01, 1: sensitivity.ERASURE_FLOOR, 2: 0.5}

    def fake_erased(canvas, seed, config):
        return canvas, fractions[seed]

    monkeypatch.setattr(sensitivity, "erased", fake_erased)

    result = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={}, seed=0, occlude=0)
    erasure = result["perturbations"]["erasure"]

    assert erasure["drawings"] == 2, "0.05 is at the floor and counts; 0.01 does not"
    assert erasure["below_floor"] == 1.0
    assert erasure["ink_removed_median"] == sensitivity.ERASURE_FLOOR


def test_thinner_lines_lose_ink_and_so_lose_the_fake_label():
    # Label 0's score is its ink, so 1px strokes drop it out of the top five on drawing "a".
    result = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={}, seed=0, occlude=0)

    thin, thick = result["perturbations"]["line_width_1"], result["perturbations"]["line_width_7"]
    assert thin["top5_delta"] < 0 <= thick["top5_delta"]


def test_erasure_reports_the_ink_it_removed_and_skips_misses():
    result = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={}, seed=0, occlude=0)
    erasure = result["perturbations"]["erasure"]

    assert erasure["floor"] == sensitivity.ERASURE_FLOOR
    assert erasure["drawings"] + erasure["below_floor"] == 3
    assert 0.0 <= erasure["ink_removed_median"] <= 1.0


def test_prompt_sets_score_the_unperturbed_canvases_with_their_own_prompts():
    scorer = InkScorer()
    other = tuple(f"q{i}" for i in range(8))

    result = measure(drawings(), scorer, PROMPTS, prompt_sets={"alt {label}": other}, seed=0, occlude=0)

    assert result["perturbations"]["prompt:alt {label}"]["drawings"] == 3
    assert sum(1 for _, prompts in scorer.calls if prompts == other) == 3


def test_only_drawings_the_baseline_ranks_correctly_are_occluded():
    scorer = InkScorer()
    first = scored(scorer(render_strokes(THREE, RENDER), PROMPTS))
    assert 0 in first.ranking[:5] and 3 in first.ranking[:5]

    result = measure(
        [Drawing("a", THREE, 0), Drawing("x", THREE, 7)], scorer, PROMPTS, prompt_sets={}, occlude=5
    )

    assert list(result["occlusion"]["maps"]) == ["a"], "label 7 is ranked last, so 'x' is not occluded"


def test_measure_is_reproducible_for_a_seed():
    one = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={}, seed=4, occlude=1)
    two = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={}, seed=4, occlude=1)

    assert one == two


def test_no_drawings_is_an_error():
    with pytest.raises(ValueError, match="no drawings"):
        measure([], InkScorer(), PROMPTS, prompt_sets={})


# ------------------------------------------------------------------------------------- MLflow


#: The names main's own logger gave this fixture before S6f2 (captured from `sensitivity.mlflow_metrics`
#: at a38c7ab). Runs d99d75b5 and dfaee7e2 carry these names; a renamed metric would split the history.
PRE_S6F2_FIGURES = (
    "confident_drawings",
    "drawings",
    "top1_change_rate",
    "top1_change_rate_confident",
    "top5_acc_after",
    "top5_acc_before",
    "top5_delta",
    "top5_delta_ci_high",
    "top5_delta_ci_low",
    "top5_overlap",
)
PRE_S6F2_NAMES = (
    {"baseline_top5_acc"}
    | {
        f"sensitivity/{name}/{figure}"
        for name in ("drop_last_1", "drop_last_2", "line_width_1", "line_width_5", "line_width_7", "prompt_0")
        for figure in PRE_S6F2_FIGURES
    }
    | {f"sensitivity/erasure/{f}" for f in ("below_floor", "drawings", "floor", "ink_removed_median")}
    | {
        f"sensitivity/occlusion/{f}"
        for f in ("inked_patches_median", "max_drop_median", "top3_share_undefined")
    }
)


def test_metric_names_are_unchanged_by_the_move_to_tracking():
    result = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={"a {label}": PROMPTS}, seed=0, occlude=1)

    names = sensitivity_metrics(result)

    assert set(names) == PRE_S6F2_NAMES and len(PRE_S6F2_NAMES) == 68
    assert all(":" not in n and "=" not in n for n in names), "MLflow refuses ':' and '=' in metric names"


def test_sensitivity_no_longer_imports_mlflow_itself():
    import ast
    import inspect

    imported = {
        alias.name.split(".")[0] if isinstance(node, ast.Import) else (node.module or "").split(".")[0]
        for node in ast.walk(ast.parse(inspect.getsource(sensitivity)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert "mlflow" not in imported, "logging goes through tracking.log_run, one set of rules"
    assert "mlflow" not in vars(sensitivity)
    from primer.drawing import evaluate

    assert not hasattr(sensitivity, "mlflow_metrics")
    assert sensitivity.log_to_mlflow is evaluate.log_to_mlflow, "the shared logger, not a second copy"


# ------------------------------------------------------------------------------------- CLI


def _val_frame(rows: int) -> pd.DataFrame:
    from primer.drawing.data import split_of

    ids = [f"s{i}" for i in range(rows * 40) if split_of(f"s{i}") == "val"][:rows]
    return pd.DataFrame(
        {
            "record_id": [f"r-{i}" for i in ids],
            "source_id": ids,
            "label": ["the eiffel tower"] * rows,
            "strokes": json.dumps(THREE),
            "license": "CC-BY-4.0",
            "attribution": "Quick, Draw!",
        }
    )


def test_the_cli_runs_end_to_end_on_a_stubbed_world(monkeypatch, tmp_path):
    from primer.data import catalog, storage
    from primer.drawing import model

    seen: list[bytes] = []

    class FakeEncoder:
        def __init__(self, config, device=None):
            self.device = device or "cpu"
            self.dtype_name = "float32"

        def encode(self, image, prompts):
            seen.append(image)
            return SimpleNamespace(confidences=[0.5 - 0.001 * i for i in range(len(prompts))])

    monkeypatch.setattr(catalog, "read_dataset", lambda *a, **k: _val_frame(4))
    monkeypatch.setattr(storage, "Store", lambda: SimpleNamespace(uri="gs://bucket"))
    monkeypatch.setattr(model, "SiglipEncoder", FakeEncoder)
    monkeypatch.setattr(sensitivity, "_git_state", lambda: ("abc1234", True))
    logged: dict[str, object] = {}

    class Run:
        info = SimpleNamespace(run_id="r1")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(
            set_experiment=lambda name: None,
            start_run=lambda run_name, tags: logged.__setitem__("tags", tags) or Run(),
            log_params=lambda params: None,
            log_metrics=lambda metrics: logged.__setitem__("metrics", metrics),
            log_artifact=lambda path, artifact_path=None: logged.setdefault("artifacts", []).append(
                (Path(path).name, artifact_path)
            ),
        ),
    )
    out = tmp_path / "sens.json"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("YIP_WHERE", raising=False)

    code = sensitivity.main(
        [
            "--version",
            "v1.1",
            "--sample",
            "3",
            "--occlude",
            "1",
            "--mlflow",
            "--owner",
            "m1",
            "--out",
            str(out),
        ]
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and result["drawings"] == 3 and result["split"] == "val"
    assert all(image.startswith(b"\x89PNG") for image in seen), "the encoder gets PNG bytes, as an upload"
    assert result["license"] == ["CC-BY-4.0"] and result["attribution"] == ["Quick, Draw!"]
    # Every alternative except the shipped template, which is the baseline itself.
    shipped = result["shipped_prompt_template"]
    assert shipped in sensitivity.PROMPT_TEMPLATES
    compared = [k for k in result["perturbations"] if k.startswith("prompt:")]
    assert len(compared) == len(sensitivity.PROMPT_TEMPLATES) - 1
    assert not any(shipped in k for k in compared)
    assert result["device"] == "cpu" and result["dtype"] == "float32"
    # The commit and dirty flag come from `_git_state`, in the file and in the MLflow tags.
    assert (result["git_sha"], result["git_dirty"]) == ("abc1234", True)
    assert logged["tags"]["git_sha"] == "abc1234" and logged["tags"]["git_dirty"] == "true"
    # The tags model-development.md requires on every run.
    required = {"stage", "owner", "data_version", "git_sha", "base_model", "gate_group", "where"}
    assert required <= logged["tags"].keys()
    assert "sensitivity/occlusion/top3_share_undefined" in logged["metrics"]
    # Through tracking.log_run (S6f2): the owner given, the shared kind tags, the file and the card.
    assert logged["tags"]["owner"] == "m1" and logged["tags"]["kind"] == "sensitivity"
    assert logged["tags"]["gate_eligible"] == "false"
    # Exactly the shared tags, nothing written by hand beside them (cross-slice-coherence S0).
    from primer.drawing.tracking import run_tags

    assert logged["tags"] == run_tags(result, owner="m1", kind="sensitivity")
    assert (out.name, "eval") in logged["artifacts"] and ("CARD.md", None) in logged["artifacts"]


def test_the_shipped_template_is_the_one_every_recorded_figure_was_measured_under():
    from primer.drawing.model import DrawingConfig

    config = DrawingConfig.from_yaml()
    digest = hashlib.sha256("\n".join(config.prompts()).encode()).hexdigest()

    assert config.prompt_template == "a child's drawing of a {label}."
    assert digest.startswith("d3425b53"), (
        "the plan's S6c (0.5436) and S6c2 figures, and the card, were measured under this prompt set; "
        "changing the template means re-measuring them (Decisions, 2026-11-16)"
    )


def test_the_cli_refuses_the_training_split():
    with pytest.raises(SystemExit):
        sensitivity.parser().parse_args(["--version", "v1.1", "--split", "train", "--out", "x.json"])


def test_render_and_preprocess_are_imported_not_reimplemented():
    from primer.drawing import preprocess, render

    assert sensitivity.render_strokes is render.render_strokes
    assert sensitivity.augment_canvas is preprocess.augment_canvas


@pytest.mark.parametrize(
    "extra, env, message",
    [
        (["--mlflow"], {"MLFLOW_TRACKING_URI": "http://localhost:5000"}, "--owner is required"),
        (["--mlflow", "--owner", "m1"], {}, "MLFLOW_TRACKING_URI"),
    ],
    ids=["no-owner", "no-tracking-server"],
)
def test_a_bad_mlflow_request_is_refused_before_anything_is_read(
    monkeypatch, tmp_path, capsys, extra, env, message
):
    from primer.data import catalog
    from primer.drawing import model

    def boom(*args, **kwargs):
        raise AssertionError("the bucket and the GPU must not be touched")

    monkeypatch.setattr(catalog, "read_dataset", boom)
    monkeypatch.setattr(model, "SiglipEncoder", boom)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(SystemExit):
        sensitivity.main(["--version", "v1.1", "--out", str(tmp_path / "s.json"), *extra])
    assert message in capsys.readouterr().err
