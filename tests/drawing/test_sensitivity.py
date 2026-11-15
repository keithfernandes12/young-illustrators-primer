"""S6f's sensitivity analysis in `drawing/sensitivity.py`, against a fake scorer.

No weights, no bucket, no Pillow, no MLflow: the "recogniser" scores a canvas by how much ink it
holds, so every perturbation has an effect that can be predicted by hand.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
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
    mlflow_metrics,
    occlusion,
    paired_delta,
)

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
    from dataclasses import replace

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


def test_metric_names_are_ones_mlflow_accepts():
    result = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={"a {label}": PROMPTS}, seed=0, occlude=1)

    names = mlflow_metrics(result)

    assert "sensitivity/prompt_0/top5_delta" in names
    assert all(":" not in n and "=" not in n for n in names), "MLflow refuses ':' and '=' in metric names"


def test_logging_writes_one_run_with_the_templates_and_the_report(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    class Run:
        info = SimpleNamespace(run_id="r1")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = SimpleNamespace(
        set_experiment=lambda name: calls.__setitem__("experiment", name),
        start_run=lambda run_name, tags: calls.__setitem__("tags", tags) or Run(),
        log_params=lambda params: calls.setdefault("params", {}).update(params),
        log_metrics=lambda metrics: calls.__setitem__("metrics", metrics),
        log_artifact=lambda path, artifact_path: calls.__setitem__("artifact", (path, artifact_path)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    result = measure(drawings(), InkScorer(), PROMPTS, prompt_sets={"a {label}": PROMPTS}, seed=0, occlude=0)
    result.update({"split": "val", "seed": 0})
    out = tmp_path / "s.json"

    run_id = sensitivity.log_to_mlflow(result, out, {"stage": "drawing"})

    assert run_id == "r1" and calls["experiment"] == "yip-drawing"
    assert calls["params"]["prompt_0"] == "a {label}"
    assert calls["artifact"] == (str(out), "sensitivity")
    assert calls["tags"] == {"stage": "drawing"}


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
    out = tmp_path / "sens.json"

    code = sensitivity.main(["--version", "v1.1", "--sample", "3", "--occlude", "1", "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and result["drawings"] == 3 and result["split"] == "val"
    assert all(image.startswith(b"\x89PNG") for image in seen), "the encoder gets PNG bytes, as an upload"
    assert result["license"] == ["CC-BY-4.0"] and result["attribution"] == ["Quick, Draw!"]
    assert len([k for k in result["perturbations"] if k.startswith("prompt:")]) == len(
        sensitivity.PROMPT_TEMPLATES
    )
    assert result["device"] == "cpu" and result["dtype"] == "float32"


def test_the_cli_refuses_the_training_split():
    with pytest.raises(SystemExit):
        sensitivity.parser().parse_args(["--version", "v1.1", "--split", "train", "--out", "x.json"])


def test_render_and_preprocess_are_imported_not_reimplemented():
    from primer.drawing import preprocess, render

    assert sensitivity.render_strokes is render.render_strokes
    assert sensitivity.augment_canvas is preprocess.augment_canvas
