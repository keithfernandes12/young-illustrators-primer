"""S6c's `drawing_top5_acc` in `evaluate.py` and `slices.py`, against a fake encoder.

No weights, no bucket, no Pillow: each "drawing" is a token the fake maps to a fixed score vector,
so every hit can be checked by hand. The last test runs the real release gate on a file this code
wrote, which is the slice's acceptance.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from primer.data.bias import SMALL_SLICE
from primer.drawing import evaluate, slices
from primer.drawing.evaluate import accuracy_by_slice, chance_baseline, metrics_file, top5_hits

ROOT = Path(__file__).resolve().parents[2]
CHECK_GATE = ROOT / ".claude" / "skills" / "golden-gate" / "scripts" / "check_gate.py"
PROMPTS = tuple(f"label {i}" for i in range(8))


class TableEncoder:
    """Scores each drawing with the vector registered for its bytes."""

    def __init__(self, table: dict[bytes, Sequence[float]]) -> None:
        self.table = table
        self.calls: list[tuple[bytes, tuple[str, ...]]] = []

    def encode(self, image: bytes, prompts: Sequence[str]) -> SimpleNamespace:
        self.calls.append((image, tuple(prompts)))
        return SimpleNamespace(confidences=self.table[image])


#: Label i scores (8 - i) / 10, so the ranking is 0, 1, 2, ... and the top five are 0-4.
DESCENDING = tuple((8 - i) / 10 for i in range(8))


# ------------------------------------------------------------------------------------------- hits


def test_a_label_ranked_fifth_is_a_hit_and_sixth_is_a_miss():
    encoder = TableEncoder({b"d": DESCENDING})

    hits = top5_hits([(b"d", 0), (b"d", 4), (b"d", 5), (b"d", 7)], encoder, PROMPTS)

    assert hits == [True, True, False, False]
    assert encoder.calls == [(b"d", PROMPTS)] * 4


def test_ties_break_by_vocabulary_order_so_the_same_scores_give_the_same_hits():
    # Six labels tied: vocabulary order puts labels 0-4 in the top five and 5 out of it.
    encoder = TableEncoder({b"tie": (0.5,) * 6 + (0.1, 0.1)})

    assert top5_hits([(b"tie", 4), (b"tie", 5)], encoder, PROMPTS) == [True, False]


def test_the_metric_reads_five_guesses_whatever_the_config_top_k():
    assert evaluate.TOP_K == 5


def test_no_drawings_is_an_error_not_a_score():
    with pytest.raises(ValueError, match="no drawings"):
        top5_hits([], TableEncoder({}), PROMPTS)


def test_a_score_outside_0_1_is_refused():
    encoder = TableEncoder({b"bad": (float("nan"),) + DESCENDING[1:]})

    with pytest.raises(ValueError, match="sigmoid"):
        top5_hits([(b"bad", 0)], encoder, PROMPTS)


# --------------------------------------------------------------------------------------- baseline


def test_the_baseline_is_chance_five_of_the_vocabulary():
    assert chance_baseline(345)["drawing_top5_acc"] == 0.0145
    assert "chance" in chance_baseline(345)["kind"]


def test_a_vocabulary_of_five_or_fewer_makes_chance_certain():
    assert chance_baseline(3)["drawing_top5_acc"] == 1.0


def test_an_empty_vocabulary_has_no_baseline():
    with pytest.raises(ValueError, match="empty"):
        chance_baseline(0)


# ----------------------------------------------------------------------------------------- slices


def test_each_slice_value_gets_its_own_accuracy_and_small_ones_are_not_judged():
    big = SMALL_SLICE
    hits = [True] * big + [False, True]
    groups = {"country_code": ["US"] * big + ["NZ", "NZ"]}

    result = accuracy_by_slice(hits, groups)

    assert result == [
        {"key": "country_code", "value": "NZ", "n": 2, "judged": False, "metrics": {"drawing_top5_acc": 0.5}},
        {
            "key": "country_code",
            "value": "US",
            "n": big,
            "judged": True,
            "metrics": {"drawing_top5_acc": 1.0},
        },
    ]
    assert [s["value"] for s in slices.judged(result)] == ["US"]


def test_a_slice_key_with_the_wrong_number_of_values_is_refused():
    with pytest.raises(ValueError, match="2 values for 3 drawings"):
        accuracy_by_slice([True, True, False], {"country_code": ["US", "GB"]})


def test_slice_values_name_a_missing_country_unknown():
    frame = pd.DataFrame({"country_code": ["US", None, "GB"]})

    assert list(slices.slice_values(frame, "country_code")) == ["US", "unknown", "GB"]


def test_slicing_by_an_absent_column_raises_rather_than_making_one_slice():
    with pytest.raises(KeyError, match="country_code"):
        slices.slice_values(pd.DataFrame({"label": ["cat"]}), "country_code")


def test_slice_of_reads_one_record_and_says_none_when_it_is_absent():
    assert slices.slice_of({"country_code": "US"}, "country_code") == "US"
    assert slices.slice_of({"country_code": float("nan")}, "country_code") is None
    assert slices.slice_of({}, "country_code") is None


def test_country_code_is_the_reported_slice_key():
    assert slices.SLICE_KEYS == ("country_code",)


# ----------------------------------------------------------------------------------- metrics file


def _file(hits: list[bool], **provenance: object) -> dict:
    return metrics_file(hits, {"country_code": ["US"] * len(hits)}, 345, **provenance)


def test_the_file_is_the_contract_shape_with_the_gated_key_also_at_the_top():
    result = _file([True, False, True, True], data_version="v1.1")

    assert result["stage"] == "drawing"
    assert result["overall"] == {"drawing_top5_acc": 0.75, "drawings": 4}
    assert result["drawing_top5_acc"] == result["overall"]["drawing_top5_acc"]
    assert result["baseline"]["drawing_top5_acc"] == 0.0145
    assert result["slices"][0]["value"] == "US"
    assert result["data_version"] == "v1.1"
    assert set(evaluate.OWNED_METRICS) <= set(result["overall"])


def test_a_limited_probe_cannot_be_gated(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(_file([True] * 10, whole_split=False)), encoding="utf-8")

    run = subprocess.run(
        [sys.executable, str(CHECK_GATE), "--metrics", str(path), "--only", "drawing"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "metric missing" in run.stdout
    assert "GATE_RESULT: FAIL" in run.stdout


def test_the_file_never_claims_to_be_fake_scored():
    # check_gate reads `fake_models` for the whole file; a real number must not carry it.
    assert "fake_models" not in _file([True])


@pytest.mark.parametrize(("hits", "verdict"), [([True] * 9 + [False], "PASS"), ([True, False], "FAIL")])
def test_the_release_gate_returns_a_verdict_on_this_file_not_metric_missing(tmp_path, hits, verdict):
    path = tmp_path / "drawing_metrics.json"
    path.write_text(json.dumps(_file(hits)), encoding="utf-8")

    run = subprocess.run(
        [sys.executable, str(CHECK_GATE), "--metrics", str(path), "--only", "drawing"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "metric missing" not in run.stdout
    assert f"GATE_RESULT: {verdict}" in run.stdout
    assert "FAKE" not in run.stdout


# --------------------------------------------------------------------------------------------- CLI


def _stub_world(monkeypatch, frame: pd.DataFrame, winner: str) -> list[object]:
    """The bucket and the tower replaced; rendering, splitting and label mapping stay real."""
    from primer.data import catalog, storage
    from primer.drawing import model

    labels = model.DrawingConfig.from_yaml().labels()
    devices: list[object] = []

    class FakeEncoder:
        def __init__(self, config, device=None):
            devices.append(device)
            self.device = device or "auto-picked"
            self.dtype_name = "bfloat16" if device == "cuda" else "float32"
            self.prepare = None

        def encode(self, image, prompts):
            assert image.startswith(b"\x89PNG"), "the encoder gets PNG bytes, as an upload"
            scores = [0.1] * len(prompts)
            scores[labels.index(winner)] = 0.9
            return SimpleNamespace(confidences=scores)

    monkeypatch.setattr(catalog, "read_dataset", lambda *args, **kwargs: frame)
    monkeypatch.setattr(storage, "Store", lambda: SimpleNamespace(uri="gs://bucket"))
    monkeypatch.setattr(model, "SiglipEncoder", FakeEncoder)
    return devices


def _test_split_frame(rows: int) -> pd.DataFrame:
    """``rows`` drawings whose source ids the real split function puts in ``test``."""
    from primer.drawing.data import split_of

    ids = [f"s{i}" for i in range(rows * 30) if split_of(f"s{i}") == "test"][:rows]
    half = rows // 2
    return pd.DataFrame(
        {
            "record_id": [f"r-{i}" for i in ids],
            "source_id": ids,
            # The data's lower-case spelling; the vocabulary's is `The Eiffel Tower`.
            "label": ["the eiffel tower"] * half + ["zigzag"] * (rows - half),
            "strokes": json.dumps([[[0, 50, 100], [0, 80, 20]]]),
            "country_code": ["US"] * half + ["GB"] * (rows - half),
            "license": "CC-BY-4.0",
            "attribution": "Quick, Draw!",
        }
    )


def test_the_cli_scores_the_test_split_end_to_end(monkeypatch, tmp_path):
    devices = _stub_world(monkeypatch, _test_split_frame(6), winner="The Eiffel Tower")
    out = tmp_path / "m.json"
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    code = evaluate.main(
        ["--model", str(model_yaml), "--version", "v1.1", "--device", "cuda", "--out", str(out)]
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and devices == ["cuda"]
    # Eiffel is top-1 (a hit despite the case); `zigzag` sits far down a tie-broken ranking.
    assert result["drawing_top5_acc"] == 0.5 and result["overall"]["drawings"] == 6
    assert {s["value"]: s["metrics"]["drawing_top5_acc"] for s in result["slices"]} == {"GB": 0.0, "US": 1.0}
    assert result["device"] == "cuda" and result["dtype"] == "bfloat16"
    assert result["split"] == "test" and result["data_version"] == "v1.1" and result["limit"] is None
    assert result["license"] == ["CC-BY-4.0"] and result["attribution"] == ["Quick, Draw!"]


def test_a_limited_cli_run_writes_no_gated_key(monkeypatch, tmp_path):
    _stub_world(monkeypatch, _test_split_frame(6), winner="The Eiffel Tower")
    out = tmp_path / "probe.json"
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--limit", "2", "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert "drawing_top5_acc" not in result
    assert result["overall"] == {"drawing_top5_acc": 1.0, "drawings": 2} and result["limit"] == 2


def test_the_cli_refuses_the_training_split():
    with pytest.raises(SystemExit):
        evaluate.accuracy_parser().parse_args(
            ["--model", "m.yaml", "--version", "v1.1", "--split", "train", "--out", "x.json"]
        )


def test_the_cli_needs_a_frozen_version():
    with pytest.raises(SystemExit):
        evaluate.accuracy_parser().parse_args(["--model", "m.yaml", "--out", "x.json"])


def test_stability_still_routes_to_its_own_subcommand(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(evaluate, "_stability", lambda argv: seen.append(argv) or 0)

    assert evaluate.main(["stability", "--version", "v1.1", "--out", "s.json"]) == 0
    assert seen == [["--version", "v1.1", "--out", "s.json"]]
