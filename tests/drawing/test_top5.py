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
from dataclasses import asdict
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


def _file(hits: list[bool], gated: bool = True, **provenance: object) -> dict:
    return metrics_file(hits, {"country_code": ["US"] * len(hits)}, 345, gated=gated, **provenance)


def test_gating_is_a_required_choice_not_a_default():
    # A caller that forgets the flag must fail, not write a file the gate can certify.
    with pytest.raises(TypeError, match="gated"):
        metrics_file([True], {"country_code": ["US"]}, 345)


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
    path.write_text(json.dumps(_file([True] * 10, gated=False)), encoding="utf-8")

    run = subprocess.run(
        [sys.executable, str(CHECK_GATE), "--metrics", str(path), "--only", "drawing"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "metric missing" in run.stdout
    assert "GATE_RESULT: FAIL" in run.stdout


def test_the_gated_ratio_is_not_rounded_so_a_near_miss_cannot_round_up_to_a_pass(tmp_path):
    # 53,811 of 63,310 is 0.8499605: rounded to four places it reads 0.85 and the gate passes it.
    hits = [True] * 53_811 + [False] * (63_310 - 53_811)
    result = _file(hits)

    assert result["drawing_top5_acc"] < 0.85
    assert result["overall"]["drawing_top5_acc"] == result["drawing_top5_acc"] == 53_811 / 63_310
    path = tmp_path / "near_miss.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(CHECK_GATE), "--metrics", str(path), "--only", "drawing"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "GATE_RESULT: FAIL" in run.stdout


def test_the_file_says_whether_it_is_gate_eligible():
    assert _file([True])["gate_eligible"] is True
    probe = _file([True], gated=False)
    assert probe["gate_eligible"] is False and "drawing_top5_acc" not in probe


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize(
    "key", ["overall", "slices", "baseline", "stage", "gate_eligible", "drawing_top5_acc", "fake_models"]
)
def test_provenance_cannot_set_a_key_the_writer_owns(key, gated):
    # With gated=False the top-level `drawing_top5_acc` is absent, so ordering alone would let a
    # provenance key of that name put a gate-readable number into an ineligible file.
    with pytest.raises(ValueError, match="writer owns"):
        _file([True, False], gated=gated, **{key: 0.99})


def test_the_file_never_claims_to_be_fake_scored():
    # check_gate reads `fake_models` for the whole file; a real number must not carry it.
    assert "fake_models" not in _file([True])


@pytest.mark.parametrize(("hits", "verdict"), [([True] * 10, "PASS"), ([True, False], "FAIL")])
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
            # The winner's neighbour is ranked last. Without this, a label index off by one lands
            # in the top five anyway, because ties fall back to vocabulary order and the winner is
            # index 0: the mapping from label to index would go untested.
            scores[labels.index(winner) + 1] = 0.0
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
    assert result["gate_eligible"] is True and result["data_uri"] == "gs://bucket"
    assert isinstance(result["git_dirty"], bool) and result["git_sha"] != "unknown"
    assert result["preprocessing"] == asdict(evaluate.default_prepare_config())


def _rows(split: str, label: str, count: int, tag: str) -> pd.DataFrame:
    """``count`` drawings of ``label`` whose source ids the real split function puts in ``split``."""
    from primer.drawing.data import split_of

    ids = [f"{tag}{i}" for i in range(count * 40) if split_of(f"{tag}{i}") == split][:count]
    return pd.DataFrame(
        {
            "record_id": [f"r-{i}" for i in ids],
            "source_id": ids,
            "label": label,
            "strokes": json.dumps([[[0, 50, 100], [0, 80, 20]]]),
            "country_code": "US",
            "license": "CC-BY-4.0",
            "attribution": "Quick, Draw!",
        }
    )


def _mixed_frame() -> pd.DataFrame:
    """Every split present. Only the rows of the split being scored may count.

    The stub's winner is `The Eiffel Tower`. In `test` and `val` three drawings are Eiffel Towers
    (spelled as the vocabulary does, so the case-fold has to cope with it) and two are zigzags, so
    each scored split reads 0.6; the other splits are all hits, so any leak raises the number.
    """
    parts = []
    for split, tag in (("test", "t"), ("val", "v"), ("train", "r")):
        miss = 0 if split == "train" else 2
        parts.append(_rows(split, "The Eiffel Tower", 3 if split != "train" else 7, tag))
        if miss:
            parts.append(_rows(split, "zigzag", miss, tag + "z"))
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize(("split", "gated"), [("test", True), ("val", False)])
def test_only_the_asked_split_is_scored(monkeypatch, tmp_path, split, gated):
    _stub_world(monkeypatch, _mixed_frame(), winner="The Eiffel Tower")
    out = tmp_path / "m.json"
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--split", split, "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["overall"] == {"drawing_top5_acc": 0.6, "drawings": 5}
    assert ("drawing_top5_acc" in result) is gated


def test_a_drawing_that_cannot_be_rendered_is_named_and_aborts_the_run(monkeypatch, tmp_path):
    frame = _rows("test", "The Eiffel Tower", 2, "t").assign(strokes="not json")
    _stub_world(monkeypatch, frame, winner="The Eiffel Tower")
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    with pytest.raises(ValueError, match=r"record r-t\w* could not be rendered"):
        evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--out", str(tmp_path / "m.json")])


def test_a_failed_rerun_does_not_leave_the_previous_file_to_be_gated(monkeypatch, tmp_path):
    _stub_world(monkeypatch, _rows("test", "The Eiffel Tower", 2, "t").iloc[0:0], winner="The Eiffel Tower")
    out = tmp_path / "m.json"
    out.write_text('{"stage": "drawing", "drawing_top5_acc": 0.99}', encoding="utf-8")
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    with pytest.raises(ValueError, match="holds no Quick, Draw"):
        evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--out", str(out)])

    assert not out.exists()
    # Moved aside, not destroyed: it is the 12-minute result a probe reusing the path must not lose.
    assert (tmp_path / "m.json.previous").read_text(encoding="utf-8").count("0.99") == 1


@pytest.mark.parametrize(
    "bad_args",
    [["--limit", "0"], ["--version", ""], ["--model", "no_such_config.yaml"]],
    ids=["limit", "version", "model"],
)
def test_a_refused_rerun_does_not_leave_the_previous_file_to_be_gated(tmp_path, bad_args):
    out = tmp_path / "m.json"
    out.write_text('{"stage": "drawing", "drawing_top5_acc": 0.99}', encoding="utf-8")
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"
    args = {"--model": str(model_yaml), "--version": "v1.1", "--out": str(out)}
    args.update(dict(zip(bad_args[::2], bad_args[1::2], strict=True)))

    with pytest.raises((SystemExit, FileNotFoundError)):
        evaluate.main([item for pair in args.items() for item in pair])

    assert not out.exists()


def test_reusing_the_path_twice_keeps_every_earlier_result(tmp_path):
    out = tmp_path / "m.json"
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"
    for marker in ("first", "second", "third"):
        out.write_text(json.dumps({"stage": "drawing", "marker": marker}), encoding="utf-8")
        with pytest.raises(SystemExit):
            evaluate.main(["--model", str(model_yaml), "--version", "", "--out", str(out)])

    kept = {json.loads(p.read_text(encoding="utf-8"))["marker"] for p in tmp_path.glob("m.json.previous*")}
    assert kept == {"first", "second", "third"} and not out.exists()


def test_a_crash_while_writing_leaves_nothing_at_out(monkeypatch, tmp_path):
    _stub_world(monkeypatch, _test_split_frame(6), winner="The Eiffel Tower")
    out = tmp_path / "m.json"
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"
    real_write = Path.write_text

    def half_then_die(self, data, *args, **kwargs):
        real_write(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", half_then_die)

    with pytest.raises(OSError, match="disk full"):
        evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--out", str(out)])

    assert not out.exists()


def test_an_out_path_that_is_not_a_previous_metrics_file_is_refused_and_kept(tmp_path):
    other = tmp_path / "config.yaml"
    other.write_text("name: not a metrics file\n", encoding="utf-8")
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    with pytest.raises(SystemExit):
        evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--out", str(other)])

    assert other.read_text(encoding="utf-8") == "name: not a metrics file\n"


@pytest.mark.parametrize(
    "content",
    ['{"fake_models": true, "drawing_top5_acc": 0.9}', '{"top5_overlap": 0.4}', '{"stage": "speech"}', "[]"],
    ids=["golden-like", "stability-like", "other-stage", "list"],
)
def test_a_json_file_that_is_not_a_drawing_result_is_refused_and_kept(tmp_path, content):
    other = tmp_path / "other.json"
    other.write_text(content, encoding="utf-8")
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    with pytest.raises(SystemExit):
        evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--out", str(other)])

    assert other.read_text(encoding="utf-8") == content


def test_a_successful_run_leaves_the_result_and_no_partial_file(monkeypatch, tmp_path):
    _stub_world(monkeypatch, _test_split_frame(6), winner="The Eiffel Tower")
    out = tmp_path / "m.json"
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--out", str(out)])

    assert json.loads(out.read_text(encoding="utf-8"))["stage"] == "drawing"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.json"]


@pytest.mark.parametrize(
    ("porcelain", "dirty"),
    [(" M src/x.py\n", True), ("", False), (None, None)],
    ids=["dirty", "clean", "git-fails"],
)
def test_git_dirty_follows_git_status(monkeypatch, porcelain, dirty):
    calls: list[tuple[list[str], object]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs.get("cwd")))
        if command[1] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout="abc1234\n")
        if porcelain is None:
            raise OSError("git is not installed")
        return SimpleNamespace(returncode=0, stdout=porcelain)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert evaluate._git_state() == ("abc1234", dirty)
    assert {cwd for _, cwd in calls} == {Path(evaluate.__file__).resolve().parent}


def test_the_recorded_commit_is_the_packages_not_the_callers(monkeypatch, tmp_path):
    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    sha, dirty = evaluate._git_state()

    assert sha == expected
    assert isinstance(dirty, bool)


def test_an_out_folder_that_does_not_exist_is_refused_up_front(tmp_path):
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    with pytest.raises(SystemExit):
        evaluate.main(
            ["--model", str(model_yaml), "--version", "v1.1", "--out", str(tmp_path / "no" / "m.json")]
        )


def test_a_limit_below_one_is_refused(tmp_path):
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    with pytest.raises(SystemExit):
        evaluate.main(
            [
                "--model",
                str(model_yaml),
                "--version",
                "v1.1",
                "--limit",
                "0",
                "--out",
                str(tmp_path / "m.json"),
            ]
        )


def test_a_label_outside_the_vocabulary_is_refused_before_scoring(monkeypatch, tmp_path):
    frame = _rows("test", "not a quick draw category", 2, "t")
    _stub_world(monkeypatch, frame, winner="The Eiffel Tower")
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    with pytest.raises(ValueError, match="not a quick draw category"):
        evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--out", str(tmp_path / "m.json")])


def test_a_limited_cli_run_writes_no_gated_key(monkeypatch, tmp_path):
    _stub_world(monkeypatch, _test_split_frame(6), winner="The Eiffel Tower")
    out = tmp_path / "probe.json"
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--limit", "2", "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert "drawing_top5_acc" not in result
    assert result["overall"] == {"drawing_top5_acc": 1.0, "drawings": 2} and result["limit"] == 2


def test_a_validation_run_cannot_be_gated(monkeypatch, tmp_path):
    # `val` selects the model, so it is diagnostic: no top-level key means the gate cannot certify it.
    from primer.drawing.data import split_of

    frame = _test_split_frame(6)
    val_ids = [f"v{i}" for i in range(400) if split_of(f"v{i}") == "val"][:6]
    frame = frame.assign(source_id=val_ids, record_id=[f"r-{i}" for i in val_ids])
    _stub_world(monkeypatch, frame, winner="The Eiffel Tower")
    out = tmp_path / "val.json"
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    evaluate.main(["--model", str(model_yaml), "--version", "v1.1", "--split", "val", "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["split"] == "val" and result["overall"]["drawings"] == 6
    assert "drawing_top5_acc" not in result


def test_an_empty_version_is_refused_before_anything_is_read(monkeypatch, tmp_path):
    from primer.data import catalog

    def boom(*args, **kwargs):
        raise AssertionError("the bucket must not be read")

    monkeypatch.setattr(catalog, "read_dataset", boom)
    model_yaml = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"

    for version in ("", "  "):
        with pytest.raises(SystemExit):
            evaluate.main(
                ["--model", str(model_yaml), "--version", version, "--out", str(tmp_path / "m.json")]
            )


def test_a_template_without_label_is_refused_before_anything_is_read(monkeypatch, tmp_path):
    from primer.data import catalog
    from primer.drawing import model

    def boom(*args, **kwargs):
        raise AssertionError("the bucket and the weights must not be touched")

    monkeypatch.setattr(catalog, "read_dataset", boom)
    monkeypatch.setattr(model, "SiglipEncoder", boom)
    base = ROOT / "src" / "primer" / "drawing" / "configs" / "drawing_baseline.yaml"
    text = base.read_text(encoding="utf-8").replace("This is a photo of {label}.", "a drawing")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text, encoding="utf-8")
    # `labels_file` resolves beside the config, so the vocabulary has to sit next to the copy.
    (tmp_path / "quickdraw_categories.txt").write_text(
        (base.parent / "quickdraw_categories.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        evaluate.main(["--model", str(bad), "--version", "v1.1", "--out", str(tmp_path / "m.json")])


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
