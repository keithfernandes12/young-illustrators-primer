"""S6c3's MLflow logging in `tracking.py`, against a fake ``mlflow`` module.

No server and no ``mlflow`` package: the name builders are plain functions, and `log_run` is checked
by putting a recording fake in ``sys.modules``, the way `test_sensitivity.py` checks its logging.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from primer.drawing import tracking

#: The characters MLflow accepts in a metric or param name on every client OS.
LEGAL = re.compile(r"^[A-Za-z0-9_\-. /]+$")
CONVENTION_TAGS = {"stage", "owner", "data_version", "git_sha", "base_model", "gate_group", "where"}


def accuracy_file(**overrides):
    result = {
        "stage": "drawing",
        "drawing_top5_acc": 0.5,
        "gate_eligible": True,
        "overall": {"drawing_top5_acc": 0.5, "drawings": 4},
        "baseline": {"drawing_top5_acc": 0.0145},
        "slices": [
            {
                "key": "country_code",
                "value": "US",
                "n": 3,
                "judged": True,
                "metrics": {"drawing_top5_acc": 2 / 3},
            },
            {
                "key": "country_code",
                "value": "GB",
                "n": 1,
                "judged": False,
                "metrics": {"drawing_top5_acc": 0.0},
            },
        ],
        "data_version": "v1.1",
        "encoder": "google/siglip2-base-patch16-224",
        "encoder_revision": "abc",
        "split": "test",
        "device": "cuda",
        "dtype": "bfloat16",
        "prompt_count": 345,
        "prompts_sha256": "d34",
        "license": ["CC-BY-4.0"],
        "git_sha": "b749f34",
        "git_dirty": False,
        "seconds": 942.1,
        "scoring": {"prompt_template": "a child's drawing of a {label}.", "top_k": 5},
        "preprocessing": {"binarise": False, "pad": {"colour": 255}},
    }
    result.update(overrides)
    return result


def stability_file():
    return {
        "drawings": 1000,
        "variants": {
            "as_published": {"top1_confidence_median": 0.9, "distinct_top1": 160.0, "top1_label": "x"},
            "greyscale": {"top1_confidence_median": 0.8, "distinct_top1": 144.0, "top1_label": "y"},
        },
        "pairs": {"as_published->greyscale": {"top1_change_rate": 0.337, "top5_overlap": 0.745}},
        "data_version": "v1.1",
        "encoder": "google/siglip2-base-patch16-224",
        "git_sha": "b749f34",
        "git_dirty": True,
        "seconds": 999.8,
        "scoring": {"prompt_template": "t {label}"},
        "preprocessing": {"as_published": {"binarise": False}},
    }


# ------------------------------------------------------------------------------------------ names


def test_slices_are_logged_as_paths_with_their_size():
    metrics = tracking.accuracy_metrics(accuracy_file())

    assert metrics["slice/country_code/US/drawing_top5_acc"] == pytest.approx(2 / 3)
    assert metrics["slice/country_code/US/n"] == 3.0
    assert metrics["slice/country_code/GB/n"] == 1.0
    assert metrics["drawing_top5_acc"] == 0.5 and metrics["baseline/drawing_top5_acc"] == 0.0145
    assert metrics["drawings"] == 4.0 and metrics["seconds"] == 942.1
    assert all(LEGAL.match(name) for name in metrics), "every name must be one MLflow accepts"


def test_a_flag_is_not_logged_as_a_measurement():
    # `True` is an `int`: without the explicit check it would log as 1.0 beside the accuracy.
    metrics = tracking.accuracy_metrics(accuracy_file(overall={"drawing_top5_acc": 0.5, "complete": True}))

    assert "complete" not in metrics


def test_two_slice_values_that_clean_to_one_name_are_refused():
    # "a:b" and "a_b" both become "a_b"; MLflow would keep the second and drop the first silently.
    slices = [
        {"key": "k", "value": "a:b", "n": 1, "metrics": {"drawing_top5_acc": 1.0}},
        {"key": "k", "value": "a_b", "n": 1, "metrics": {"drawing_top5_acc": 0.0}},
    ]
    with pytest.raises(ValueError, match="both be logged"):
        tracking.accuracy_metrics(accuracy_file(slices=slices))


def test_a_value_with_nothing_legal_left_is_refused():
    with pytest.raises(ValueError):
        tracking.metric_part("")


def test_stability_pairs_lose_the_arrow_mlflow_cannot_take():
    metrics = tracking.stability_metrics(stability_file())

    assert metrics["stability/as_published_to_greyscale/top1_change_rate"] == 0.337
    assert metrics["stability/greyscale/top1_confidence_median"] == 0.8
    assert metrics["drawings"] == 1000.0
    assert not any("top1_label" in name for name in metrics), "a label name is not a number"
    assert all(LEGAL.match(name) for name in metrics)


def test_params_carry_the_prompt_and_nested_settings_as_json():
    params = tracking.run_params(accuracy_file())

    assert params["scoring.prompt_template"] == "a child's drawing of a {label}."
    assert params["preprocessing.pad"] == '{"colour": 255}'
    assert params["device"] == "cuda" and params["dtype"] == "bfloat16"
    assert params["license"] == "CC-BY-4.0"


# ------------------------------------------------------------------------------------------- tags


def test_tags_hold_the_convention_and_say_what_the_run_is(monkeypatch):
    monkeypatch.delenv("YIP_WHERE", raising=False)
    tags = tracking.run_tags(accuracy_file(), owner="member-1", kind="accuracy")

    assert CONVENTION_TAGS <= set(tags)
    assert tags["where"] == "local" and tags["base_model"] == "google/siglip2-base-patch16-224"
    assert tags["gate_eligible"] == "true" and tags["git_dirty"] == "false" and tags["kind"] == "accuracy"


def test_a_stability_run_is_never_gate_eligible_and_says_its_tree_was_dirty():
    tags = tracking.run_tags(stability_file(), owner="member-1", kind="stability")

    assert tags["gate_eligible"] == "false" and tags["git_dirty"] == "true"


def test_a_missing_commit_is_recorded_as_unknown_not_guessed():
    result = stability_file()
    del result["git_sha"], result["git_dirty"]

    tags = tracking.run_tags(result, owner="member-1", kind="stability")

    assert tags["git_sha"] == "unknown" and tags["git_dirty"] == "unknown"


def test_an_empty_owner_is_refused():
    with pytest.raises(ValueError, match="owner"):
        tracking.run_tags(accuracy_file(), owner="  ", kind="accuracy")


# ---------------------------------------------------------------------------------------- log_run


class FakeMlflow:
    def __init__(self):
        self.calls: list[tuple] = []

    def set_experiment(self, name):
        self.calls.append(("experiment", name))

    def start_run(self, run_name=None, tags=None):
        self.calls.append(("start", run_name, dict(tags)))
        calls = self.calls

        class Run:
            info = SimpleNamespace(run_id="run-1")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                calls.append(("end",))
                return False

        return Run()

    def log_params(self, params):
        self.calls.append(("params", dict(params)))

    def log_metrics(self, metrics):
        self.calls.append(("metrics", dict(metrics)))

    def log_artifact(self, path, artifact_path=None):
        self.calls.append(("artifact", Path(path).name, artifact_path))


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    return fake


def test_log_run_copies_the_file_into_one_run(fake_mlflow, tmp_path):
    out = tmp_path / "m.json"
    out.write_text("{}", encoding="utf-8")
    card = tmp_path / "CARD.md"
    card.write_text("card", encoding="utf-8")

    run_id = tracking.log_run(
        kind="accuracy", result=accuracy_file(), metrics_path=out, owner="m1", run_name="r", card_path=card
    )

    assert run_id == "run-1"
    kinds = [call[0] for call in fake_mlflow.calls]
    assert kinds == ["experiment", "start", "params", "metrics", "artifact", "artifact", "end"]
    assert fake_mlflow.calls[0] == ("experiment", "yip-drawing")
    assert ("artifact", "m.json", "eval") in fake_mlflow.calls and (
        "artifact",
        "CARD.md",
        None,
    ) in fake_mlflow.calls


def test_a_malformed_file_fails_before_any_run_is_opened(fake_mlflow, tmp_path):
    # Built first, so the server is not left with an empty run named after a file that never logged.
    with pytest.raises(KeyError):
        tracking.log_run(
            kind="accuracy",
            result={"data_version": "v1.1", "encoder": "e"},
            metrics_path=tmp_path / "m.json",
            owner="m1",
            run_name="r",
        )
    assert fake_mlflow.calls == []


def test_an_unknown_kind_is_refused_before_any_run(fake_mlflow, tmp_path):
    with pytest.raises(ValueError, match="kind"):
        tracking.log_run(
            kind="speed", result=accuracy_file(), metrics_path=tmp_path, owner="m1", run_name="r"
        )
    assert fake_mlflow.calls == []


# ------------------------------------------------------------------- stdout around the logging call


def _cp1252_stdout(monkeypatch):
    import io

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    return stream


def test_stdout_replaces_unprintable_text_only_while_logging(monkeypatch, tmp_path):
    from primer.drawing import evaluate

    stream = _cp1252_stdout(monkeypatch)
    seen = []

    def log_run(**kwargs):
        seen.append(stream.errors)
        print("\U0001f3c3 View run")  # what the 3.x client prints at end_run; cp1252 cannot encode it
        return "run-1"

    monkeypatch.setattr(tracking, "log_run", log_run)

    assert evaluate.log_to_mlflow("accuracy", {}, tmp_path / "m.json", "m1", "r") == "run-1"
    assert seen == ["replace"], "the client's emoji must not raise inside end_run"
    assert stream.errors == "strict", "the stream's own error mode is restored afterwards"


def test_stdout_is_restored_even_when_logging_fails(monkeypatch, tmp_path):
    from primer.drawing import evaluate

    stream = _cp1252_stdout(monkeypatch)

    def log_run(**kwargs):
        raise RuntimeError("server down")

    monkeypatch.setattr(tracking, "log_run", log_run)

    with pytest.raises(RuntimeError):
        evaluate.log_to_mlflow("accuracy", {}, tmp_path / "m.json", "m1", "r")
    assert stream.errors == "strict"
