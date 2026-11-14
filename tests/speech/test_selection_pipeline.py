"""Data, slices, sensitivity, model selection and registration for the speech stage.

Synthetic arrays and text only: no weights are loaded, nothing is downloaded, no child is recorded.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from primer.speech import sensitivity
from primer.speech.data import resample, with_synthetic_misreads
from primer.speech.evaluate import ReadAloudClip, evaluate, flagged_positions, is_runaway, word_errors
from primer.speech.interface import SAMPLE_RATE, FakeSpeechRecognizer
from primer.speech.model import (
    RecognizerConfig,
    Wav2Vec2Recognizer,
    WhisperRecognizer,
    build,
    generation_kwargs,
)
from primer.speech.register import refusal
from primer.speech.slices import SMALL_SLICE, age_band, bias_report, disparities, slice_of
from primer.speech.train import main as train_main
from primer.speech.train import select_best, slice_metric_name

SENTENCES = [
    "the cat sat on the mat",
    "a big red dog ran home",
    "we like to read books at night",
    "the little bird can sing",
    "my friend has a green kite",
    "she found a shell by the sea",
]


def clean_clips(n: int = 60) -> list[ReadAloudClip]:
    return [
        ReadAloudClip(f"c{i}", f"s{i % 5}", text, text, frozenset())
        for i, text in enumerate(SENTENCES[i % len(SENTENCES)] for i in range(n))
    ]


# ---------------------------------------------------------------------------- synthetic misreads


def test_a_perfect_recogniser_finds_exactly_the_synthetic_labels():
    """The labels are only worth something if alignment agrees with them when the audio is heard right."""
    for seed in range(5):
        for clip in with_synthetic_misreads(clean_clips(), rate=1.0, seed=seed):
            assert flagged_positions(clip.page_text, clip.reference) == clip.misread_positions, clip


def test_misreads_change_the_page_not_what_was_said():
    original = clean_clips(12)
    for before, after in zip(original, with_synthetic_misreads(original, rate=1.0)):
        assert after.reference == before.reference
        assert len(after.misread_positions) == 1


def test_misreads_are_deterministic_and_respect_the_rate():
    clips = clean_clips(200)
    first, again = with_synthetic_misreads(clips, seed=3), with_synthetic_misreads(clips, seed=3)
    assert first == again
    share = sum(bool(c.misread_positions) for c in first) / len(first)
    assert 0.55 < share < 0.85, share
    assert all(not c.misread_positions for c in with_synthetic_misreads(clips, rate=0.0))


def test_some_synthetic_misreads_are_skips():
    clips = with_synthetic_misreads(clean_clips(200), rate=1.0, skip_share=0.5)
    longer = [c for c in clips if len(c.page_text.split()) > len(c.reference.split())]
    assert longer, "a skip adds a page word the reader never said"


def test_resample_changes_the_rate_and_keeps_the_duration():
    one_second_8k = np.zeros(8_000, dtype=np.float32)
    assert resample(one_second_8k, 8_000).shape == (SAMPLE_RATE,)
    assert resample(np.ones(10, dtype=np.float32), SAMPLE_RATE).shape == (10,)


# ----------------------------------------------------------------------------------- slices


@pytest.mark.parametrize(
    ("age", "band"), [(3, "3-5"), (5, "3-5"), (6, "6-8"), (11, "9-12"), (16, "13-17"), (30, "adult")]
)
def test_age_bands_follow_the_product_modes(age, band):
    assert age_band(age) == band


def test_slice_of_reads_the_clip():
    clip = ReadAloudClip("c", "s", "a", "a", frozenset(), slices={"age_band": "3-5"})
    assert slice_of(clip, "age_band") == "3-5"
    assert slice_of(clip, "accent") is None


def _slice(value: str, n: int, wer: float) -> dict:
    return {"key": "age_band", "value": value, "n": n, "metrics": {"asr_wer": wer}}


def test_disparities_ignore_slices_too_small_to_judge():
    slices = [_slice("3-5", SMALL_SLICE, 0.4), _slice("9-12", SMALL_SLICE, 0.1), _slice("adult", 3, 0.9)]
    assert disparities(slices) == {
        "age_band": {"metric": "asr_wer", "best": "9-12", "worst": "3-5", "gap": 0.3}
    }


def test_bias_report_says_when_no_gap_can_be_claimed():
    report = bias_report([_slice("3-5", 2, 0.4)])
    assert "too small" in report
    assert "no gap is claimed" in report


# ------------------------------------------------------------------------------ sensitivity


def test_noise_lands_at_the_requested_snr():
    signal = np.sin(np.linspace(0, 400 * np.pi, SAMPLE_RATE)).astype(np.float32)
    noisy = sensitivity.add_noise(signal, 10, seed=1)
    snr = 10 * np.log10(np.mean(signal**2) / np.mean((noisy - signal) ** 2))
    assert snr == pytest.approx(10, abs=0.3)


def test_noise_is_seeded_and_silence_stays_silent():
    signal = np.ones(100, dtype=np.float32)
    assert np.array_equal(sensitivity.add_noise(signal, 5, seed=2), sensitivity.add_noise(signal, 5, seed=2))
    assert not sensitivity.add_noise(np.zeros(100, dtype=np.float32), 5).any()


def test_speed_changes_the_duration():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert sensitivity.change_speed(audio, 0.9).size == round(SAMPLE_RATE / 0.9)
    assert sensitivity.change_speed(audio, 1.1).size == round(SAMPLE_RATE / 1.1)


def test_the_sensitivity_run_reports_each_perturbation_against_clean():
    clip = replace(clean_clips(1)[0], samples=np.zeros(SAMPLE_RATE, dtype=np.float32))
    rows = sensitivity.run(FakeSpeechRecognizer(clip.reference), [clip])
    assert [r["perturbation"] for r in rows] == [p.name for p in sensitivity.PERTURBATIONS]
    assert rows[0]["wer_change"] == 0.0
    assert "| noise_10db |" in sensitivity.table(rows)


# -------------------------------------------------------------------------------- selection


SELECTION = {"max_wer": 0.25, "max_rtf": 0.5, "rank_by": "asr_misread_f1"}


def test_selection_ranks_by_misread_f1_among_eligible_candidates():
    results = {
        "accurate-but-autocorrects": {"asr_wer": 0.05, "asr_rtf": 0.1, "asr_misread_f1": 0.4},
        "literal": {"asr_wer": 0.15, "asr_rtf": 0.2, "asr_misread_f1": 0.8},
        "too-slow": {"asr_wer": 0.05, "asr_rtf": 2.0, "asr_misread_f1": 0.9},
        "too-wrong": {"asr_wer": 0.4, "asr_rtf": 0.1, "asr_misread_f1": 0.95},
    }
    choice = select_best(results, SELECTION)
    assert choice["selected"] == "literal"
    assert set(choice["rejected"]) == {"too-slow", "too-wrong"}


def test_an_undefined_score_ranks_last_and_ties_go_to_lower_wer():
    results = {
        "unknown": {"asr_wer": 0.01, "asr_rtf": None, "asr_misread_f1": None},
        "a": {"asr_wer": 0.2, "asr_rtf": 0.1, "asr_misread_f1": 0.7},
        "b": {"asr_wer": 0.1, "asr_rtf": 0.1, "asr_misread_f1": 0.7},
    }
    assert select_best(results, SELECTION)["ranking"] == ["b", "a", "unknown"]


def test_nothing_is_selected_when_no_candidate_qualifies():
    assert select_best({"x": {"asr_wer": 0.9, "asr_misread_f1": 1.0}}, SELECTION)["selected"] is None


def test_slice_metric_names_are_valid_for_mlflow():
    assert slice_metric_name("age_band", "3-5", "asr_wer") == "slice/age_band/3-5/asr_wer"
    assert "=" not in slice_metric_name("accent", "en=GB?", "asr_wer")


def test_the_smoke_run_writes_every_artifact(tmp_path):
    assert train_main(["--smoke", "--out", str(tmp_path)]) == 0
    selection = json.loads((tmp_path / "selection.json").read_text())
    winner = tmp_path / selection["selected"]
    for name in ("metrics.json", "bias_report.md", "recognizer.json", "sensitivity.md"):
        assert (winner / name).exists(), name
    metrics = json.loads((winner / "metrics.json").read_text())
    assert metrics["stage"] == "speech" and metrics["baseline"]
    assert "(selected)" in (tmp_path / "comparison.md").read_text()


def test_a_batch_without_the_baseline_needs_an_earlier_baseline_run(tmp_path):
    with pytest.raises(SystemExit):
        train_main(["--smoke", "--out", str(tmp_path), "--only", "whisper-tiny-en"])


# ------------------------------------------------------------------------- model and registry


def test_build_picks_the_family_and_loads_nothing():
    whisper = build(RecognizerConfig("w", "whisper", "openai/whisper-tiny.en"))
    ctc = build(RecognizerConfig("c", "wav2vec2", "facebook/wav2vec2-base-960h"))
    assert isinstance(whisper, WhisperRecognizer) and whisper._loaded is None
    assert isinstance(ctc, Wav2Vec2Recognizer) and ctc._loaded is None


def test_recogniser_config_rejects_typos_and_unknown_families():
    with pytest.raises(ValueError, match="unknown"):
        RecognizerConfig.from_dict({"name": "x", "family": "whisper", "checkpoint": "c", "beam": 5})
    with pytest.raises(ValueError, match="family"):
        RecognizerConfig.from_dict({"name": "x", "family": "conformer", "checkpoint": "c"})


def test_every_configured_candidate_is_valid():
    from primer.speech.evaluate import CONFIG, load_config

    for candidate in load_config(CONFIG)["candidates"]:
        RecognizerConfig.from_dict(candidate)


GOOD_TAGS = {"stage": "speech", "selected": "true"}


def test_a_selected_run_on_target_with_a_pinned_revision_may_register():
    assert refusal(GOOD_TAGS, {"asr_wer": 0.2}, revision="a1b2c3d") is None


@pytest.mark.parametrize(
    ("tags", "metrics", "revision", "reason"),
    [
        ({"stage": "story", "selected": "true"}, {"asr_wer": 0.1}, "abc", "stage"),
        ({"stage": "speech"}, {"asr_wer": 0.1}, "abc", "not selected"),
        (GOOD_TAGS, {}, "abc", "no asr_wer"),
        (GOOD_TAGS, {"asr_wer": 0.3}, "abc", "above"),
        (GOOD_TAGS, {"asr_wer": 0.1}, "main", "pin"),
    ],
)
def test_registration_is_refused_with_a_reason(tags, metrics, revision, reason):
    assert reason in (refusal(tags, metrics, revision) or "")


def test_the_selection_beats_nothing_it_has_not_measured():
    """evaluate on clean clips with a fake gives zero misreads to find: recall is undefined, not perfect."""
    overall, _ = evaluate(FakeSpeechRecognizer("the cat sat on the mat"), clean_clips(1))
    assert overall["asr_misread_recall"] is None


# ----------------------------------------------------------------------------- runaway guards


def test_guards_are_off_by_default_so_the_published_model_is_measured_as_published():
    kwargs = generation_kwargs(RecognizerConfig("w", "whisper", "openai/whisper-small.en"), duration_s=4.0)
    assert "max_new_tokens" not in kwargs and "no_repeat_ngram_size" not in kwargs
    assert "language" not in kwargs, "English-only checkpoints refuse a language flag"


def test_the_length_cap_scales_with_the_clip_and_never_exceeds_whispers_limit():
    guarded = RecognizerConfig(
        "g", "whisper", "openai/whisper-small.en", max_tokens_per_second=6, no_repeat_ngram_size=3
    )
    four_seconds = generation_kwargs(guarded, duration_s=4.0)
    assert four_seconds["max_new_tokens"] == 4 * 6 + 8
    assert four_seconds["no_repeat_ngram_size"] == 3
    assert generation_kwargs(guarded, duration_s=600.0)["max_new_tokens"] == 440


def test_multilingual_checkpoints_are_told_to_transcribe_english():
    kwargs = generation_kwargs(RecognizerConfig("t", "whisper", "openai/whisper-large-v3-turbo"), 2.0)
    assert kwargs["language"] == "en" and kwargs["task"] == "transcribe"


def test_a_looping_transcript_counts_as_a_runaway_and_a_misheard_one_does_not():
    looping = ReadAloudClip("a", "s", "lilly likes biscuit", "lilly likes biscuit", frozenset())
    misheard = ReadAloudClip("b", "s", "the cat sat", "the cat sat", frozenset())
    overall, _ = evaluate(_Scripted(["ni " * 40, "a bat sat"]), [looping, misheard])
    assert overall["asr_runaway_rate"] == 0.5
    assert is_runaway(word_errors("the cat sat", "the cat sat down there now")) is False


class _Scripted:
    def __init__(self, heard):
        self._heard = iter(heard)

    def transcribe(self, samples, sample_rate=SAMPLE_RATE, prompt=None):
        return FakeSpeechRecognizer(next(self._heard)).transcribe(samples)


def test_sensitivity_can_run_on_a_named_candidate_when_none_is_selected(tmp_path):
    """With an impossible target nothing is selected, but the team's chosen model still gets analysed."""
    import yaml

    from primer.speech.evaluate import CONFIG

    config = yaml.safe_load(CONFIG.read_text())
    config["selection"]["max_wer"] = -1.0
    strict = tmp_path / "strict.yaml"
    strict.write_text(yaml.safe_dump(config))
    out = tmp_path / "out"
    args = [
        "--smoke",
        "--config",
        str(strict),
        "--out",
        str(out),
        "--only",
        "whisper-small-en",
        "--only",
        "distil-small-en-guarded",
    ]
    assert train_main([*args, "--sensitivity-on", "distil-small-en-guarded"]) == 1
    assert json.loads((out / "selection.json").read_text())["selected"] is None
    assert (out / "distil-small-en-guarded" / "sensitivity.md").exists()


def test_sensitivity_on_a_candidate_outside_the_batch_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        train_main(
            [
                "--smoke",
                "--out",
                str(tmp_path),
                "--only",
                "whisper-small-en",
                "--sensitivity-on",
                "wav2vec2-base-960h",
            ]
        )
