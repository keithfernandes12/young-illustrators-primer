"""The speech evaluation harness: word error rate, misread detection and the metrics-file shape.

Runs on scripted recognisers and synthetic text only. No weights, no audio, no children's data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from primer.speech.evaluate import (
    ReadAloudClip,
    WordErrors,
    evaluate,
    flagged_positions,
    main,
    metrics_file,
    read_baseline,
    word_error_rate,
    word_errors,
)
from primer.speech.interface import SAMPLE_RATE, FakeSpeechRecognizer, SpeechRecognizer, Transcript


class ScriptedRecognizer:
    """Hears the next scripted utterance on each call, whatever the audio."""

    def __init__(self, heard: Sequence[str]) -> None:
        self._heard = iter(heard)
        self.prompts: list[str | None] = []

    def transcribe(
        self, samples: Sequence[float], sample_rate: int = SAMPLE_RATE, prompt: str | None = None
    ) -> Transcript:
        self.prompts.append(prompt)
        return FakeSpeechRecognizer(next(self._heard)).transcribe(samples, sample_rate)


def clip(
    page: str, said: str, misread: frozenset[int] | set[int] = frozenset(), **slices: str
) -> ReadAloudClip:
    return ReadAloudClip(
        clip_id="clip-1",
        speaker_id="speaker-1",
        page_text=page,
        reference=said,
        misread_positions=frozenset(misread),
        slices=slices,
    )


# ------------------------------------------------------------------------- word error rate


def test_a_perfect_transcript_has_no_errors():
    assert word_errors("the cat sat", "the cat sat") == WordErrors(0, 0, 0, 3)


def test_case_and_punctuation_are_not_errors():
    assert word_error_rate("The cat sat.", "the CAT sat") == 0.0


def test_a_substitution_counts_once():
    assert word_errors("the cat sat", "the bat sat") == WordErrors(1, 0, 0, 3)


def test_a_dropped_word_is_a_deletion():
    assert word_errors("the cat sat down", "the cat down") == WordErrors(0, 1, 0, 4)


def test_an_extra_word_is_an_insertion_and_can_push_the_rate_past_one():
    assert word_errors("hi", "oh hi there friend") == WordErrors(0, 0, 3, 1)
    assert word_error_rate("hi", "oh hi there friend") == 3.0


def test_an_empty_hypothesis_deletes_every_reference_word():
    assert word_errors("the cat sat", "") == WordErrors(0, 3, 0, 3)
    assert word_error_rate("the cat sat", "") == 1.0


def test_two_empty_utterances_agree():
    assert word_error_rate("", "") == 0.0


def test_an_empty_reference_with_words_heard_is_undefined():
    with pytest.raises(ValueError, match="undefined"):
        word_error_rate("", "hello")


def test_wer_is_strict_where_alignment_is_lenient():
    """A near-miss is correct for the child but still an error for the recogniser."""
    assert word_error_rate("the dragon", "the dragin") == 0.5
    assert flagged_positions("the dragon", "the dragin") == frozenset()


def test_errors_sum_across_utterances():
    total = word_errors("a b", "a c") + word_errors("x y z", "x y")
    assert total == WordErrors(1, 1, 0, 5)
    assert total.rate == pytest.approx(0.4)


# ------------------------------------------------------------------------ misread detection


def test_flagged_positions_index_the_page_words_including_skips():
    assert flagged_positions("The cat sat on the mat.", "the bat sat the mat") == {1, 3}


def test_inserted_words_are_never_flagged():
    assert flagged_positions("the cat", "um the big cat") == frozenset()


def test_precision_and_recall_against_labelled_misreads():
    # The child said "bat" for "cat" (labelled), and the recogniser heard it; it also misheard "mat"
    # as "hat", which is flagged but the child said it correctly.
    clips = [clip("the cat sat on the mat", "the bat sat on the mat", misread={1})]
    overall, _ = evaluate(ScriptedRecognizer(["the bat sat on the hat"]), clips)
    assert overall["asr_misread_precision"] == 0.5
    assert overall["asr_misread_recall"] == 1.0
    assert overall["asr_wer"] == pytest.approx(round(1 / 6, 4))


def test_an_auto_corrected_misread_shows_up_as_lost_recall_not_as_wer():
    """Whisper hearing the page word instead of the child's misreading hides the error."""
    clips = [clip("the cat sat", "the bat sat", misread={1})]
    overall, _ = evaluate(ScriptedRecognizer(["the cat sat"]), clips)
    assert overall["asr_misread_recall"] == 0.0
    assert overall["asr_misread_precision"] is None, "nothing flagged: precision is undefined, not 0"


def test_recall_is_undefined_when_nothing_was_labelled():
    overall, _ = evaluate(ScriptedRecognizer(["the cat sat"]), [clip("the cat sat", "the cat sat")])
    assert overall["asr_misread_recall"] is None
    assert overall["asr_wer"] == 0.0


def test_no_clips_is_an_error_not_a_perfect_score():
    with pytest.raises(ValueError, match="no clips"):
        evaluate(FakeSpeechRecognizer(), [])


# ---------------------------------------------------------------------------- the file shape


def test_slices_report_each_value_with_its_count():
    clips = [
        clip("a b", "a b", speaker_age="3-4"),
        clip("c d", "c d", speaker_age="3-4"),
        clip("e f", "e f", speaker_age="7-8"),
    ]
    _, slices = evaluate(ScriptedRecognizer(["a b", "c x", "e f"]), clips)
    assert [(s["key"], s["value"], s["n"]) for s in slices] == [
        ("speaker_age", "3-4", 2),
        ("speaker_age", "7-8", 1),
    ]
    assert slices[0]["metrics"]["asr_wer"] == 0.25
    assert slices[1]["metrics"]["asr_wer"] == 0.0


def test_the_metrics_file_has_the_contract_shape_and_is_json():
    overall, slices = evaluate(ScriptedRecognizer(["the cat"]), [clip("the cat", "the cat", accent="x")])
    data = metrics_file(
        model_uri="models:/primer-asr/1",
        data_version="v0.1",
        git_sha="abc1234",
        overall=overall,
        baseline={"asr_wer": 0.3},
        slices=slices,
    )
    assert set(data) == {"stage", "model_uri", "data_version", "git_sha", "overall", "baseline", "slices"}
    assert data["stage"] == "speech"
    assert data["baseline"] == {"asr_wer": 0.3}
    assert json.loads(json.dumps(data)) == data


def test_without_a_baseline_this_run_is_the_baseline():
    data = metrics_file(
        model_uri="m", data_version="v", git_sha="s", overall={"asr_wer": 0.2}, baseline=None, slices=[]
    )
    assert data["baseline"] == data["overall"]


def test_read_baseline_takes_overall_and_refuses_another_stage(tmp_path):
    good = tmp_path / "speech.json"
    good.write_text(json.dumps({"stage": "speech", "overall": {"asr_wer": 0.25}}), encoding="utf-8")
    assert read_baseline(good) == {"asr_wer": 0.25}

    other = tmp_path / "drawing.json"
    other.write_text(json.dumps({"stage": "drawing", "overall": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="drawing"):
        read_baseline(other)


def test_the_scripted_recognizer_satisfies_the_protocol():
    assert isinstance(ScriptedRecognizer([]), SpeechRecognizer)


def test_the_page_is_passed_as_the_prompt_only_when_asked():
    clips = [clip("the cat sat", "the cat sat")]
    plain, prompted = ScriptedRecognizer(["the cat sat"]), ScriptedRecognizer(["the cat sat"])
    evaluate(plain, clips)
    evaluate(prompted, clips, prompt_with_page=True)
    assert plain.prompts == [None]
    assert prompted.prompts == ["the cat sat"]


def test_real_time_factor_is_transcribe_time_over_audio_time():
    ticks = iter([0.0, 0.5])  # one transcription that takes half a second, by a clock the test owns
    one_second = ReadAloudClip("c", "s", "hi", "hi", frozenset(), samples=[0.0] * SAMPLE_RATE)
    overall, _ = evaluate(ScriptedRecognizer(["hi"]), [one_second], clock=lambda: next(ticks))
    assert overall["asr_rtf"] == 0.5


def test_rtf_is_undefined_without_audio():
    overall, _ = evaluate(ScriptedRecognizer(["hi"]), [clip("hi", "hi")])
    assert overall["asr_rtf"] is None


def test_f1_combines_precision_and_recall():
    clips = [clip("the cat sat on the mat", "the bat sat on the mat", misread={1})]
    overall, _ = evaluate(ScriptedRecognizer(["the bat sat on the hat"]), clips)
    assert overall["asr_misread_f1"] == pytest.approx(round(2 * 0.5 * 1.0 / 1.5, 4))


def test_slices_under_the_floor_are_reported_but_not_judged():
    _, slices = evaluate(ScriptedRecognizer(["a b"]), [clip("a b", "a b", speaker_age="3-5")])
    assert slices[0]["judged"] is False


def test_the_cli_refuses_a_candidate_the_config_does_not_have(tmp_path):
    with pytest.raises(SystemExit):
        main(["--model", "no-such-model", "--out", str(tmp_path / "m.json")])
