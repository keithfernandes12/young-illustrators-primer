"""The ASR contract: a Fake any member can import, and the page alignment that finds misread words.

Nothing here loads weights, touches GCS, reaches the network or holds audio.
"""

from __future__ import annotations

from primer.speech.alignment import align, normalize
from primer.speech.interface import FakeSpeechRecognizer, SpeechRecognizer


def read(page_text: str, heard: str):
    transcript = FakeSpeechRecognizer(heard).transcribe([0.0] * 16_000)
    return align(0, page_text, transcript)


def test_the_fake_satisfies_the_protocol():
    assert isinstance(FakeSpeechRecognizer(), SpeechRecognizer)


def test_the_fake_returns_timed_words_and_ignores_the_audio():
    transcript = FakeSpeechRecognizer("the cat sat", word_s=0.5).transcribe([])
    assert transcript.text == "the cat sat"
    assert [(w.start_s, w.end_s) for w in transcript.words] == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]


def test_silence_is_an_empty_transcript():
    assert FakeSpeechRecognizer().transcribe([]).words == ()


def test_normalize_drops_case_and_punctuation_but_keeps_contractions():
    assert normalize("The dragon's here, ISN'T it?") == ["the", "dragon's", "here", "isn't", "it"]


def test_a_perfect_read_has_no_misread_words():
    result = read("The cat sat.", "the cat sat")
    assert [w.status for w in result.words] == ["correct"] * 3
    assert result.misread_words == ()


def test_a_wrong_word_is_misread():
    result = read("The cat sat", "the dog sat")
    assert result.misread_words == ("cat",)
    assert [w.status for w in result.words] == ["correct", "misread", "correct"]


def test_a_near_miss_is_forgiven():
    """'dragon' heard as 'dragin' is an accent, not an error."""
    assert read("a purple dragon", "a purple dragin").misread_words == ()


def test_a_skipped_word_is_reported():
    result = read("The big cat sat", "the cat sat")
    assert result.misread_words == ("big",)
    assert result.words[1].status == "skipped"


def test_an_extra_word_is_inserted_but_not_held_against_the_child():
    result = read("The cat sat", "the big cat sat")
    assert result.misread_words == ()
    assert [w.status for w in result.words].count("inserted") == 1


def test_a_replace_of_unequal_length_pairs_then_skips():
    result = read("one two three", "won")
    assert [w.status for w in result.words] == ["misread", "skipped", "skipped"]


def test_nothing_heard_skips_every_word():
    assert read("The cat sat", "").misread_words == ("the", "cat", "sat")


def test_an_empty_page_has_nothing_to_misread():
    assert read("", "hello").misread_words == ()


def test_the_page_index_is_carried_through():
    transcript = FakeSpeechRecognizer("hi").transcribe([])
    assert align(3, "hi", transcript).page_index == 3
