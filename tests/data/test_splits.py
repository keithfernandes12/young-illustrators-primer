"""The shared book split (KAN-120): pure, configurable, and identical to the areas' existing copies."""

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from primer.data import splits
from primer.data.splits import SplitFractions, book_key, hash_fraction, split_of

KEYS = [book_key(source, str(i)) for source in ("storyweaver", "african_storybook", "tinystories")
        for i in range(4000)]


def test_the_book_key_is_source_slash_book_id():
    assert book_key("storyweaver", "storyweaver-12") == "storyweaver/storyweaver-12"


def test_hash_fraction_is_the_first_eight_bytes_of_sha256_over_two_to_the_64():
    key = "storyweaver/storyweaver-12"
    expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    assert hash_fraction(key) == expected
    assert 0 <= hash_fraction(key) < 1
    assert hash_fraction(key, salt="pool:") != hash_fraction(key)   # a salt gives an independent draw


def test_the_default_split_is_80_10_10_and_covers_every_book():
    counts = Counter(split_of(key) for key in KEYS)
    assert set(counts) == {"train", "val", "test"}
    shares = {name: counts[name] / len(KEYS) for name in counts}
    assert shares["train"] == pytest.approx(0.8, abs=0.02)
    assert shares["val"] == pytest.approx(0.1, abs=0.02)
    assert shares["test"] == pytest.approx(0.1, abs=0.02)


def test_a_books_split_depends_on_its_key_alone():
    # Pure in the key: the same book, asked again or among a different set of books, never moves.
    first = {key: split_of(key) for key in KEYS}
    assert {key: split_of(key) for key in reversed(KEYS)} == first
    assert all(split_of(key) == first[key] for key in KEYS[::7])


def test_fractions_are_laid_out_train_then_val_then_test():
    fractions = SplitFractions(train=0.6, val=0.3, test=0.1)
    assert fractions.upper_bounds() == (("train", 0.6), ("val", 0.9), ("test", 1.0))
    for key in KEYS[:2000]:
        position = hash_fraction(key)
        expected = "train" if position < 0.6 else "val" if position < 0.9 else "test"
        assert split_of(key, fractions) == expected


def test_a_book_exactly_on_a_boundary_goes_to_the_next_split(monkeypatch):
    # As both areas' copies do (`fraction < bound`); a hash landing exactly on 0.8 is rare, not impossible.
    monkeypatch.setattr(splits, "hash_fraction", lambda key, salt="": 0.8)
    assert split_of("any/book") == "val"
    monkeypatch.setattr(splits, "hash_fraction", lambda key, salt="": 0.0)
    assert split_of("any/book") == "train"


def test_float_addition_cannot_move_a_boundary():
    # 0.8 + 0.1 is 0.9000000000000001 in floating point; the bound must still be 0.9.
    assert SplitFractions().upper_bounds() == (("train", 0.8), ("val", 0.9), ("test", 1.0))


def test_the_last_split_always_ends_at_exactly_one(monkeypatch):
    # The validator accepts shares that sum to 1 within 1e-9; the last bound must still be 1.0,
    # or a hash above 0.9999999999 would fall off the end and split_of would raise.
    nearly = SplitFractions(train=0.7, val=0.2, test=0.0999999999)
    assert nearly.upper_bounds()[-1] == ("test", 1.0)
    monkeypatch.setattr(splits, "hash_fraction", lambda key, salt="": 0.99999999995)
    assert split_of("any/book", nearly) == "test"


def test_a_zero_share_is_allowed_and_never_assigned():
    no_train = SplitFractions(train=0.0, val=0.5, test=0.5)
    assert "train" not in {split_of(key, no_train) for key in KEYS}


@pytest.mark.parametrize("fractions", [
    {"train": 0.8, "val": 0.1, "test": 0.2},     # sums to 1.1
    {"train": 0.8, "val": 0.1, "test": 0.05},    # sums to 0.95
    {"train": 1.2, "val": -0.1, "test": -0.1},   # outside [0, 1]
    {"train": True, "val": 0.0, "test": 0.0},    # a bool is not a share
    {"train": "0.8", "val": 0.1, "test": 0.1},   # nor is a string
])
def test_bad_fractions_are_refused(fractions):
    with pytest.raises(ValueError):
        SplitFractions(**fractions)


# ------------------------------------------------------------------ parity with the areas' copies
#
# The point of KAN-120: both areas can switch to this module without moving a single book.

def test_it_reproduces_the_story_areas_80_10_10_split_exactly():
    from primer.story.data import split_of_book

    assert [split_of(key) for key in KEYS] == [split_of_book(key) for key in KEYS]


def test_it_reproduces_the_illustration_areas_val_test_split_exactly():
    from primer.illustration.consistency.experiment import split_of as illustration_split_of

    for val_fraction in (0.3, 0.5):
        fractions = SplitFractions(train=0.0, val=val_fraction, test=1 - val_fraction)
        assert [split_of(key, fractions) for key in KEYS] == [illustration_split_of(key, val_fraction)
                                                             for key in KEYS]


def test_it_reproduces_the_story_areas_salted_hash():
    from primer.story.data import hash_fraction as story_hash_fraction

    assert all(hash_fraction(key, salt="pool:") == story_hash_fraction(key, salt="pool:") for key in KEYS[:500])


def test_the_module_imports_no_model_area():
    # Model areas import this, so it must not import them back.
    source = Path(splits.__file__).read_text(encoding="utf-8")
    assert "import primer." not in source and "from primer." not in source
