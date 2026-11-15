"""Strata and samplers over Amateur Drawings.

Each test here is one of the slice's acceptance clauses made executable, because the interesting
failures in this module are all silent: a sampler that ignores its seed still returns rows, a
degenerate stratum column still produces a key, and a uniform draw used where a proportional one
belongs still reports a number. None of those raise on their own.
"""

from __future__ import annotations

import pandas as pd
import pytest

from primer.data.strata import (
    BIN_COUNT,
    EVERYTHING,
    DegenerateStratum,
    build_strata,
    carries_provenance,
    occupancy,
    sample_proportional,
    sample_uniform,
    stratum_inputs,
)


def drawings(
    sizes: list[tuple[int, int]],
    *,
    framing: list[float] | None = None,
    areas: list[float] | None = None,
) -> pd.DataFrame:
    """A frame shaped like ``CHARACTER_DRAWINGS`` for the columns the strata read."""
    n = len(sizes)
    frames = framing or [0.1 + 0.8 * (i / max(n - 1, 1)) for i in range(n)]
    area = areas or [1_000.0 * (i + 1) for i in range(n)]
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "original_width": [w for w, _ in sizes],
            "original_height": [h for _, h in sizes],
            "width": [512] * n,
            "height": [512] * n,
            "bbox_width": [512 * f for f in frames],
            "bbox_height": [512.0] * n,
            "area": area,
            "license": ["CC-BY-NC-4.0"] * n,
            "attribution": ["Amateur Drawings Dataset, Meta AI (Smith et al., 2023)"] * n,
        }
    )


def spread(n: int = 400) -> pd.DataFrame:
    """A corpus with four separable size signatures, big enough that nothing pools away."""
    shapes = [(3024, 4032), (2550, 3300), (800, 600), (512, 512)]
    sizes = [shapes[i % len(shapes)] for i in range(n)]
    return drawings(sizes)


# --- clause 1: strata exist, with their realised bin edges recorded alongside ----------------


def test_build_strata_returns_labels_and_the_edges_that_produced_them():
    frame = spread()
    strata, edges = build_strata(frame, min_size=1)

    assert len(strata) == len(frame)
    assert strata.nunique() > 1, "a corpus with four distinct shapes should not be one stratum"
    assert set(edges) == {"long_edge", "aspect", "framing", "area"}
    for name, cuts in edges.items():
        assert cuts == sorted(cuts), f"{name} edges are not ascending"
        assert len(cuts) >= 2, f"{name} produced no usable bin"
        assert len(cuts) - 1 <= BIN_COUNT


def test_the_recorded_edges_are_what_make_a_later_draw_reproducible():
    """Same corpus, same edges. The edges are a property of the corpus, so a different data
    version would re-bin every row -- which is why they are returned rather than recomputed."""
    frame = spread()
    _, first = build_strata(frame, min_size=1)
    _, again = build_strata(frame, min_size=1)
    assert first == again


# --- clause 2: a one-valued column raises, naming the column ---------------------------------


def test_a_constant_column_raises_and_names_itself():
    """``n_keypoints`` is the live case: 17 for every Amateur Drawings row today."""
    frame = spread()
    frame["area"] = 17.0

    with pytest.raises(DegenerateStratum, match="area"):
        build_strata(frame, min_size=1)


def test_the_constant_column_failure_is_invisible_to_a_distribution_test():
    """Why clause 2 is behavioural rather than statistical.

    With one stratum the uniform and the proportional draw return the *same* rows, and each
    returns exactly the distribution it claims, so a test comparing the two passes. That is the
    whole reason the rule is "raise" and not "the distributions differ".
    """
    frame = spread()
    one = pd.Series(["single"] * len(frame), index=frame.index)

    uniform = sample_uniform(frame, one, 40)
    proportional = sample_proportional(frame, one, 40)

    assert uniform.equals(proportional)
    assert len(uniform) == 40


def test_missing_stratum_columns_raise_rather_than_returning_the_whole_corpus():
    frame = spread().drop(columns=["bbox_width"])
    with pytest.raises(KeyError, match="bbox_width"):
        stratum_inputs(frame)


# --- clause 3: both samplers take a seed, and the same seed returns the same rows -------------


@pytest.mark.parametrize("sampler", [sample_uniform, sample_proportional])
def test_the_same_seed_returns_the_same_rows_twice(sampler):
    frame = spread()
    strata, _ = build_strata(frame, min_size=1)

    first = sampler(frame, strata, 40, seed=7)
    again = sampler(frame, strata, 40, seed=7)

    assert first["record_id"].tolist() == again["record_id"].tolist()


@pytest.mark.parametrize("sampler", [sample_uniform, sample_proportional])
def test_a_different_seed_returns_a_different_draw(sampler):
    """Otherwise ``seed`` is decoration and the reproducibility above proves nothing."""
    frame = spread()
    strata, _ = build_strata(frame, min_size=1)

    assert (
        sampler(frame, strata, 40, seed=1)["record_id"].tolist()
        != sampler(frame, strata, 40, seed=2)["record_id"].tolist()
    )


# --- clause 4: proportional reproduces a skew, uniform flattens it ---------------------------


def skewed() -> tuple[pd.DataFrame, pd.Series]:
    """900 rows in one stratum, 100 in another: a 9:1 corpus."""
    frame = drawings([(3024, 4032)] * 900 + [(800, 600)] * 100)
    strata = pd.Series(["big"] * 900 + ["small"] * 100, index=frame.index)
    return frame, strata


def test_proportional_reproduces_the_skew():
    frame, strata = skewed()
    drawn = sample_proportional(frame, strata, 100, seed=0)
    share = (strata.loc[drawn.index] == "big").mean()
    assert share == pytest.approx(0.9, abs=0.02)


def test_uniform_flattens_the_skew():
    frame, strata = skewed()
    drawn = sample_uniform(frame, strata, 100, seed=0)
    share = (strata.loc[drawn.index] == "big").mean()
    assert share == pytest.approx(0.5, abs=0.02)


def test_the_two_samplers_disagree_on_a_skewed_corpus():
    """The point of having two. If this ever passes with them equal, one of them is wrong."""
    frame, strata = skewed()
    uniform = sample_uniform(frame, strata, 100, seed=0)
    proportional = sample_proportional(frame, strata, 100, seed=0)
    assert (strata.loc[uniform.index] == "big").mean() < (
        strata.loc[proportional.index] == "big"
    ).mean()


@pytest.mark.parametrize("sampler", [sample_uniform, sample_proportional])
def test_a_sampler_returns_what_was_asked_for(sampler):
    frame, strata = skewed()
    assert len(sampler(frame, strata, 250, seed=0)) == 250


def test_uniform_takes_a_small_stratum_whole_rather_than_shrinking_the_sample():
    """A stratum below its even share must not silently cost the caller rows."""
    frame = drawings([(3024, 4032)] * 500 + [(800, 600)] * 10)
    strata = pd.Series(["big"] * 500 + ["small"] * 10, index=frame.index)

    drawn = sample_uniform(frame, strata, 200, seed=0)

    assert len(drawn) == 200
    assert (strata.loc[drawn.index] == "small").sum() == 10


# --- clause 5: occupancy, the slice's only reported output ------------------------------------


def test_occupancy_counts_every_row_once_and_shares_sum_to_one():
    frame = spread()
    strata, _ = build_strata(frame, min_size=1)

    table = occupancy(strata)

    assert table["rows"].sum() == len(frame)
    assert table["share"].sum() == pytest.approx(1.0)
    assert table["rows"].is_monotonic_decreasing


# --- clause 6: provenance survives a draw -----------------------------------------------------


@pytest.mark.parametrize("sampler", [sample_uniform, sample_proportional])
def test_a_drawn_sample_still_carries_licence_and_attribution(sampler):
    frame = spread()
    strata, _ = build_strata(frame, min_size=1)
    assert carries_provenance(sampler(frame, strata, 40, seed=0))


def test_provenance_check_fails_on_a_blank_licence():
    """Otherwise the clause above passes on anything and checks nothing."""
    frame = spread()
    frame.loc[0, "license"] = "   "
    assert not carries_provenance(frame)


# --- pooling ----------------------------------------------------------------------------------


def test_small_strata_are_pooled_into_a_neighbour():
    frame = spread(400)
    strata, _ = build_strata(frame, min_size=1)
    pooled, _ = build_strata(frame, min_size=150)
    assert pooled.nunique() < strata.nunique()
    assert len(pooled) == len(frame)


def test_everything_pools_into_one_stratum_when_nothing_is_big_enough():
    """The terminal fallback, matching ``split_strata``'s 'all eras'."""
    frame = spread(40)
    pooled, _ = build_strata(frame, min_size=10_000)
    assert set(pooled) == {EVERYTHING}
