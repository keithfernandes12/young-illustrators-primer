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
    _pool,
    bin_column,
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


def test_building_strata_twice_on_one_frame_gives_the_same_edges():
    """Determinism, which is all this can show. The edges are returned so a later draw
    *could* re-use them, but no function accepts them yet, so reproducing a draw across
    data versions is not achievable through this API today."""
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
    frame, strata = skewed()

    table = occupancy(strata)

    assert table["rows"].sum() == len(frame)
    assert table["share"].sum() == pytest.approx(1.0)
    assert table["rows"].tolist() == [900, 100], "largest first"
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


# --- pooling: the neighbour loop, which nothing reached before -------------------------------


def test_a_small_stratum_pools_into_the_neighbour_it_shares_most_labels_with():
    """The neighbour selection itself, which the `min_size` tests above never reach.

    Those build strata where *nothing* reaches ``min_size``, so ``_pool`` returns at its
    terminal fallback before the loop runs. Deleting the loop entirely left them all passing.
    Here two strata are big enough to be pooling targets and three small ones have to choose.
    """
    keys = (
        ["x=0 / y=0 / z=0"] * 200
        + ["x=1 / y=1 / z=1"] * 150
        + ["x=0 / y=0 / z=1"] * 5
        + ["x=1 / y=1 / z=0"] * 3
    )
    strata = pd.Series(keys)

    pooled = _pool(strata, min_size=150)

    assert set(pooled) == {"x=0 / y=0 / z=0", "x=1 / y=1 / z=1"}
    assert (pooled[strata == "x=0 / y=0 / z=1"] == "x=0 / y=0 / z=0").all()
    assert (pooled[strata == "x=1 / y=1 / z=0"] == "x=1 / y=1 / z=1").all()
    assert len(pooled) == len(strata)


def test_pooling_ties_on_shared_labels_go_to_the_larger_stratum():
    keys = ["x=0 / y=0"] * 200 + ["x=1 / y=1"] * 150 + ["x=0 / y=1"] * 4
    strata = pd.Series(keys)

    pooled = _pool(strata, min_size=150)

    assert (pooled[strata == "x=0 / y=1"] == "x=0 / y=0").all()


def test_a_stratum_exactly_at_min_size_is_a_valid_pooling_target():
    """``>=`` rather than ``>``. With ``>`` the 150-row stratum stops being a target and its
    own near neighbour is pooled into the wrong place."""
    keys = ["x=0 / y=0 / z=0"] * 200 + ["x=1 / y=1 / z=1"] * 150 + ["x=1 / y=1 / z=0"] * 3
    strata = pd.Series(keys)

    pooled = _pool(strata, min_size=150)

    assert (pooled[strata == "x=1 / y=1 / z=0"] == "x=1 / y=1 / z=1").all()


def test_every_pooled_stratum_reaches_min_size():
    keys = ["x=0 / y=0"] * 200 + ["x=1 / y=1"] * 150 + ["x=0 / y=1"] * 4
    pooled = _pool(pd.Series(keys), min_size=150)
    assert pooled.value_counts().min() >= 150


# --- the remainder branches, which no fixture reached ----------------------------------------


def test_proportional_returns_exactly_n_when_shares_do_not_divide():
    """Three equal strata and n=100: each share rounds to 33, leaving one row to redistribute."""
    frame = drawings([(3024, 4032)] * 300)
    strata = pd.Series(["a"] * 100 + ["b"] * 100 + ["c"] * 100, index=frame.index)

    assert len(sample_proportional(frame, strata, 100, seed=0)) == 100


def test_proportional_returns_exactly_n_when_rounding_overshoots():
    """Six strata and n=9: every share rounds up, so the correction has to subtract."""
    frame = drawings([(3024, 4032)] * 600)
    strata = pd.Series([c for c in "abcdef" for _ in range(100)], index=frame.index)

    assert len(sample_proportional(frame, strata, 9, seed=0)) == 9


def test_uniform_returns_exactly_n_across_many_strata():
    """40 rows over 16 strata: 2 each leaves 8 to spread one at a time."""
    frame = spread(400)
    strata, _ = build_strata(frame, min_size=1)

    assert len(sample_uniform(frame, strata, 40, seed=0)) == 40


# --- the concentrated-column path, which is the live case on the real corpus ------------------


def concentrated() -> pd.DataFrame:
    """70% of rows share one aspect ratio, like the real corpus at 3:4.

    The 3:4 rows are drawn at four different scales so that long_edge still varies --
    otherwise it would be the constant column and the degenerate guard fires first.
    """
    portrait = [(300, 400), (600, 800), (900, 1200), (1200, 1600)]
    sizes = [portrait[i % 4] for i in range(70)] + [(w, 500) for w in range(200, 230)]
    return drawings(sizes)


def test_a_concentrated_column_drops_duplicate_edges_rather_than_raising():
    """Without ``duplicates="drop"`` this raises, and the real corpus hits it: `aspect`'s three
    quartiles are all 0.75, so it supports two bins rather than four."""
    strata, edges = build_strata(concentrated(), min_size=1)

    assert len(edges["aspect"]) - 1 < BIN_COUNT
    assert len(edges["aspect"]) - 1 >= 2
    assert len(strata) == 70 + 30


# --- missing values, which become their own stratum level -------------------------------------


def test_a_missing_input_becomes_an_unknown_bin_rather_than_dropping_the_row():
    frame = concentrated()
    frame.loc[0, "original_height"] = 0

    strata, _ = build_strata(frame, min_size=1)

    assert len(strata) == len(frame), "a row with a missing input must not vanish"
    assert "unknown" in strata.iloc[0]


def test_bin_column_labels_a_missing_value_unknown_itself():
    """Pins bin_column's own contract rather than the pipeline's.

    stratify_key also maps NaN to "unknown", so a build_strata-level test passes even if
    bin_column stops labelling them -- the two are redundant by design and the redundancy
    hides the mutation.
    """
    values = pd.Series([1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0], name="framing")

    labels, _ = bin_column(values, bins=2)

    assert labels.iloc[4] == "unknown"
    assert labels.notna().all(), "no row may come back unlabelled"


# --- a non-unique index silently draws the wrong rows ------------------------------------------


def test_a_non_unique_index_raises_rather_than_corrupting_the_draw():
    """`.loc` with duplicate labels returns every match, so a draw would silently mix strata."""
    frame = pd.concat([spread(40), spread(40)])

    with pytest.raises(ValueError, match="non-unique index"):
        stratum_inputs(frame)


# --- what the four inputs actually compute -----------------------------------------------------


def test_the_four_inputs_are_what_the_slice_says_they_are():
    frame = drawings([(300, 400)], framing=[0.5], areas=[1234.0])
    frame.loc[0, "width"] = 500
    frame.loc[0, "height"] = 400
    frame.loc[0, "bbox_width"] = 250.0
    frame.loc[0, "bbox_height"] = 400.0

    row = stratum_inputs(frame).iloc[0]

    assert row["long_edge"] == 400
    assert row["aspect"] == pytest.approx(0.75)
    assert row["framing"] == pytest.approx(250.0 * 400.0 / (500 * 400))
    assert row["area"] == 1234.0


def test_the_missing_column_guard_names_every_missing_column_and_why():
    frame = spread(40).drop(columns=["bbox_width", "area"])
    with pytest.raises(KeyError, match="needed to build strata"):
        stratum_inputs(frame)
    with pytest.raises(KeyError, match="bbox_width, area"):
        stratum_inputs(frame)


# --- provenance ---------------------------------------------------------------------------------


def test_provenance_check_fails_when_a_column_is_absent_entirely():
    """Dropping the column is the likeliest way a derived set loses provenance."""
    assert not carries_provenance(spread(40).drop(columns=["attribution"]))
