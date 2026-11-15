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
    MIN_STRATUM,
    DegenerateStratum,
    _draw,
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
    """Determinism on one frame. `build_strata(..., edges=...)` re-uses them across frames,
    covered separately -- and note that reproduces the bins, not the strata, because pooling
    is re-derived from whatever sizes the new frame has."""
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


def test_a_non_unique_index_raises_when_building_strata():
    """`.loc` with duplicate labels returns every match, so a draw would silently mix strata."""
    frame = pd.concat([spread(40), spread(40)])

    with pytest.raises(ValueError, match="index is not unique"):
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


# --- round 2: occupancy ordering, which the "largest first" assertion did not hold -------------


def test_occupancy_orders_by_size_not_by_label():
    """The `skewed()` fixture cannot show this: "big" is both the larger stratum and the
    alphabetically earlier label, so a label-sorted implementation produces the same table."""
    frame = drawings([(3024, 4032)] * 300)
    strata = pd.Series(["alpha"] * 100 + ["zebra"] * 200, index=frame.index)

    table = occupancy(strata)

    assert table.index.tolist() == ["zebra", "alpha"]
    assert table["rows"].tolist() == [200, 100]


# --- round 2: edges can be passed back, which is what returning them was for -------------------


def test_edges_from_one_frame_can_be_re_used_on_another():
    """Acceptance clause 1's actual purpose: a later draw bins rows the same way even though the
    corpus has changed underneath it. Re-deriving quantiles would silently re-bin every row."""
    first = spread(400)
    _, edges = build_strata(first, min_size=1)

    grown = pd.concat([first, spread(120)], ignore_index=True)
    reused, same_edges = build_strata(grown, min_size=1, edges=edges)
    rederived, new_edges = build_strata(grown, min_size=1)

    assert same_edges == edges, "supplied edges must come back unchanged"
    assert new_edges != edges or reused.equals(rederived), "fixture must actually shift the bins"
    assert reused.iloc[: len(first)].tolist() == build_strata(first, min_size=1)[0].tolist()


def test_values_outside_the_supplied_edges_raise_rather_than_being_clipped():
    frame = spread(400)
    _, edges = build_strata(frame, min_size=1)
    bigger = frame.copy()
    bigger.loc[0, "original_width"] = 99_999

    with pytest.raises(ValueError, match="outside the supplied edges"):
        build_strata(bigger, min_size=1, edges=edges)


# --- round 2: a categorical strata reports categories no row uses ------------------------------


@pytest.mark.parametrize("sampler", [sample_uniform, sample_proportional])
def test_an_unused_category_does_not_break_the_draw(sampler):
    """`value_counts` on a categorical lists every declared category; `groupby` yields only the
    observed ones. Any boolean filter of a categorical keeps all its categories."""
    frame = drawings([(3024, 4032)] * 10)
    strata = pd.Series(
        pd.Categorical(["a"] * 5 + ["b"] * 5, categories=["a", "b", "c"]), index=frame.index
    )

    assert len(sampler(frame, strata, 4, seed=0)) == 4


# --- round 2: the index guard has to cover the layer where the .loc happens ---------------------


@pytest.mark.parametrize("sampler", [sample_uniform, sample_proportional])
def test_a_non_unique_index_raises_in_the_samplers_too(sampler):
    """The samplers reach `_draw` without ever calling `stratum_inputs`, so a guard only there
    leaves the corrupting path open: duplicate labels returned each drawing twice."""
    frame = pd.concat([spread(5), spread(5)])
    strata = pd.Series(["x"] * 5 + ["y"] * 5, index=frame.index)

    with pytest.raises(ValueError, match="index is not unique"):
        sampler(frame, strata, 4, seed=0)


# --- round 2: remainders, with fixtures that can see which strata absorb them -------------------


def test_the_proportional_remainder_is_taken_from_the_largest_stratum():
    """Sizes 261/189/70/57 at n=8 round to 4+3+1+1 = 9, so one has to come back off.

    An earlier fixture here rounded to exactly n, so the whole correction was dead code in CI.
    This one also distinguishes *which* stratum pays: taking it from the smallest instead still
    returns 8 rows, so a length assertion alone would not notice.
    """
    sizes = {"a": 261, "b": 189, "c": 70, "d": 57}
    frame = drawings([(3024, 4032)] * sum(sizes.values()))
    strata = pd.Series(
        [k for k, v in sizes.items() for _ in range(v)], index=frame.index
    )

    drawn = sample_proportional(frame, strata, 8, seed=0)
    counts = strata.loc[drawn.index].value_counts()

    assert len(drawn) == 8
    assert counts["a"] == 3, "the largest stratum gives the row back"
    assert counts["d"] == 1, "the smallest keeps its share"


def test_uniform_spreads_a_shortfall_over_several_passes():
    """Two tiny strata and one large one: the first pass caps both tiny ones and the remainder
    has to go round again. The mutant that skips the second pass returns 10 for a request of 11."""
    frame = drawings([(3024, 4032)] * 46)
    strata = pd.Series(["a"] * 3 + ["b"] * 3 + ["c"] * 40, index=frame.index)

    for n in (11, 12, 13):
        assert len(sample_uniform(frame, strata, n, seed=0)) == n


def test_asking_for_more_than_the_corpus_returns_the_corpus():
    frame = spread(40)
    strata, _ = build_strata(frame, min_size=1)
    assert len(sample_uniform(frame, strata, 10_000, seed=0)) == len(frame)


# --- round 2: the remaining unheld contracts ----------------------------------------------------


def test_strata_follow_the_frame_index_rather_than_position():
    """Index alignment is the module's whole contract — `_draw` selects by label."""
    frame = spread(400)
    frame.index = range(1000, 1000 + len(frame))

    strata, _ = build_strata(frame, min_size=1)
    drawn = sample_uniform(frame, strata, 40, seed=0)

    assert strata.index.tolist() == frame.index.tolist()
    assert len(drawn) == 40
    assert drawn["record_id"].nunique() == 40


def test_a_stratum_key_names_its_columns():
    """The key is `_pool`'s distance metric. Bare bin numbers would make `long_edge=0` and
    `area=0` the same token, silently conflating two columns."""
    strata, _ = build_strata(spread(400), min_size=1)

    key = strata.iloc[0]

    assert key.count(" / ") == 3
    for name in ("long_edge", "aspect", "framing", "area"):
        assert f"{name}=" in key


def test_provenance_check_fails_on_a_null_licence():
    """`astype(str)` turns None into "None", which is not blank — the null check is separate."""
    frame = spread(40)
    frame.loc[0, "license"] = None
    assert not carries_provenance(frame)


def test_pooling_prefers_the_larger_stratum_when_labels_would_say_otherwise():
    """Separates the size tie-break from the label one: here the larger target is also the
    alphabetically later, so a label-ordered rule sends the small rows the other way."""
    keys = ["a=0 / c=0"] * 150 + ["z=1 / c=1"] * 200 + ["a=0 / c=1"] * 4
    strata = pd.Series(keys)

    pooled = _pool(strata, min_size=150)

    assert (pooled[strata == "a=0 / c=1"] == "z=1 / c=1").all()


def test_the_default_min_stratum_is_pinned():
    """It equals `bias.SMALL_SLICE` today but is a separate decision; nothing should move it
    without this failing."""
    assert MIN_STRATUM == 100


# --- round 3: the contracts the third commit added but did not hold ---------------------------


def test_occupancy_counts_nan_strata_so_the_shares_still_sum_to_one():
    """`value_counts()` drops NaN while the denominator is `len(strata)`, which would make the
    table silently omit rows and the shares sum to less than 1."""
    strata = pd.Series(["a"] * 5 + ["b"] * 3 + [None] * 2)

    table = occupancy(strata)

    assert table["rows"].sum() == 10
    assert table["share"].sum() == pytest.approx(1.0)


def test_occupancy_leaves_out_categories_no_row_uses():
    """The stratum count is read from this table, so a declared-but-unused category would
    inflate it."""
    strata = pd.Series(pd.Categorical(["a"] * 5 + ["b"] * 5, categories=["a", "b", "c"]))

    assert len(occupancy(strata)) == 2


def test_pooling_is_stable_whatever_order_the_rows_arrive_in():
    """Two equally sized targets with equal label overlap: without the character tier the answer
    depends on which one `max` happens to see first."""
    small = ["x=0 / y=1"] * 4
    one = pd.Series(["x=0 / y=0"] * 150 + ["x=1 / y=1"] * 150 + small)
    other = pd.Series(["x=1 / y=1"] * 150 + ["x=0 / y=0"] * 150 + small)

    first = _pool(one, min_size=150)
    second = _pool(other, min_size=150)

    assert set(first[one == "x=0 / y=1"]) == set(second[other == "x=0 / y=1"]) == {"x=0 / y=0"}


def test_a_draw_comes_back_in_index_order():
    """Two draws that select the same rows must compare equal regardless of stratum enumeration."""
    frame = spread(400)
    strata, _ = build_strata(frame, min_size=1)

    drawn = sample_uniform(frame, strata, 40, seed=0)

    assert drawn.index.tolist() == sorted(drawn.index.tolist())


def test_the_uniform_remainder_goes_to_the_largest_open_strata():
    """Pins the direction, which the docstring now records: one row left over goes to the
    larger stratum, not the smaller."""
    frame = drawings([(3024, 4032)] * 105)
    strata = pd.Series(["big"] * 100 + ["small"] * 5, index=frame.index)

    drawn = sample_uniform(frame, strata, 1, seed=0)

    assert strata.loc[drawn.index].tolist() == ["big"]


def test_partial_edges_name_the_input_they_are_missing():
    """Both halves, because the name claims both: that it refuses, and which input it names.
    Matching only the generic tail let `', '.join(absent)` be replaced by a constant."""
    frame = spread(400)
    _, edges = build_strata(frame, min_size=1)
    del edges["aspect"]

    with pytest.raises(KeyError) as caught:
        build_strata(frame, min_size=1, edges=edges)

    assert "aspect" in str(caught.value)
    assert "supply every input's edges or none" in str(caught.value)


def test_an_edges_mapping_with_a_null_value_is_refused_like_a_missing_one():
    """A mapping round-tripped through JSON or YAML carries nulls. Keyed-presence-only would let
    that column re-derive its quantiles against the new corpus while the others use the old
    edges -- the mixed binning the guard's own message says it refuses."""
    frame = spread(400)
    _, edges = build_strata(frame, min_size=1)
    edges["aspect"] = None  # type: ignore[assignment]

    with pytest.raises(KeyError) as caught:
        build_strata(frame, min_size=1, edges=edges)

    assert "aspect" in str(caught.value)


def test_a_draw_refuses_a_stratum_no_row_is_in_rather_than_quietly_coming_up_short():
    """`_sizes` keeps an unused categorical level out of `wanted`, so this is unreachable through
    the samplers -- but `observed=True` is what makes the failure loud if it ever becomes
    reachable, and an earlier round was wrong to call a guard untestable without trying."""
    frame = spread(8)
    strata = pd.Series(
        pd.Categorical(["a"] * 4 + ["b"] * 4, categories=["a", "b", "unused"]),
        index=frame.index,
    )

    with pytest.raises(KeyError):
        _draw(frame, strata, {"a": 2, "unused": 2}, seed=0)


def test_supplied_edges_reproduce_the_bins_but_not_the_strata():
    """The limitation the docstring claims, made falsifiable. The only other `edges` test runs at
    `min_size=1`, which switches pooling off -- precisely why the earlier round could not see
    this. Same edges, a 40% subset: the unpooled cells come back identical, and 300 of 800 rows
    still land in a different stratum once pooling runs against the smaller frame."""
    frame = spread(2000)
    _, edges = build_strata(frame, min_size=100)
    subset = frame.iloc[:800]

    cells_full, _ = build_strata(frame, min_size=1, edges=edges)
    cells_subset, _ = build_strata(subset, min_size=1, edges=edges)
    pooled_full, _ = build_strata(frame, min_size=100, edges=edges)
    pooled_subset, _ = build_strata(subset, min_size=100, edges=edges)

    assert cells_full.loc[subset.index].equals(cells_subset), "the edges should pin the bins"
    moved = (pooled_full.loc[subset.index] != pooled_subset).sum()
    assert moved > 0, (
        "if this passes, pooling has become reproducible across frames and bin_column's "
        "docstring should stop saying it is not"
    )


def test_min_stratum_still_matches_the_reporting_floor_it_was_taken_from():
    """Not a coupling -- `MIN_STRATUM` is deliberately its own constant. This fails if
    `bias.SMALL_SLICE` moves, so the choice to follow it or not is made on purpose."""
    from primer.data.bias import SMALL_SLICE

    assert MIN_STRATUM == SMALL_SLICE, (
        "bias.SMALL_SLICE has changed; decide whether pooling should follow it"
    )
