"""Strata over Amateur Drawings, and two samplers that draw from them.

Why this exists: `configs/sources.yaml` takes the *first* ``max_per_category`` drawings of each
Quick, Draw! category, which is harmless there because the corpus is already partitioned by
category. Amateur Drawings is one undifferentiated pile of ~178,000 uploads, so "first N" inherits
Meta's upload order and whatever that correlates with. Both consumers of this corpus need a better
answer: a training draw that covers the corpus's modes, and an evaluation draw that matches its
real shape.

**Two samplers, deliberately.** :func:`sample_uniform` draws evenly from every stratum, so rare
modes are learned rather than averaged away -- that is what a *training* set wants.
:func:`sample_proportional` draws in proportion to stratum size, so the sample looks like the
corpus -- that is what an *evaluation* set wants, because its job is to predict production. Using
the uniform draw for evaluation over-represents rare modes and reports a number the product never
sees, and nothing downstream would notice. That is why they are two named functions rather than a
flag.

The strata themselves are a *size signature*, not a claim about how a drawing was captured. A
3024x4032 upload is plausibly a phone photograph and a 756x1008 one plausibly a digital canvas, but
nobody has checked, so :func:`occupancy` reports the distribution and this module asserts nothing
about it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import pandas as pd

from primer.data.bias import SMALL_SLICE, stratify_key

log = logging.getLogger(__name__)

#: Quantile bins per input. Four keeps the key readable (256 cells at four inputs) while leaving
#: enough resolution to separate a phone photograph from a thumbnail.
BIN_COUNT = 4

#: Strata smaller than this are pooled into a neighbour. Taken from ``bias.SMALL_SLICE``
#: rather than restated, so the two cannot drift: below it a stratum cannot be split or
#: sampled from without the draw being mostly luck.
MIN_STRATUM = SMALL_SLICE

#: The label a pooled-away stratum collapses to when nothing is big enough to pool into.
EVERYTHING = "all"

#: Columns :func:`stratum_inputs` reads. Named here so a caller can check a frame before building.
REQUIRED_COLUMNS = (
    "original_width",
    "original_height",
    "width",
    "height",
    "bbox_width",
    "bbox_height",
    "area",
)


class DegenerateStratum(ValueError):
    """A column cannot stratify anything, so building strata from it would be a silent no-op.

    Raised rather than warned because the failure is invisible downstream: with a single stratum
    the uniform and the proportional draw return the *same* rows, and each returns exactly the
    distribution it claims, so a test comparing the two passes. ``n_keypoints`` is the live
    example -- it is 17 for every Amateur Drawings row today.
    """


def stratum_inputs(drawings: pd.DataFrame) -> pd.DataFrame:
    """The four quantities the strata are built from, derived from ``drawings``.

    * ``long_edge`` -- the longer side **before** Meta's 512 px resize, which is the only column
      that still carries the original capture size.
    * ``aspect`` -- original width over height. Orientation is kept rather than folded away: a
      portrait phone photograph and a landscape scan are different populations.
    * ``framing`` -- the annotated box as a fraction of the image, i.e. how much of the frame the
      figure fills. Derived; there is no ``bbox_area`` column.
    * ``area`` -- the segmentation area, which stands in for figure complexity.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in drawings.columns]
    if missing:
        raise KeyError(f"drawings is missing {', '.join(missing)}; needed to build strata")
    if not drawings.index.is_unique:
        dupes = drawings.index[drawings.index.duplicated()].unique()[:3].tolist()
        raise ValueError(
            f"drawings has a non-unique index (e.g. {dupes}); label-based selection would"
            " draw the wrong rows. Concatenate batches with ignore_index=True."
        )

    long_edge = drawings[["original_width", "original_height"]].max(axis=1)
    aspect = drawings["original_width"] / drawings["original_height"].replace(0, pd.NA)
    frame = drawings["width"] * drawings["height"]
    framing = (drawings["bbox_width"] * drawings["bbox_height"]) / frame.replace(0, pd.NA)
    return pd.DataFrame(
        {
            "long_edge": long_edge,
            "aspect": aspect,
            "framing": framing,
            "area": drawings["area"],
        },
        index=drawings.index,
    )


def bin_column(values: pd.Series, bins: int = BIN_COUNT) -> tuple[pd.Series, list[float]]:
    """``values`` cut into quantile bins, with the realised edges returned alongside.

    The edges come back because they are what makes a later draw reproducible: quantiles are a
    property of the corpus, so re-deriving them from a different data version silently re-bins
    every row. Store them with the strata and pass them back rather than recomputing.

    A concentrated column can have fewer distinct quantile edges than ``bins`` -- ``pd.qcut`` would
    raise on the duplicates -- so duplicate edges are dropped and the column simply gets fewer bins.
    A column with one distinct value raises instead: it cannot stratify anything.

    Rows whose value is missing get the label ``"unknown"`` rather than being dropped, so they
    stay in the frame and form their own stratum level. ``stratum_inputs`` manufactures NaN on
    purpose -- a zero ``original_height`` or a zero-area frame would otherwise divide by zero --
    so this is the path those rows take. The real corpus has none today.
    """
    clean = values.dropna()
    if clean.nunique() <= 1:
        raise DegenerateStratum(
            f"{values.name!r} has {clean.nunique()} distinct value(s); it cannot stratify anything"
        )
    binned, edges = pd.qcut(clean, bins, duplicates="drop", retbins=True, labels=False)
    if len(edges) - 1 < bins:
        log.info("%r supported %d bins rather than %d after dropping duplicate edges",
                 values.name, len(edges) - 1, bins)
    labels = pd.Series("unknown", index=values.index, dtype=object)
    labels.loc[clean.index] = [f"{values.name}={int(b)}" for b in binned]
    return labels, [float(e) for e in edges]


def _pool(keys: pd.Series, min_size: int) -> pd.Series:
    """Strata under ``min_size`` folded into the nearest stratum that is already big enough.

    "Nearest" is how many bin labels two keys share, which is the natural distance over a key built
    by concatenation. Ties go to the larger target, then by a stable ordering of the key
    itself, so the result never depends on dict ordering. (That last tier compares
    character by character, so a longer key beats a prefix of itself -- irrelevant here,
    where every key has the same shape, but it is not plain lexicographic order.) If no stratum reaches ``min_size`` the whole frame
    becomes one stratum, the same terminal fallback ``split_strata`` uses.
    """
    sizes = keys.value_counts()
    big = sizes[sizes >= min_size]
    if big.empty:
        log.warning("No stratum reached %d rows; pooling everything into %r", min_size, EVERYTHING)
        return pd.Series(EVERYTHING, index=keys.index, dtype=object)

    pooled = keys.copy()
    for key, count in sizes.items():
        if count >= min_size:
            continue
        parts = set(str(key).split(" / "))
        target = max(
            big.index,
            key=lambda b: (len(parts & set(str(b).split(" / "))), big[b], [-ord(c) for c in str(b)]),
        )
        pooled[keys == key] = target
    return pooled


def build_strata(
    drawings: pd.DataFrame,
    bins: int = BIN_COUNT,
    min_size: int = MIN_STRATUM,
) -> tuple[pd.Series, dict[str, list[float]]]:
    """One stratum label per row, plus the realised bin edges per input.

    Returns the labels and the edges together because neither is much use alone: the labels say
    which rows go together and the edges say what "together" meant, which is what a later draw
    needs to reproduce the same bins.
    """
    inputs = stratum_inputs(drawings)
    labelled: dict[str, pd.Series] = {}
    edges: dict[str, list[float]] = {}
    for name in inputs.columns:
        labelled[name], edges[name] = bin_column(inputs[name], bins)
    keys = stratify_key(pd.DataFrame(labelled, index=drawings.index), list(labelled))
    return _pool(keys, min_size), edges


def occupancy(strata: pd.Series) -> pd.DataFrame:
    """How many rows fall in each stratum, largest first, with each stratum's share.

    This is the slice's only reported output, and it is evidence rather than a claim: it says how
    the corpus distributes over size signatures, not that those signatures correspond to how a
    drawing was captured.
    """
    counts = strata.value_counts()
    return pd.DataFrame({"rows": counts, "share": counts / len(strata)})


def _draw(
    drawings: pd.DataFrame,
    strata: pd.Series,
    wanted: Mapping[str, int],
    seed: int,
) -> pd.DataFrame:
    """``wanted[stratum]`` rows from each stratum, seeded, returned in index order."""
    groups = strata.groupby(strata).groups
    picked = []
    for key, n in wanted.items():
        rows = drawings.loc[groups[key]]
        picked.append(rows if n >= len(rows) else rows.sample(n=n, random_state=seed))
    if not picked:
        return drawings.iloc[:0]
    return pd.concat(picked).sort_index()


def sample_uniform(
    drawings: pd.DataFrame,
    strata: pd.Series,
    n: int,
    seed: int = 0,
) -> pd.DataFrame:
    """``n`` rows drawn as evenly across strata as their sizes allow. **For training sets.**

    Evening out the strata is the point: a mode holding 2% of the corpus contributes far more than
    2% of this sample, so a model trained on it learns that mode rather than averaging it away.
    That same property makes this the wrong sampler for an evaluation set -- see
    :func:`sample_proportional`.

    A stratum smaller than its share is taken whole and the shortfall is spread over the rest, so
    the result is ``n`` rows whenever the corpus has that many.
    """
    return _draw(drawings, strata, _even_split(n, strata.value_counts().to_dict()), seed)


def sample_proportional(
    drawings: pd.DataFrame,
    strata: pd.Series,
    n: int,
    seed: int = 0,
) -> pd.DataFrame:
    """``n`` rows drawn in proportion to stratum size. **For evaluation sets.**

    The sample looks like the corpus, which is what an evaluation set is for: its job is to predict
    what happens in production, and a draw that over-represents rare modes reports a number the
    product never sees. Use :func:`sample_uniform` for training instead.
    """
    sizes = strata.value_counts().to_dict()
    total = sum(sizes.values())
    wanted = {k: round(n * size / total) for k, size in sizes.items()}
    short = n - sum(wanted.values())
    for key in sorted(sizes, key=lambda k: -sizes[k])[: abs(short)]:
        wanted[key] += 1 if short > 0 else -1
    return _draw(drawings, strata, wanted, seed)


def _even_split(n: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """``n`` shared as evenly as possible over ``sizes``, capped by what each stratum holds.

    Strata that cannot meet their share are taken whole and their shortfall is redistributed over
    the strata that can, repeating until nothing more can be moved. Without this a single small
    stratum would silently shrink the whole sample.
    """
    wanted = {k: 0 for k in sizes}
    remaining, open_keys = n, [k for k, size in sizes.items() if size > 0]
    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        if share == 0:
            for key in sorted(open_keys, key=lambda k: -sizes[k])[:extra]:
                wanted[key] += 1
            break
        moved = 0
        for key in list(open_keys):
            take = min(share, sizes[key] - wanted[key])
            wanted[key] += take
            moved += take
            if wanted[key] == sizes[key]:
                open_keys.remove(key)
        if moved == 0:
            break
        remaining -= moved
    return wanted


def carries_provenance(frame: pd.DataFrame, columns: Iterable[str] = ("license", "attribution")) -> bool:
    """Whether every row of ``frame`` carries non-empty ``license`` and ``attribution``.

    AGENTS.md and the review checklist both require it of every processed record, and Amateur
    Drawings is CC-BY-NC-4.0, so a derived set that loses the fields loses the only record of that.
    """
    for column in columns:
        if column not in frame.columns:
            return False
        values = frame[column]
        if values.isna().any() or (values.astype(str).str.strip() == "").any():
            return False
    return True
