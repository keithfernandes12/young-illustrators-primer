"""`data.py`: the recogniser's split over a frozen Quick, Draw! version (S6b2).

Hermetic: frames are built in memory, and `read_dataset` reads a version written to a temporary
local store. No bucket, no network, numpy and pandas only, so CI runs all of it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from primer.data.storage import Store
from primer.drawing import data, preprocess, render
from primer.drawing.data import RenderedSplit, SplitConfig, build_splits, read_dataset, split_of
from primer.drawing.preprocess import prepare_drawing
from primer.drawing.render import render_strokes

LABELS = ("cat", "house", "tree")
SQUARE = [[[10, 200, 200, 10, 10], [10, 10, 200, 200, 10]]]


def frame(count: int = 60, **overrides: object) -> pd.DataFrame:
    rows = {
        "record_id": [f"quickdraw:{i}" for i in range(count)],
        "source": ["quickdraw"] * count,
        "source_id": [str(5_000_000_000 + i) for i in range(count)],
        "label": [LABELS[i % len(LABELS)] for i in range(count)],
        "strokes": [json.dumps(SQUARE)] * count,
        "license": ["CC-BY-4.0"] * count,
        "attribution": ["Google Creative Lab, Quick, Draw! dataset"] * count,
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def splits(**kwargs: object) -> dict[str, RenderedSplit]:
    return build_splits(frame(), labels=LABELS, **kwargs)


# ---------------------------------------------------------------------------------------- splits


def test_a_drawings_split_is_a_function_of_its_id_alone():
    assert split_of("123") == split_of("123")
    assert [split_of(str(i)) for i in range(50)] == [split_of(str(i)) for i in range(50)]


def test_the_salt_is_part_of_the_split():
    other = replace(data.default_split_config(), salt="something-else")

    moved = sum(split_of(str(i)) != split_of(str(i), other) for i in range(2_000))

    assert moved > 500, "a new salt reshuffles; that is why changing it needs a new data version"


def test_the_shares_are_honoured():
    counts = pd.Series([split_of(str(i)) for i in range(40_000)]).value_counts(normalize=True)

    assert abs(counts["train"] - 0.8) < 0.01
    assert abs(counts["val"] - 0.1) < 0.01
    assert abs(counts["test"] - 0.1) < 0.01


def test_no_drawing_is_in_two_splits_and_none_is_lost():
    parts = data.split_frames(frame(500))
    ids = [set(part["source_id"]) for part in parts.values()]

    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert set().union(*ids) == set(frame(500)["source_id"])


def test_a_drawing_keeps_its_split_when_the_version_around_it_changes():
    """Adding drawings to a version must not move any that were already there."""
    small, large = data.split_frames(frame(100)), data.split_frames(frame(400))

    for split in data.SPLITS:
        assert set(small[split]["source_id"]) <= set(large[split]["source_id"])


@pytest.mark.parametrize(
    ("override", "message"),
    [({"train": 0.7}, "must be 1"), ({"val": 0.0}, "strictly between"), ({"salt": ""}, "non-empty")],
)
def test_a_split_config_that_would_mislead_is_refused(override, message):
    with pytest.raises(ValueError, match=message):
        replace(data.default_split_config(), **override)


def test_the_shipped_split_config_loads():
    assert SplitConfig.from_yaml() == data.default_split_config()


# ------------------------------------------------------------------------------------ the items


def test_an_evaluation_item_is_exactly_what_the_recogniser_sees_at_inference():
    item, label = splits()["test"][0]

    assert np.array_equal(item, prepare_drawing(render_strokes(SQUARE)))
    assert label in range(len(LABELS))


def test_every_item_is_a_prepared_drawing():
    for split in splits().values():
        if len(split):
            image, _ = split[0]
            assert image.shape == (224, 224, 3) and image.dtype == np.uint8


def test_augmentation_is_applied_to_train_and_nothing_else():
    built = splits()

    assert built["train"].augment and not built["val"].augment and not built["test"].augment
    plain = prepare_drawing(render_strokes(SQUARE))
    assert not np.array_equal(built["train"][0][0], plain), "train is augmented"


@pytest.mark.parametrize("split", ["val", "test"])
def test_asking_to_augment_an_evaluation_split_is_refused(split):
    with pytest.raises(ValueError, match="training only"):
        RenderedSplit(frame(), split, LABELS, augment=True)


def test_an_epoch_reproduces_and_the_next_one_differs():
    train = splits(seed=7)["train"]
    first = train[0][0]

    assert np.array_equal(train[0][0], first), "same epoch, same item, same pixels"
    train.set_epoch(1)
    assert not np.array_equal(train[0][0], first), "a new epoch sees a new augmentation"
    train.set_epoch(0)
    assert np.array_equal(train[0][0], first)


def test_labels_index_the_recognisers_vocabulary():
    built = build_splits(frame(), labels=LABELS)
    for split in built.values():
        for i in range(len(split)):
            record = split.record(i)
            assert LABELS[record.label_index] == record.label


def test_a_label_that_differs_from_the_vocabulary_only_in_case_is_the_same_label():
    """v1.1 stores `the eiffel tower`; the vocabulary says `The Eiffel Tower`. Same category."""
    vocabulary = ("The Eiffel Tower", "cat")
    lowered = build_splits(frame(30, label=["the eiffel tower"] * 30), labels=vocabulary)

    for split in lowered.values():
        if len(split):
            assert split.record(0).label_index == 0


def test_a_vocabulary_with_labels_differing_only_in_case_is_refused():
    with pytest.raises(ValueError, match="differ only in case"):
        data.label_index(("Cat", "cat"))


def test_every_item_carries_its_licence_and_attribution():
    record = splits()["train"].record(0)

    assert record.license == "CC-BY-4.0"
    assert record.attribution.startswith("Google Creative Lab")


def test_a_drawing_the_renderer_refuses_raises_naming_its_record():
    broken = frame(30, strokes=[json.dumps([[[1, 2, 3], [1, 2]]])] * 30)
    test = build_splits(broken, labels=LABELS)["test"]

    with pytest.raises(ValueError, match=r"record quickdraw:\d+ in test does not render"):
        test[0]


# ------------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (frame().drop(columns=["license"]), "lacks"),
        (frame(license=[None] * 60), "no license"),
        (frame(attribution=[None] * 60), "no attribution"),
        (frame(label=["dragon"] * 60), "not in the recogniser's vocabulary"),
        (frame(label=[None] * 60), "no label"),
        (frame(record_id=[None] * 60), "no record_id"),
    ],
)
def test_a_frame_the_loader_cannot_honour_is_refused_before_rendering(bad, message):
    with pytest.raises(ValueError, match=message):
        build_splits(bad, labels=LABELS)


UNUSABLE_TEXT = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
    pytest.param("\t\n", id="tab-newline"),
    pytest.param(7, id="number"),
    pytest.param(pd.NA, id="pd.NA"),
    pytest.param(float("nan"), id="nan"),
]


@pytest.mark.parametrize("column", ["license", "attribution"])
@pytest.mark.parametrize("value", UNUSABLE_TEXT)
def test_provenance_that_is_blank_or_not_text_is_refused(column, value):
    """Present is not enough: `""` and `7` are not a licence anyone can read."""
    bad = frame(**{column: [value] + ["usable"] * 59})

    with pytest.raises(ValueError, match=f"1 drawings have no {column}"):
        build_splits(bad, labels=LABELS)


@pytest.mark.parametrize("split", data.SPLITS)
@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (frame().drop(columns=["license"]), "lacks"),
        (frame(license=[None] * 60), "no license"),
        (frame(attribution=[None] * 60), "no attribution"),
        (frame(license=["  "] * 60), "no license"),
        (frame(attribution=[""] * 60), "no attribution"),
        (frame(license=[7] * 60), "no license"),
        (frame(label=["dragon"] * 60), "not in the recogniser's vocabulary"),
        (frame(label=[None] * 60), "no label"),
    ],
)
def test_the_public_constructor_refuses_a_frame_without_provenance_too(split, bad, message):
    """`RenderedSplit(...)` is public; building one directly is not a way round the checks."""
    with pytest.raises(ValueError, match=message):
        RenderedSplit(bad, split, LABELS, augment=False)


@pytest.mark.parametrize("column", ["license", "attribution"])
def test_one_blank_row_among_good_ones_is_still_counted(column):
    bad = frame(**{column: ["usable"] * 57 + ["", "  ", None]})

    with pytest.raises(ValueError, match=f"3 drawings have no {column}"):
        build_splits(bad, labels=LABELS)


# --------------------------------------------------------------------------------- read_dataset


@pytest.fixture
def store(tmp_path) -> Store:
    """A local store holding one frozen version, `vt`, of real-vocabulary drawings."""
    local = Store(tmp_path.as_posix())
    path = "processed/drawings/source=quickdraw/batch=20261010T000000Z.parquet"
    local.write_parquet(path, frame(90, label=["cat", "house", "tree"] * 30))
    local.write_json(
        "metadata/versions/vt.json",
        {"tag": "vt", "datasets": {"drawings": {"sources": {"quickdraw": {"files": [{"path": path}]}}}}},
    )
    return local


def test_read_dataset_reads_one_split_of_a_frozen_version(store):
    train = read_dataset("vt", "train", store=store)
    test = read_dataset("vt", "test", store=store)

    assert train.augment and not test.augment
    assert len(train) + len(test) + len(read_dataset("vt", "val", store=store)) == 90


def test_unfrozen_data_is_never_read(store):
    with pytest.raises(ValueError, match="frozen data version is required"):
        read_dataset("", store=store)


def test_an_unknown_version_says_so(store):
    with pytest.raises(KeyError, match="No dataset version"):
        read_dataset("v9", store=store)


def test_read_dataset_refuses_to_augment_evaluation(store):
    with pytest.raises(ValueError, match="training only"):
        read_dataset("vt", "val", store=store, augment=True)


# ---------------------------------------------------------------------------------------- guards


def test_the_loader_uses_the_one_renderer_and_the_one_preprocessing_path():
    """KAN-87 and KAN-31 in two lines: imported from their homes, not re-implemented."""
    assert data.render_strokes is render.render_strokes
    assert data.augment_canvas is preprocess.augment_canvas
    assert data.prepare_drawing is preprocess.prepare_drawing
