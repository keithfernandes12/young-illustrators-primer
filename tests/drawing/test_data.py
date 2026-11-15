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

LABELS = ("tree", "cat", "house")  # unsorted on purpose: the index follows the vocabulary order
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


GOLDEN = [
    "train", "train", "train", "val", "train", "train", "train", "train", "train", "train",
    "train", "train", "train", "train", "train", "val", "val", "test", "train", "val",
]  # fmt: skip


def test_the_shipped_salt_puts_known_ids_in_known_splits():
    """Frozen: if this moves, every drawing's split moved, and that needs a new data version."""
    assert [split_of(str(i)) for i in range(20)] == GOLDEN


def test_a_drawing_keeps_its_split_wherever_its_row_is():
    """Not by position: shuffling the frame must not move any drawing."""
    ordered = data.split_frames(frame(500))
    shuffled = data.split_frames(frame(500).sample(frac=1.0, random_state=0))

    for split in data.SPLITS:
        assert set(ordered[split]["source_id"]) == set(shuffled[split]["source_id"])


def test_rows_sharing_a_source_id_share_a_split():
    twins = data.split_frames(frame(40, source_id=["same"] * 20 + ["other"] * 20))

    for source_id in ("same", "other"):
        homes = {split: int((part["source_id"] == source_id).sum()) for split, part in twins.items()}
        assert sorted(homes.values()) == [0, 0, 20], f"{source_id}'s rows are split up: {homes}"


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


def test_each_item_has_its_own_augmentation_and_each_run_seed_its_own():
    """Same canvas everywhere, so any difference in pixels comes from the seed alone."""
    train = splits(seed=7)["train"]
    other = splits(seed=8)["train"]

    assert len({train.item_seed(i) for i in range(len(train))}) == len(train)
    assert train.item_seed(0) != other.item_seed(0)
    assert any(not np.array_equal(train[0][0], train[i][0]) for i in range(1, 6)), "items share a seed"
    assert not np.array_equal(train[0][0], other[0][0]), "the run seed does nothing"
    train.set_epoch(1)
    assert train.item_seed(0) != splits(seed=7)["train"].item_seed(0)


@pytest.mark.parametrize(
    ("seed", "error"),
    [(-1, ValueError), (1.5, TypeError), ("3", TypeError), (None, TypeError)],
)
def test_a_seed_that_numpy_would_refuse_later_is_refused_now(seed, error):
    with pytest.raises(error, match="seed"):
        RenderedSplit(frame(), "train", LABELS, augment=True, seed=seed)
    train = splits()["train"]
    with pytest.raises(error, match="epoch"):
        train.set_epoch(seed)


def test_numpy_integers_are_accepted_as_a_seed_and_an_epoch():
    train = RenderedSplit(frame(), "train", LABELS, augment=True, seed=np.int64(5))
    train.set_epoch(np.int32(2))

    plain = RenderedSplit(frame(), "train", LABELS, augment=True, seed=5)
    plain.set_epoch(2)

    assert train.epoch == 2 and train.seed == 5
    assert train.item_seed(0) == plain.item_seed(0)


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


def test_indices_past_either_end_raise_instead_of_wrapping_to_another_drawing():
    """With three items, `split[-4]` used to come back as `split[-1]`: the wrong drawing, silently."""
    small = RenderedSplit(frame(3), "test", LABELS, augment=False)
    size = len(small)
    assert size == 3

    for bad in (-size - 1, -size - 2, size, size + 5):
        with pytest.raises(IndexError):
            small[bad]
        with pytest.raises(IndexError):
            small.record(bad)


def test_negative_indices_count_from_the_end_down_to_minus_len():
    small = RenderedSplit(frame(3), "test", LABELS, augment=False)

    assert [small.record(i) for i in (-3, -2, -1)] == [small.record(i) for i in (0, 1, 2)]
    assert np.array_equal(small[-1][0], small[2][0]) and small[-1][1] == small[2][1]
    assert np.array_equal(small[-3][0], small[0][0])


def test_a_negative_index_gets_the_augmentation_of_the_item_it_names():
    """The seed comes from the normalised position, so `train[-1]` is `train[len - 1]` exactly."""
    train = RenderedSplit(frame(5), "train", LABELS, augment=True, seed=3)

    assert np.array_equal(train[-1][0], train[len(train) - 1][0])
    assert np.array_equal(train[-5][0], train[0][0])


def test_item_seed_answers_for_positions_like_indexing_does():
    train = RenderedSplit(frame(5), "train", LABELS, augment=True, seed=3)

    assert train.item_seed(-1) == train.item_seed(4)
    assert train.item_seed(np.int64(2)) == train.item_seed(2)
    for bad in (5, -6):
        with pytest.raises(IndexError):
            train.item_seed(bad)


def test_numpy_integers_index_and_other_types_do_not():
    small = RenderedSplit(frame(3), "test", LABELS, augment=False)

    assert np.array_equal(small[np.int64(1)][0], small[1][0])
    assert small.record(np.int32(-1)) == small.record(2)
    for bad in (1.0, "1", None):
        with pytest.raises(TypeError):
            small[bad]
        with pytest.raises(TypeError):
            small.record(bad)


def test_slices_still_work_and_stay_in_bounds():
    small = RenderedSplit(frame(3), "test", LABELS, augment=False)

    assert len(small[:]) == 3 and len(small[-2:]) == 2 and small[5:] == []


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
        (frame(label=[1] * 60), "no label"),
        (frame(record_id=[None] * 60), "no record_id"),
        (frame(strokes=[None] * 60), "no strokes"),
        (frame(source_id=[None] * 60), "no source_id"),
        (frame(source_id=[0.0] * 60), "no source_id"),
        (frame(source_id=[""] * 60), "no source_id"),
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


def test_a_split_with_no_drawings_is_valid():
    """A small version hashes nothing into one of the three splits; that is an empty split, not an error."""
    for count in (1, 2):  # fewer drawings than splits: some split is empty whatever the salt
        built = build_splits(frame(count), labels=LABELS)
        assert sum(len(split) for split in built.values()) == count
        assert any(len(split) == 0 for split in built.values())

    empty = RenderedSplit(frame().iloc[0:0], "test", LABELS, augment=False)
    assert len(empty) == 0 and list(empty) == []
    with pytest.raises(IndexError):
        empty[0]

    bare = pd.DataFrame({column: [] for column in data.REQUIRED})  # float64 columns, no `.str`
    assert len(RenderedSplit(bare, "val", LABELS, augment=False)) == 0


@pytest.mark.parametrize("column", ["license", "attribution"])
def test_provenance_held_as_a_categorical_is_judged_by_its_values(column):
    usable = frame(**{column: pd.Categorical(["usable"] * 60)})
    blank = frame(**{column: pd.Categorical(["usable"] * 59 + [""])})

    assert len(build_splits(usable, labels=LABELS)["train"]) > 0
    with pytest.raises(ValueError, match=f"1 drawings have no {column}"):
        build_splits(blank, labels=LABELS)


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


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"split": "val", "augment": True}, ValueError, "training only"),
        ({"split": "dev"}, ValueError, "split must be one of"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1.5}, TypeError, "seed"),
    ],
)
def test_a_bad_request_is_refused_before_the_version_is_read(monkeypatch, store, kwargs, error, message):
    """The read is the slow part (15 s on v1.1); a request that cannot succeed must not pay for it."""

    def read(*args, **kw):
        raise AssertionError("the version was read before the request was checked")

    monkeypatch.setattr(data.catalog, "read_dataset", read)

    with pytest.raises(error, match=message):
        read_dataset("vt", store=store, **kwargs)


# ---------------------------------------------------------------------------------------- guards


def test_the_loader_uses_the_one_renderer_and_the_one_preprocessing_path():
    """KAN-87 and KAN-31 in two lines: imported from their homes, not re-implemented."""
    assert data.render_strokes is render.render_strokes
    assert data.augment_canvas is preprocess.augment_canvas
    assert data.prepare_drawing is preprocess.prepare_drawing


def test_an_item_really_goes_through_render_then_augment_then_prepare(monkeypatch):
    """The identity checks above show where the names point; this shows an item calls them, in order."""
    calls: list[str] = []

    def spy(name, original):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return wrapper

    for name in ("render_strokes", "augment_canvas", "prepare_drawing"):
        monkeypatch.setattr(data, name, spy(name, getattr(data, name)))
    built = splits()

    built["train"][0]
    assert calls == ["render_strokes", "augment_canvas", "prepare_drawing"]
    calls.clear()
    built["test"][0]
    assert calls == ["render_strokes", "prepare_drawing"], "evaluation is never augmented"


# ----------------------------------------------------- what an item and a record carry, field by field


def test_the_label_an_item_trains_on_is_the_label_its_record_names():
    """Training reads `split[i][1]`; the record is the audit trail. They must be the same label."""
    whole = RenderedSplit(frame(9), "test", LABELS, augment=False)

    for i in range(len(whole)):
        record = whole.record(i)
        assert whole[i][1] == record.label_index == LABELS.index(record.label)
    assert {whole[i][1] for i in range(len(whole))} == {0, 1, 2}, "use every label, or this proves little"


def test_every_field_of_a_record_is_that_rows_own():
    licences = [f"licence-{i}" for i in range(6)]
    credits = [f"credit-{i}" for i in range(6)]
    rows = frame(6, license=licences, attribution=credits)
    whole = RenderedSplit(rows, "test", LABELS, augment=False)

    for i in range(6):
        assert whole.record(i) == data.DrawingRecord(
            record_id=f"quickdraw:{i}",
            source_id=str(5_000_000_000 + i),
            label=LABELS[i % len(LABELS)],
            label_index=i % len(LABELS),
            license=f"licence-{i}",
            attribution=f"credit-{i}",
        )


def test_iterating_yields_every_item_in_order():
    whole = RenderedSplit(frame(5), "test", LABELS, augment=False)

    walked = list(whole)

    assert len(walked) == len(whole) == 5
    for i, (image, label) in enumerate(walked):
        assert np.array_equal(image, whole[i][0]) and label == whole[i][1]


# ------------------------------------------------------------------- an override is honoured, not dropped


def test_a_split_config_passed_in_is_the_one_used():
    other = replace(data.default_split_config(), salt="a-different-salt")
    rows = frame(300)

    assert data.split_frames(rows, other)["train"]["source_id"].tolist() != (
        data.split_frames(rows)["train"]["source_id"].tolist()
    )
    default_sizes = {k: len(v) for k, v in build_splits(rows, labels=LABELS).items()}
    other_sizes = {k: len(v) for k, v in build_splits(rows, labels=LABELS, config=other).items()}
    assert default_sizes != other_sizes


def test_a_render_config_passed_in_is_the_one_used():
    shipped = preprocess.default_prepare_config().render
    thick = replace(shipped, line_width=shipped.line_width + 4)

    plain = RenderedSplit(frame(3), "test", LABELS, augment=False)[0][0]
    drawn = RenderedSplit(frame(3), "test", LABELS, augment=False, render=thick)[0][0]

    assert not np.array_equal(plain, drawn)


def test_a_prepare_config_passed_in_is_the_one_used():
    base = preprocess.default_prepare_config()
    # A smaller output size: a clean black-on-white square survives greyscale or binarising
    # unchanged, so only a setting that moves the shape shows the config reached `prepare_drawing`.
    variant = replace(base, render=replace(base.render, size=112))

    plain = build_splits(frame(), labels=LABELS)["test"][0][0]
    changed = build_splits(frame(), labels=LABELS, prepare=variant)["test"][0][0]
    direct = RenderedSplit(frame(3), "test", LABELS, augment=False, prepare=variant)[0][0]

    assert plain.shape == (224, 224, 3)
    assert changed.shape == direct.shape == (112, 112, 3)


def test_an_augmentation_config_passed_in_is_the_one_used():
    base = preprocess.default_augment_config()
    quiet = replace(
        base,
        wobble_probability=0.0,
        erasure_probability=0.0,
        ink_colour_probability=0.0,
        fill_probability=0.0,
        lighting_probability=0.0,
    )

    default = build_splits(frame(), labels=LABELS, seed=1)["train"][0][0]
    configured = build_splits(frame(), labels=LABELS, seed=1, augmentation=quiet)["train"][0][0]
    direct = RenderedSplit(frame(), "train", LABELS, augment=True, seed=1, augmentation=quiet)[0][0]

    assert not np.array_equal(default, configured)
    assert np.array_equal(configured, direct)


def test_read_dataset_passes_its_seed_on(store):
    seeded = read_dataset("vt", "train", store=store, seed=7)

    assert seeded.seed == 7
    assert seeded.item_seed(0) == RenderedSplit(frame(), "train", LABELS, augment=True, seed=7).item_seed(0)
    assert seeded.item_seed(0) != read_dataset("vt", "train", store=store).item_seed(0)


# ------------------------------------------------------------------------- the data's own spelling


def test_a_data_label_in_capitals_matches_a_lower_case_vocabulary_entry():
    """The other direction from `the eiffel tower`: the match is case-insensitive both ways."""
    shouting = frame(30, label=["CAT", "House", "TREE"] * 10)

    for split in build_splits(shouting, labels=LABELS).values():
        for i in range(len(split)):
            assert split.record(i).label_index == LABELS.index(split.record(i).label.casefold())
    with pytest.raises(ValueError, match="not in the recogniser's vocabulary"):
        build_splits(frame(30, label=["Dragon"] * 30), labels=LABELS)


def test_only_the_quickdraw_source_of_a_version_is_read(tmp_path):
    local = Store(tmp_path.as_posix())
    quick = "processed/drawings/source=quickdraw/batch=20261010T000000Z.parquet"
    other = "processed/drawings/source=amateur/batch=20261010T000000Z.parquet"
    local.write_parquet(quick, frame(12))
    local.write_parquet(other, frame(12, label=["dragon"] * 12))
    local.write_json(
        "metadata/versions/vm.json",
        {
            "tag": "vm",
            "datasets": {
                "drawings": {
                    "sources": {
                        "quickdraw": {"files": [{"path": quick}]},
                        "amateur": {"files": [{"path": other}]},
                    }
                }
            },
        },
    )

    total = sum(len(read_dataset("vm", split, store=local)) for split in data.SPLITS)

    assert total == 12, "another source's drawings are not Quick, Draw! drawings"


def test_a_version_with_no_quickdraw_rows_says_so(tmp_path):
    local = Store(tmp_path.as_posix())
    path = "processed/drawings/source=quickdraw/batch=20261010T000000Z.parquet"
    local.write_parquet(path, frame().iloc[0:0])
    local.write_json(
        "metadata/versions/ve.json",
        {"tag": "ve", "datasets": {"drawings": {"sources": {"quickdraw": {"files": [{"path": path}]}}}}},
    )

    with pytest.raises(ValueError, match="holds no Quick, Draw! drawings"):
        read_dataset("ve", store=local)
