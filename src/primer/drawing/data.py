"""The recogniser's train / val / test split over a frozen Quick, Draw! version (KAN-38, slice S6b2).

``read_dataset(version, split)`` returns a ``RenderedSplit``: an indexable sequence whose items are
``(image, label_index)``, rendered on demand. Each item is the one path a drawing takes to the
recogniser anywhere in this repository:

    render_strokes  ->  augment_canvas (train only)  ->  prepare_drawing

``render_strokes`` (S6b) and ``prepare_drawing`` / ``augment_canvas`` (S6b1) are imported, never
re-implemented: two renderers or two preprocessing paths that drift are the failure KAN-87 and
KAN-31 exist to prevent, and ``tests/drawing`` asserts this loader imports both from their homes.

**Frozen data only.** ``version`` is required and read through ``primer.data.catalog`` at the exact
object generations the version recorded, so a number measured on ``v1.1`` is reproducible.

**Splits** come from a salted hash of each drawing's ``source_id`` with the shares in
``configs/drawing_data.yaml`` -- see that file for why. Nothing is rendered or held per split
beyond the rows of the frame, so building all three is cheap.

**Augmentation** is applied to ``train`` only, and refused for ``val`` and ``test`` rather than
merely defaulted off: an augmented evaluation set would measure the augmentation. Each item's
seed is derived from the split's seed, the epoch and the item's index, so a run reproduces and two
epochs see different augmentations of the same drawing (``set_epoch``).

**A drawing the renderer refuses raises**, naming its record, rather than being skipped. The
pipeline's ``preprocess`` step already filters invalid strokes before a record is written, so a
refusal here means the two definitions of a valid stroke have diverged, and that is worth stopping
for. Measured over every record of ``v1.1`` before shipping -- see the plan's Verified facts.

**Licence and attribution** travel with every item: ``RenderedSplit.record(i)`` returns them, and a
frame with either column missing, null, blank or not text is refused before anything is rendered --
on every path that builds a ``RenderedSplit``, the constructor included, not only ``read_dataset``.
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from primer.data import catalog
from primer.data.storage import Store
from primer.drawing.model import DrawingConfig
from primer.drawing.preprocess import (
    AugmentConfig,
    PrepareConfig,
    augment_canvas,
    default_augment_config,
    default_prepare_config,
    prepare_drawing,
)
from primer.drawing.render import RenderConfig, render_strokes

CONFIG_PATH = Path(__file__).parent / "configs" / "drawing_data.yaml"
SPLITS = ("train", "val", "test")
#: Columns a frame must carry. `license` and `attribution` are AGENTS.md's rule for every record.
REQUIRED = ("record_id", "source_id", "label", "strokes", "license", "attribution")

#: The two columns AGENTS.md requires on every record; they must hold text, not merely be present.
PROVENANCE = ("license", "attribution")

Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class SplitConfig:
    """How drawings are assigned to splits. See ``configs/drawing_data.yaml``."""

    salt: str
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        if not isinstance(self.salt, str) or not self.salt:
            raise ValueError(f"salt must be a non-empty string, not {self.salt!r}")
        for name in SPLITS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
                raise ValueError(f"{name} must be a share strictly between 0 and 1, not {value!r}")
            object.__setattr__(self, name, float(value))
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"train + val + test must be 1, not {total}")

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> SplitConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"{path.name} should be a mapping of settings")
        wanted = [f.name for f in fields(cls)]
        unknown = sorted(set(raw) - set(wanted) - {"name"})
        missing = [key for key in wanted if key not in raw]
        if missing or unknown:
            raise ValueError(f"{path.name}: missing {missing}, unknown {unknown}")
        return cls(**{key: raw[key] for key in wanted})


@lru_cache(maxsize=1)
def default_split_config() -> SplitConfig:
    return SplitConfig.from_yaml()


def split_of(source_id: str, config: SplitConfig | None = None) -> str:
    """The split a drawing belongs to, from a salted SHA-256 of its ``source_id``.

    SHA-256 rather than Python's ``hash``, which is salted per process: the same drawing must land
    in the same split on every machine and in every run.
    """
    config = config or default_split_config()
    digest = hashlib.sha256(f"{config.salt}:{source_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < config.train:
        return "train"
    if position < config.train + config.val:
        return "val"
    return "test"


@dataclass(frozen=True)
class DrawingRecord:
    """What an item is, beside its pixels: identity, label, and the licence it carries."""

    record_id: str
    source_id: str
    label: str
    label_index: int
    license: str
    attribution: str


class RenderedSplit(Sequence[tuple[Image, int]]):
    """One split's drawings, rendered on demand. ``split[i]`` is ``(image, label_index)``."""

    def __init__(
        self,
        frame: pd.DataFrame,
        split: str,
        labels: Sequence[str],
        *,
        augment: bool,
        seed: int = 0,
        render: RenderConfig | None = None,
        prepare: PrepareConfig | None = None,
        augmentation: AugmentConfig | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        if augment and split != "train":
            raise ValueError(
                f"augmentation is for training only; {split} is scored as the renderer draws it, "
                "or the evaluation measures the augmentation"
            )
        # Here as well as in `build_splits` / `read_dataset`: this is a public constructor, and a
        # frame handed to it directly must not reach `record()` without provenance.
        check_frame(frame, labels)
        index = label_index(labels)
        self.split = split
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self._frame = frame.reset_index(drop=True)
        self._label_index = self._frame["label"].str.casefold().map(index).astype(int).to_numpy()
        self._render = render
        self._prepare = prepare
        self._augmentation = augmentation

    def __len__(self) -> int:
        return len(self._frame)

    def set_epoch(self, epoch: int) -> None:
        """Give later epochs different augmentations of the same drawings, reproducibly."""
        self.epoch = epoch

    def item_seed(self, index: int) -> int:
        """The augmentation seed for item ``index`` in the current epoch."""
        state = np.random.SeedSequence([self.seed, self.epoch, index]).generate_state(1)
        return int(state[0])

    def position(self, index: SupportsIndex) -> int:
        """``index`` as a position in ``0 <= position < len(self)``, or ``IndexError``.

        Negative indices count from the end, but only down to ``-len(self)``: normalising first and
        bounds-checking after is what stops ``split[-4]`` of three items from turning into
        ``split[-1]``. ``operator.index`` rather than an ``int`` check so numpy integers, which
        samplers hand over, work, while a float or a string is a ``TypeError``.
        """
        position = operator.index(index)
        size = len(self)
        if position < 0:
            position += size
        if not 0 <= position < size:
            raise IndexError(f"index {operator.index(index)} is out of range for {size} drawings")
        return position

    def record(self, index: SupportsIndex) -> DrawingRecord:
        index = self.position(index)
        row = self._frame.iloc[index]
        return DrawingRecord(
            record_id=str(row["record_id"]),
            source_id=str(row["source_id"]),
            label=str(row["label"]),
            label_index=int(self._label_index[index]),
            license=str(row["license"]),
            attribution=str(row["attribution"]),
        )

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = self.position(index)
        row = self._frame.iloc[index]
        try:
            canvas = render_strokes(json.loads(row["strokes"]), self._render)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"record {row['record_id']} in {self.split} does not render: {error}. The pipeline's "
                "preprocess step should have filtered it, so the two definitions of a valid stroke "
                "have diverged"
            ) from error
        image = augment_canvas(canvas, self.item_seed(index), self._augmentation) if self.augment else canvas
        return prepare_drawing(image, self._prepare), int(self._label_index[index])

    def __iter__(self) -> Iterator[tuple[Image, int]]:
        for index in range(len(self)):
            yield self[index]


def label_index(labels: Sequence[str]) -> dict[str, int]:
    """Each vocabulary label's position, keyed case-insensitively.

    Case-insensitive because the two sides disagree on case and nothing else: measured over all
    632,746 drawings of ``v1.1``, the data stores ``the eiffel tower``, ``the great wall of china``
    and ``the mona lisa`` where the vocabulary -- the bucket's own file names -- has ``The Eiffel
    Tower`` and so on. An exact match refused the whole version. Two vocabulary labels that differ
    only in case would make the key ambiguous, so that is refused instead of guessed.
    """
    index: dict[str, int] = {}
    for position, label in enumerate(labels):
        key = label.casefold()
        if key in index:
            raise ValueError(f"vocabulary labels {labels[index[key]]!r} and {label!r} differ only in case")
        index[key] = position
    return index


def is_usable_text(value: object) -> bool:
    """A provenance value that says something: a string with a non-space character in it.

    ``None``, ``NaN`` and ``pd.NA`` are not strings, so they fail here along with numbers, and
    ``""`` or ``"   "`` would reach ``DrawingRecord`` as a licence nobody can read.
    """
    return isinstance(value, str) and bool(value.strip())


def check_frame(frame: pd.DataFrame, labels: Sequence[str]) -> None:
    """Refuse a frame the loader cannot honour, before anything is rendered."""
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"drawings frame lacks {missing}")
    for column in ("record_id", "source_id", "label", "strokes"):
        nulls = int(frame[column].isna().sum())
        if nulls:
            raise ValueError(f"{nulls} drawings have no {column}; every record must carry one")
    for column in PROVENANCE:
        unusable = int((~frame[column].map(is_usable_text)).sum())
        if unusable:
            raise ValueError(
                f"{unusable} drawings have no {column} (missing, blank or not text); every record "
                "must carry one"
            )
    known = label_index(labels)
    outside = sorted({label for label in frame["label"] if label.casefold() not in known})
    if outside:
        raise ValueError(
            f"{len(outside)} labels are not in the recogniser's vocabulary, e.g. {outside[:5]}; the "
            "vocabulary is configs/quickdraw_categories.txt"
        )


def split_frames(frame: pd.DataFrame, config: SplitConfig | None = None) -> Mapping[str, pd.DataFrame]:
    """``frame`` divided into the three splits by ``split_of`` on ``source_id``."""
    config = config or default_split_config()
    assigned = frame["source_id"].astype(str).map(lambda source_id: split_of(source_id, config))
    return {split: frame[assigned == split] for split in SPLITS}


def build_splits(
    frame: pd.DataFrame,
    *,
    labels: Sequence[str] | None = None,
    config: SplitConfig | None = None,
    seed: int = 0,
    augment: bool = True,
    prepare: PrepareConfig | None = None,
    augmentation: AugmentConfig | None = None,
) -> dict[str, RenderedSplit]:
    """All three splits of an already-read frame. ``augment`` applies to ``train`` alone."""
    labels = labels if labels is not None else DrawingConfig.from_yaml().labels()
    check_frame(frame, labels)
    return {
        split: RenderedSplit(
            part,
            split,
            labels,
            augment=augment and split == "train",
            seed=seed,
            prepare=prepare or default_prepare_config(),
            augmentation=augmentation or default_augment_config(),
        )
        for split, part in split_frames(frame, config).items()
    }


def read_dataset(
    version: str,
    split: str = "train",
    *,
    store: Store | None = None,
    seed: int = 0,
    augment: bool | None = None,
) -> RenderedSplit:
    """One split of the frozen Quick, Draw! drawings in ``version``. Never reads unfrozen data.

    ``augment`` defaults to on for ``train`` and off elsewhere; asking for it on ``val`` or
    ``test`` raises.
    """
    if not version:
        raise ValueError("a frozen data version is required; unfrozen data is never read")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    frame = catalog.read_dataset("drawings", store=store, sources=["quickdraw"], version=version)
    if frame.empty:
        raise ValueError(f"version {version!r} holds no Quick, Draw! drawings")
    labels = DrawingConfig.from_yaml().labels()
    check_frame(frame, labels)
    wanted = split == "train" if augment is None else augment
    return RenderedSplit(split_frames(frame)[split], split, labels, augment=wanted, seed=seed)
