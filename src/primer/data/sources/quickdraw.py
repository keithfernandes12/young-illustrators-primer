"""Quick, Draw! (Google Creative Lab): doodles of 345 everyday categories, for the drawing recognizer.

Reads the simplified-stroke files from Google's public bucket, which needs no
credentials. Each batch covers the next ``categories_per_batch`` categories and
keeps the first ``max_per_category`` drawings of each, so a 69 MB category file
is never copied whole. The position is saved in ``metadata/state/quickdraw.json``.

Strokes are kept as-is here; rendering them to images and augmenting them to
look like children's drawings is a later preprocessing step.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from primer.data import checks
from primer.data.manifest import Check, StageRun
from primer.data.schemas import DRAWINGS, conform
from primer.data.sources.base import Source
from primer.data.text import record_id

log = logging.getLogger(__name__)

ListCategories = Callable[[], list[str]]
ReadCategory = Callable[[str, int], list[dict]]

REQUIRED_FIELDS = ["key_id", "word", "drawing", "recognized", "countrycode"]


def valid_strokes(drawing: object) -> bool:
    """A drawing is a non-empty list of strokes, each ``[[x...], [y...]]`` of equal length."""
    if not isinstance(drawing, list) or not drawing:
        return False
    for stroke in drawing:
        if not (isinstance(stroke, list) and len(stroke) >= 2):
            return False
        xs, ys = stroke[0], stroke[1]
        if not (isinstance(xs, list) and isinstance(ys, list) and xs and len(xs) == len(ys)):
            return False
    return True


class QuickDraw(Source):
    key = "quickdraw"
    dataset = "drawings"

    def __init__(self, config, store, list_categories: ListCategories | None = None,
                 read_category: ReadCategory | None = None):
        super().__init__(config, store)
        self.list_categories = list_categories or self._list_from_bucket
        self.read_category = read_category or self._read_from_bucket
        self._bucket_fs = None

    @property
    def bucket_fs(self):
        if self._bucket_fs is None:
            import gcsfs

            self._bucket_fs = gcsfs.GCSFileSystem(token="anon")  # public bucket
        return self._bucket_fs

    @property
    def bucket_path(self) -> str:
        return self.options.get("bucket_path", "quickdraw_dataset/full/simplified")

    @property
    def max_per_category(self) -> int:
        return self.options.get("max_per_category", 2000)

    def _list_from_bucket(self) -> list[str]:
        files = self.bucket_fs.ls(self.bucket_path)
        return sorted(f.rsplit("/", 1)[-1].removesuffix(".ndjson") for f in files if f.endswith(".ndjson"))

    def _read_from_bucket(self, category: str, limit: int) -> list[dict]:
        with self.bucket_fs.open(f"{self.bucket_path}/{category}.ndjson", "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in itertools.islice(fh, limit)]

    # ------------------------------------------------------------------ ingest

    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        per_batch = batch_size or self.options.get("categories_per_batch", 50)
        state = self.store.read_json(self.state_path, default={"next_category": 0, "records": 0})
        categories = self.list_categories()
        start = state["next_category"]
        chosen = categories[start:start + per_batch]

        run.records_in = 0
        if not chosen:
            log.info("All %d Quick, Draw! categories ingested", len(categories))
            run.records_out = 0
            run.records_total = state["records"]
            return False

        prefix = self.raw_prefix(run.batch_id)
        counts: dict[str, int] = {}
        for i, category in enumerate(chosen, start=start):
            rows = self.read_category(category, self.max_per_category)
            self.store.write_jsonl_gz(f"{prefix}/part-{i:05d}.jsonl.gz", rows)
            counts[category] = len(rows)
            log.info("  %3d/%d %-25s %s drawings", i + 1, len(categories), category, f"{len(rows):,}")

        total = sum(counts.values())
        self.store.write_json(f"{prefix}/_batch.json", {
            "batch_id": run.batch_id,
            "bucket_path": self.bucket_path,
            "category_index_start": start,
            "category_index_end": start + len(chosen),
            "max_per_category": self.max_per_category,
            "count": total,
            "categories": counts,
            "fetched_at": datetime.now(UTC).isoformat(),
        })
        state = {"next_category": start + len(chosen), "records": state["records"] + total,
                 "categories_total": len(categories)}
        self.store.write_json(self.state_path, state)

        run.records_out = total
        run.records_total = state["records"]
        run.output_uri = self.store.url(prefix)
        return True

    # ---------------------------------------------------------------- validate

    def validate(self, run: StageRun) -> None:
        records = self.read_raw(run.batch_id)
        batch = self.store.read_json(f"{self.raw_prefix(run.batch_id)}/_batch.json", default={})
        minimum = min(self.options.get("min_per_category", 100), self.max_per_category)
        short = sorted(c for c, n in batch.get("categories", {}).items() if n < minimum)
        recognized = sum(1 for r in records if r.get("recognized") is True)
        share = recognized / len(records) if records else 0.0

        run.records_in = run.records_out = len(records)
        run.output_uri = self.store.url(self.raw_prefix(run.batch_id))
        run.checks += [
            checks.min_records(records, 1),
            checks.required_fields(records, REQUIRED_FIELDS),
            checks.unique(records, "key_id"),
            checks.max_ratio("malformed_strokes", records, lambda r: not valid_strokes(r.get("drawing")), 0.001),
            Check("recognized_share", share >= 0.5, f"{share:.1%} recognized by the game's model (minimum 50%)"),
            Check("category_coverage", not short,
                  f"{len(short)} categories under {minimum:,}: {', '.join(short[:5])}" if short
                  else f"all {len(batch.get('categories', {}))} categories have {minimum:,}+ drawings"),
        ]

    # -------------------------------------------------------------- preprocess

    def preprocess(self, run: StageRun) -> None:
        df = pd.DataFrame.from_records(self.read_raw(run.batch_id))
        run.records_in = len(df)

        df = df[df["drawing"].map(valid_strokes)]
        if self.options.get("recognized_only", True):
            df = df[df["recognized"].astype(bool)]
        df = df.drop_duplicates("key_id").copy()

        df["source_id"] = df["key_id"].astype(str)
        df["record_id"] = df["source_id"].map(lambda sid: record_id(self.key, sid))
        df["source"] = self.key
        df["label"] = df["word"].str.strip().str.lower()
        df["strokes"] = df["drawing"].map(lambda d: json.dumps(d, separators=(",", ":")))
        df["n_strokes"] = df["drawing"].map(len)
        df["n_points"] = df["drawing"].map(lambda d: sum(len(stroke[0]) for stroke in d))
        df["image_uri"] = pd.NA
        df["country_code"] = df["countrycode"]
        df["license"] = self.config.license
        df["attribution"] = self.config.attribution
        df["batch_id"] = run.batch_id
        df["ingested_at"] = pd.Timestamp.now(tz="UTC")
        df = conform(df, DRAWINGS)

        out_path = self.processed_path(run.batch_id)
        earlier = sum(len(self.store.read_parquet(p, columns=["record_id"]))
                      for p in self.processed_paths() if p != out_path)
        self.store.write_parquet(out_path, df)
        run.records_out = len(df)
        run.records_total = earlier + len(df)
        run.output_uri = self.store.url(out_path)
