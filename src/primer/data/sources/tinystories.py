"""TinyStories (Eldan & Li, 2023): short children's stories for story LLM training.

Ingestion streams the Hugging Face dataset in batches and remembers an offset
in ``metadata/state/tinystories.json``, so each run continues where the last
one stopped and nothing is downloaded to the local disk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

import pandas as pd

from primer.data import checks
from primer.data.manifest import StageRun
from primer.data.schemas import STORIES_TEXT, conform
from primer.data.sources.base import Source
from primer.data.text import normalize_text, record_id, sentence_count, text_hash, word_count

log = logging.getLogger(__name__)

Fetch = Callable[[int, int], Iterable[dict]]

PROCESSED_DIR = "processed/stories_text/source=tinystories"


class TinyStories(Source):
    key = "tinystories"
    dataset = "stories_text"

    def __init__(self, config, store, fetch: Fetch | None = None):
        super().__init__(config, store)
        self.fetch = fetch or self._fetch_from_hub

    def _fetch_from_hub(self, offset: int, limit: int) -> Iterable[dict]:
        from datasets import load_dataset

        dataset = load_dataset(
            self.options["hf_dataset"], split=self.options.get("hf_split", "train"), streaming=True
        )
        return dataset.skip(offset).take(limit)

    @property
    def word_range(self) -> tuple[int, int]:
        return self.options.get("min_words", 20), self.options.get("max_words", 1500)

    # ------------------------------------------------------------------ ingest

    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        batch_size = batch_size or self.options.get("batch_size", 50_000)
        state = self.store.read_json(self.state_path, default={"next_offset": 0})
        offset = state["next_offset"]
        field = self.options.get("text_field", "text")

        log.info("Fetching TinyStories rows %s to %s", f"{offset:,}", f"{offset + batch_size:,}")
        records = [
            {"source_id": str(offset + i), "text": row.get(field)}
            for i, row in enumerate(self.fetch(offset, batch_size))
        ]
        run.records_in = 0
        run.records_out = len(records)
        run.records_total = offset + len(records)
        if not records:
            log.info("No new TinyStories rows after offset %s; dataset fully ingested", f"{offset:,}")
            return False

        prefix = self.raw_prefix(run.batch_id)
        self.store.write_jsonl_gz(f"{prefix}/part-00000.jsonl.gz", records)
        self.store.write_json(f"{prefix}/_batch.json", {
            "batch_id": run.batch_id,
            "hf_dataset": self.options["hf_dataset"],
            "hf_split": self.options.get("hf_split", "train"),
            "offset_start": offset,
            "offset_end": offset + len(records),
            "count": len(records),
            "fetched_at": datetime.now(UTC).isoformat(),
        })
        # Advance the offset only after the raw batch is safely written.
        self.store.write_json(self.state_path, {**state, "next_offset": offset + len(records)})
        run.output_uri = self.store.url(prefix)
        return True

    # ---------------------------------------------------------------- validate

    def validate(self, run: StageRun) -> None:
        records = self.read_raw(run.batch_id)
        lo, hi = self.word_range

        def out_of_range(r: dict) -> bool:
            return not lo <= word_count(r.get("text") or "") <= hi

        run.records_in = run.records_out = len(records)
        run.output_uri = self.store.url(self.raw_prefix(run.batch_id))
        run.checks += [
            checks.min_records(records, 1),
            checks.required_fields(records, ["source_id", "text"]),
            checks.unique(records, "source_id"),
            checks.max_ratio("empty_text", records, lambda r: not (r.get("text") or "").strip(), 0.01),
            checks.max_ratio(f"word_count_{lo}_to_{hi}", records, out_of_range, 0.10),
        ]

    # -------------------------------------------------------------- preprocess

    def preprocess(self, run: StageRun) -> None:
        records = self.read_raw(run.batch_id)
        df = pd.DataFrame.from_records(records)
        run.records_in = len(df)

        df["text"] = df["text"].map(normalize_text)
        df["word_count"] = df["text"].map(word_count)
        lo, hi = self.word_range
        df = df[df["word_count"].between(lo, hi)].copy()

        df["text_hash"] = df["text"].map(text_hash)
        df = df.drop_duplicates("text_hash")

        out_path = self.processed_path(run.batch_id)
        seen = self._existing_hashes(exclude=out_path)
        df = df[~df["text_hash"].isin(seen)].copy()

        df["record_id"] = df["source_id"].map(lambda sid: record_id(self.key, sid))
        df["source"] = self.key
        df["title"] = pd.NA
        df["sentence_count"] = df["text"].map(sentence_count)
        df["language"] = "en"
        df["license"] = self.config.license
        df["attribution"] = self.config.attribution
        df["batch_id"] = run.batch_id
        df["ingested_at"] = pd.Timestamp.now(tz="UTC")
        df = conform(df, STORIES_TEXT)

        self.store.write_parquet(out_path, df)
        run.records_out = len(df)
        run.records_total = len(seen) + len(df)
        run.output_uri = self.store.url(out_path)

    def _existing_hashes(self, exclude: str) -> set[str]:
        """Text hashes already in processed/, so re-ingested stories are dropped."""
        hashes: set[str] = set()
        for path in self.processed_paths():
            if path != exclude:
                hashes.update(self.store.read_parquet(path, columns=["text_hash"])["text_hash"])
        return hashes
