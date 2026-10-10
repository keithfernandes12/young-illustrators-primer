"""CLEAR Corpus (CommonLit & Georgia State University): ~4,700 excerpts with
teacher-rated readability scores, for the readability scorer.

A snapshot source: the corpus is one spreadsheet in the CLEAR GitHub repo, and
a new batch is made only when that file changes.
"""

from __future__ import annotations

import io

import pandas as pd

from primer.data import checks
from primer.data.manifest import StageRun
from primer.data.schemas import READABILITY, conform
from primer.data.sources.base import Download, Source, http_get
from primer.data.text import normalize_text, record_id, sentence_count, text_hash, word_count

# Spreadsheet column -> processed column
COLUMNS = {
    "ID": "source_id",
    "Title": "title",
    "Author": "author",
    "Excerpt": "text",
    "BT_easiness": "score",
    "s.e.": "score_se",
    "Flesch-Kincaid-Grade-Level": "flesch_kincaid_grade",
    "Lexile Band": "lexile_band",
    "Categ": "category",
    "Pub Year": "pub_year",
    "License": "excerpt_license",
}
SCORE_RANGE = (-5.0, 3.0)  # BT_easiness in the published corpus is about -3.7 to 1.7


def _text_or_na(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # Excel returns whole numbers like 900 as 900.0
    return str(value).strip()


class ClearCorpus(Source):
    key = "clear_corpus"
    dataset = "readability"
    snapshot = True

    def __init__(self, config, store, download: Download | None = None):
        super().__init__(config, store)
        self.download = download or http_get

    @property
    def filename(self) -> str:
        return self.options.get("filename", "CLEAR_corpus_final.xlsx")

    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        url = self.options["url"]
        return self.save_snapshot(run, {self.filename: self.download(url)}, url=url)

    def _frame(self, batch_id: str) -> pd.DataFrame:
        data = self.read_raw_file(batch_id, self.filename)
        return pd.read_excel(io.BytesIO(data), sheet_name=self.options.get("sheet", "Data"))

    def validate(self, run: StageRun) -> None:
        df = self._frame(run.batch_id)
        run.records_in = run.records_out = len(df)
        run.output_uri = self.store.url(self.raw_prefix(run.batch_id))
        columns = checks.required_columns(list(df.columns), list(COLUMNS))
        run.checks += [checks.count_between("record_count", len(df), self.options.get("min_records", 4000)),
                       columns]
        if not columns.passed:
            return

        records = df.to_dict("records")
        lo, hi = SCORE_RANGE
        run.checks += [
            checks.unique(records, "ID"),
            checks.max_ratio("empty_text", records, lambda r: not str(r["Excerpt"] or "").strip()
                             or pd.isna(r["Excerpt"]), 0.01),
            checks.max_ratio("missing_score", records, lambda r: pd.isna(r["BT_easiness"]), 0.0),
            checks.max_ratio("score_in_range", records,
                             lambda r: pd.notna(r["BT_easiness"]) and not lo <= r["BT_easiness"] <= hi, 0.0),
        ]

    def preprocess(self, run: StageRun) -> None:
        df = self._frame(run.batch_id)
        run.records_in = len(df)

        df = df[list(COLUMNS)].rename(columns=COLUMNS)
        df["source_id"] = df["source_id"].astype(str)
        df["text"] = df["text"].map(lambda t: normalize_text(t) if isinstance(t, str) else "")
        df = df[(df["text"] != "") & df["score"].notna()].copy()
        df["text_hash"] = df["text"].map(text_hash)
        df = df.drop_duplicates("text_hash")

        for col in ("title", "author", "lexile_band", "category", "excerpt_license"):
            df[col] = df[col].map(_text_or_na)
        df["record_id"] = df["source_id"].map(lambda sid: record_id(self.key, sid))
        df["source"] = self.key
        df["word_count"] = df["text"].map(word_count)
        df["sentence_count"] = df["text"].map(sentence_count)
        df["score_name"] = "BT_easiness"
        df["license"] = self.config.license
        df["attribution"] = self.config.attribution
        df["batch_id"] = run.batch_id
        df["ingested_at"] = pd.Timestamp.now(tz="UTC")
        df = conform(df, READABILITY)

        out_path = self.processed_path(run.batch_id)
        self.store.write_parquet(out_path, df)
        run.records_out = run.records_total = len(df)
        run.output_uri = self.store.url(out_path)
