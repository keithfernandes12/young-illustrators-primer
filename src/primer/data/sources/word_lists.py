"""Word lists for picking target words: Dolch and Fry sight words (ages 3-5) and
Kuperman age-of-acquisition norms (ages 5-8).

Both are snapshot sources writing to the same ``word_lists`` dataset, so the
target-word selector reads one table with a ``list_name`` column.
"""

from __future__ import annotations

import io

import pandas as pd

from primer.data import checks
from primer.data.manifest import Check, StageRun
from primer.data.schemas import WORD_LISTS, conform
from primer.data.sources.base import Download, Source, http_get
from primer.data.text import record_id

DOLCH_LEVELS = {"pre-primer": 40, "primer": 52, "grade1": 41, "grade2": 46, "grade3": 41}


def read_rda_words(data: bytes, name: str) -> list[str]:
    """Read a character vector saved in an R .rda file, straight from memory."""
    import rdata

    converted = rdata.conversion.convert(rdata.parser.parse_data(data))
    return [str(word) for word in converted[name]]


def _finish(df: pd.DataFrame, source: Source, batch_id: str) -> pd.DataFrame:
    df["record_id"] = [record_id(source.key, f"{lst}:{w}") for lst, w in zip(df["list_name"], df["word"])]
    df["source"] = source.key
    df["license"] = source.config.license
    df["attribution"] = source.config.attribution
    df["batch_id"] = batch_id
    df["ingested_at"] = pd.Timestamp.now(tz="UTC")
    return conform(df, WORD_LISTS)


class SightWords(Source):
    """Dolch (220 service words, by grade level) and Fry (first 1,000 words, by frequency).

    Fetched from the data files of the ``lexicon`` R package; the lists themselves
    are public domain.
    """

    key = "sight_words"
    dataset = "word_lists"
    snapshot = True

    def __init__(self, config, store, download: Download | None = None):
        super().__init__(config, store)
        self.download = download or http_get

    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        urls = {"sw_dolch.rda": self.options["dolch_url"], "sw_fry_1000.rda": self.options["fry_url"]}
        return self.save_snapshot(run, {name: self.download(url) for name, url in urls.items()}, urls=urls)

    def _lists(self, batch_id: str) -> tuple[list[str], list[str]]:
        dolch = read_rda_words(self.read_raw_file(batch_id, "sw_dolch.rda"), "sw_dolch")
        fry = read_rda_words(self.read_raw_file(batch_id, "sw_fry_1000.rda"), "sw_fry_1000")
        return dolch, fry

    def validate(self, run: StageRun) -> None:
        dolch, fry = self._lists(run.batch_id)
        run.records_in = run.records_out = len(dolch) + len(fry)
        run.output_uri = self.store.url(self.raw_prefix(run.batch_id))

        allowed = {w.lower() for w in self.options.get("allowed_duplicates", [])}

        def dupes(words: list[str]) -> int:
            lowered = [w.strip().lower() for w in words if w.strip().lower() not in allowed]
            return len(lowered) - len(set(lowered))

        run.checks += [
            checks.count_between("dolch_words", len(dolch), sum(DOLCH_LEVELS.values()),
                                 sum(DOLCH_LEVELS.values())),
            checks.count_between("fry_words", len(fry), 1000, 1000),
            Check("no_empty_words", all(w.strip() for w in dolch + fry), "every entry has text"),
            Check("unique_words", dupes(dolch) == 0 and dupes(fry) == 0,
                  f"{dupes(dolch)} duplicate Dolch, {dupes(fry)} duplicate Fry"
                  + (f" (known duplicates allowed: {', '.join(sorted(allowed))})" if allowed else "")),
        ]

    def preprocess(self, run: StageRun) -> None:
        dolch, fry = self._lists(run.batch_id)
        run.records_in = len(dolch) + len(fry)

        levels = [level for level, size in DOLCH_LEVELS.items() for _ in range(size)]
        rows = [{"list_name": "dolch", "word": w.strip().lower(), "level": level, "rank": i + 1}
                for i, (w, level) in enumerate(zip(dolch, levels))]
        rows += [{"list_name": "fry", "word": w.strip().lower(), "level": f"fry_{(i // 100 + 1) * 100}",
                  "rank": i + 1} for i, w in enumerate(fry)]
        # Keep the first (most frequent) position of a word the source lists twice.
        df = pd.DataFrame(rows).drop_duplicates(["list_name", "word"])
        df = _finish(df, self, run.batch_id)

        out_path = self.processed_path(run.batch_id)
        self.store.write_parquet(out_path, df)
        run.records_out = run.records_total = len(df)
        run.output_uri = self.store.url(out_path)


KUPERMAN_COLUMNS = ["Word", "AoA_Kup", "Perc_known", "Freq_pm", "Dom_PoS_SUBTLEX", "Nsyll"]


class KupermanAoA(Source):
    """Kuperman, Stadthagen-Gonzalez & Brysbaert (2012) age-of-acquisition ratings.

    Comes from Marc Brysbaert's 51,715-word file on OSF, which also carries word
    frequency and part of speech; only words with a Kuperman rating are kept.
    """

    key = "kuperman_aoa"
    dataset = "word_lists"
    snapshot = True

    def __init__(self, config, store, download: Download | None = None):
        super().__init__(config, store)
        self.download = download or http_get

    @property
    def filename(self) -> str:
        return self.options.get("filename", "AoA_51715_words.xlsx")

    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        url = self.options["url"]
        return self.save_snapshot(run, {self.filename: self.download(url)}, url=url)

    def _frame(self, batch_id: str) -> pd.DataFrame:
        return pd.read_excel(io.BytesIO(self.read_raw_file(batch_id, self.filename)))

    def validate(self, run: StageRun) -> None:
        df = self._frame(run.batch_id)
        run.records_in = run.records_out = len(df)
        run.output_uri = self.store.url(self.raw_prefix(run.batch_id))
        columns = checks.required_columns(list(df.columns), KUPERMAN_COLUMNS)
        run.checks.append(columns)
        if not columns.passed:
            return

        rated = df[df["AoA_Kup"].notna()]
        rated_records = rated.to_dict("records")
        words = df["Word"].dropna().astype(str).str.strip().str.lower()
        dupes = int(words.duplicated().sum())
        run.checks += [
            checks.count_between("rated_words", len(rated), self.options.get("min_rated", 30000)),
            Check("unique_words", dupes <= len(words) * 0.001, f"{dupes:,} duplicate words (limit 0.1%)"),
            checks.max_ratio("aoa_in_range", rated_records, lambda r: not 1 <= r["AoA_Kup"] <= 25, 0.0),
            checks.max_ratio("percent_known_in_range", rated_records,
                             lambda r: pd.notna(r["Perc_known"]) and not 0 <= r["Perc_known"] <= 1, 0.0),
        ]

    def preprocess(self, run: StageRun) -> None:
        df = self._frame(run.batch_id)
        run.records_in = len(df)

        df = df[df["AoA_Kup"].notna() & df["Word"].notna()].copy()
        df["word"] = df["Word"].astype(str).str.strip().str.lower()
        df = df[df["word"] != ""].drop_duplicates("word")
        df = df.rename(columns={"AoA_Kup": "age_of_acquisition", "Perc_known": "percent_known",
                                "Freq_pm": "frequency_per_million", "Dom_PoS_SUBTLEX": "part_of_speech",
                                "Nsyll": "n_syllables"})
        df["list_name"] = "kuperman_aoa"
        df["part_of_speech"] = df["part_of_speech"].map(lambda v: pd.NA if pd.isna(v) else str(v))
        df = _finish(df, self, run.batch_id)

        out_path = self.processed_path(run.batch_id)
        self.store.write_parquet(out_path, df)
        run.records_out = run.records_total = len(df)
        run.output_uri = self.store.url(out_path)
