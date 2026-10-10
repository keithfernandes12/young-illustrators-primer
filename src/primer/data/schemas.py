"""Standard column layouts for the processed datasets.

Every source that produces the same kind of data writes the same columns, so
model code can read all of them together.
"""

from __future__ import annotations

import pandas as pd

# processed/stories_text/source=<source>/batch=<batch_id>.parquet
STORIES_TEXT = {
    "record_id": "string",
    "source": "string",
    "source_id": "string",
    "title": "string",
    "text": "string",
    "word_count": "int32",
    "sentence_count": "int32",
    "text_hash": "string",
    "language": "string",
    "license": "string",
    "attribution": "string",
    "batch_id": "string",
    "ingested_at": "datetime64[ns, UTC]",
}


def conform(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Return ``df`` with exactly the schema's columns, in order and with its dtypes."""
    out = df.reindex(columns=list(schema))
    return out.astype(schema)
