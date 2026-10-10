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


# processed/storybook_pages/source=<source>/batch=<batch_id>.parquet
STORYBOOK_PAGES = {
    "record_id": "string",
    "source": "string",
    "book_id": "string",
    "title": "string",
    "page_number": "int32",            # 1-based, story pages only (no cover)
    "n_pages": "int32",
    "text": "string",
    "word_count": "int32",
    "image_uri": "string",             # illustration for this page (null if pages and images don't line up)
    "image_caption": "string",         # illustration description, where the source provides one
    "image_alignment": "string",       # cover+pages | cover+pages+back | pages | unaligned | no_images
    "author": "string",
    "illustrator": "string",
    "translator": "string",
    "language": "string",
    "reading_level": "string",         # not in the source; filled in later by the readability scorer
    "license": "string",
    "attribution": "string",
    "batch_id": "string",
    "ingested_at": "datetime64[ns, UTC]",
}

# processed/drawings/source=<source>/batch=<batch_id>.parquet
DRAWINGS = {
    "record_id": "string",
    "source": "string",
    "source_id": "string",
    "label": "string",
    "strokes": "string",        # JSON: [[[x...], [y...]], ...] on a 0-255 grid (null for image sources)
    "n_strokes": "int32",
    "n_points": "int32",
    "image_uri": "string",      # for sources that ship images (null for stroke sources)
    "country_code": "string",
    "recognized": "boolean",
    "license": "string",
    "attribution": "string",
    "batch_id": "string",
    "ingested_at": "datetime64[ns, UTC]",
}

# processed/readability/source=<source>/batch=<batch_id>.parquet
READABILITY = {
    "record_id": "string",
    "source": "string",
    "source_id": "string",
    "title": "string",
    "author": "string",
    "text": "string",
    "word_count": "int32",
    "sentence_count": "int32",
    "score": "float64",                 # higher = easier to read
    "score_se": "float64",
    "score_name": "string",
    "flesch_kincaid_grade": "float64",  # classic-formula baseline the scorer must beat
    "lexile_band": "string",
    "category": "string",
    "pub_year": "Int32",
    "excerpt_license": "string",
    "text_hash": "string",
    "license": "string",
    "attribution": "string",
    "batch_id": "string",
    "ingested_at": "datetime64[ns, UTC]",
}

# processed/word_lists/source=<source>/batch=<batch_id>.parquet
WORD_LISTS = {
    "record_id": "string",
    "source": "string",
    "list_name": "string",              # dolch | fry | kuperman_aoa
    "word": "string",
    "level": "string",                  # e.g. pre-primer, grade1, fry_100 (null for AoA norms)
    "rank": "Int32",                    # position in the list (frequency order for Fry)
    "age_of_acquisition": "float64",    # years (Kuperman)
    "percent_known": "float64",
    "frequency_per_million": "float64",
    "part_of_speech": "string",
    "n_syllables": "Int32",
    "license": "string",
    "attribution": "string",
    "batch_id": "string",
    "ingested_at": "datetime64[ns, UTC]",
}


def conform(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Return ``df`` with exactly the schema's columns, in order and with its dtypes."""
    out = df.reindex(columns=list(schema))
    return out.astype(schema)
