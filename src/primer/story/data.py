"""Books, splits and validation requests for the story stage, from a frozen data version.

The steps follow the plan's "Building the pairs and the splits":

1. **One pair per book.** Picture books come only from `storybook_pages` (grouped by source and
   book_id, pages in order); from `stories_text`, only `source == "tinystories"` rows. A picture book's
   `stories_text` row is the same text as its pages, so reading both would count it twice.
2. (The TinyStories hash pool is for training pairs; validation requests do not need it.)
3. **Cross-source exact dedupe** on `text_hash` over the full story, keeping a fixed source order so a
   duplicate keeps its picture-book attribution.
4. **The split is a pure hash** of `book_key = f"{source}/{book_id}"`: `split_of`'s hash
   (`illustration/consistency/experiment.py`), inlined because model areas do not import each other,
   at 80/10/10. Level is never part of the key. It moves to `src/primer/data` under KAN-120.

Validation requests are then drawn from the **val** split only, for ages 3-5. Offline only: this
module reads `primer.data`, which nothing on the request path may do.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from primer.data.text import text_hash
from primer.story.interface import StoryRequest
from primer.story.levels import FK_VERSION, fk_grade, words
from primer.story.presets import LEARN_TO_READ

#: Dedupe keeps the first source in this order, so a duplicate keeps its picture-book attribution.
SOURCE_ORDER = ("african_storybook", "storyweaver", "tinystories")
SPLITS = (("train", 0.8), ("val", 0.9), ("test", 1.0))

#: Places a 3-5 story can be "in". A request's setting is the first one its book mentions, or None.
SETTINGS = ("garden", "park", "forest", "farm", "beach", "school", "house", "kitchen", "river", "village",
            "sea", "zoo", "market", "jungle", "pond", "tree", "sky", "home")

_STOPWORDS = frozenset(
    ["the", "a", "an", "and", "or", "but", "so", "then", "when", "was", "were", "is", "are", "am", "be", "been", "to", "of", "in", "on", "at", "for", "with", "by", "from", "up", "down", "out", "into", "over", "under", "they", "them", "their", "there", "this", "that", "these", "those", "he", "she", "it", "his", "her", "its", "we", "you", "i", "me", "my", "our", "your", "said", "says", "did", "does", "do", "not", "no", "yes", "very", "too", "all", "one", "two", "can", "could", "would", "will", "just", "like", "went", "came", "got", "get", "had", "has", "have", "what", "who", "where", "why", "how", "him"]
)
_CAPITALISED = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")


def book_key(source: str, book_id: str) -> str:
    """The split key: `source/book_id`, the format `split_of` already uses."""
    return f"{source}/{book_id}"


def hash_fraction(key: str, salt: str = "") -> float:
    """A stable number in [0, 1) for a key: the first 8 bytes of SHA-256, as in `split_of`."""
    return int.from_bytes(hashlib.sha256((salt + key).encode()).digest()[:8], "big") / 2**64


def split_of_book(key: str) -> str:
    """train / val / test at 80/10/10, from the key alone. Never depends on other books or on level."""
    fraction = hash_fraction(key)
    return next(name for name, upper in SPLITS if fraction < upper)


def books_from_frames(pages: pd.DataFrame, stories: pd.DataFrame) -> pd.DataFrame:
    """One row per book: picture books from `pages`, TinyStories from `stories` (steps 1 and 3).

    Returns columns book_key, source, book_id, title, pages (tuple of str), text, text_hash, license,
    attribution, split. A duplicate dropped across sources is counted in ``attrs["dropped_duplicates"]``.
    """
    rows: list[dict[str, Any]] = []
    if not pages.empty:
        ordered = pages.sort_values(["source", "book_id", "page_number"])
        for (source, book_id), group in ordered.groupby(["source", "book_id"], sort=False):
            texts = tuple(str(t) for t in group["text"] if isinstance(t, str) and t.strip())
            if not texts:
                continue
            first = group.iloc[0]
            rows.append({"source": source, "book_id": str(book_id), "title": first.get("title"),
                         "pages": texts, "text": "\n\n".join(texts),
                         "license": first.get("license"), "attribution": first.get("attribution")})
    if not stories.empty:
        tiny = stories[stories["source"] == "tinystories"]
        for record in tiny.itertuples(index=False):
            rows.append({"source": record.source, "book_id": str(record.source_id), "title": record.title,
                         "pages": (str(record.text),), "text": str(record.text),
                         "license": record.license, "attribution": record.attribution})
    books = pd.DataFrame(rows, columns=["source", "book_id", "title", "pages", "text", "license", "attribution"])
    if books.empty:
        books = books.assign(book_key=[], text_hash=[], split=[])
        books.attrs["dropped_duplicates"] = 0
        return books
    books["book_key"] = [book_key(s, b) for s, b in zip(books["source"], books["book_id"])]
    books["text_hash"] = books["text"].map(text_hash)
    rank = {source: i for i, source in enumerate(SOURCE_ORDER)}
    books["_rank"] = books["source"].map(lambda s: rank.get(s, len(rank)))
    books = books.sort_values(["_rank", "book_key"], kind="stable")
    before = len(books)
    books = books.drop_duplicates("text_hash", keep="first").drop(columns="_rank")
    books["split"] = books["book_key"].map(split_of_book)
    books = books.reset_index(drop=True)
    books.attrs["dropped_duplicates"] = before - len(books)
    return books


@dataclass(frozen=True)
class ValidationRequest:
    """One request for model comparison, with where it came from (for credit and slicing)."""

    request_id: str
    book_key: str
    source: str
    age: int
    target_words: tuple[str, ...]
    character_name: str | None
    setting: str | None
    reading_level: float
    level_source: str
    license: str | None
    attribution: str | None

    def to_request(self) -> StoryRequest:
        return StoryRequest(
            prompt="", age=self.age, target_words=self.target_words, character_name=self.character_name,
            setting=self.setting, age_mode=LEARN_TO_READ.name, reading_level=self.reading_level,
        )


def character_of(text: str) -> str | None:
    """The most frequent capitalised word not at a sentence start, if it appears at least twice."""
    counts = Counter(m for m in _CAPITALISED.findall(text) if m.lower() not in _STOPWORDS)
    if not counts:
        return None
    name, n = counts.most_common(1)[0]
    return name if n >= 2 else None


def setting_of(text: str) -> str | None:
    lowered = {w.lower() for w in words(text)}
    for place in SETTINGS:
        if place in lowered:
            return f"the {place}"
    return None


def target_words_of(text: str, key: str, n: int, vocabulary: frozenset[str] | None = None) -> tuple[str, ...]:
    """`n` content words the story uses, chosen by a hash of the book key (deterministic).

    Prefers words in `vocabulary` (e.g. Dolch and Fry for ages 3-5) when given. Because the words come
    from the book's own text, a story that uses them exists: inclusion is satisfiable by construction.
    """
    seen: dict[str, None] = {}
    for token in words(text):
        lowered = token.lower()
        if len(lowered) >= 3 and lowered.isalpha() and lowered not in _STOPWORDS and token[0].islower():
            seen.setdefault(lowered, None)
    candidates = list(seen)
    if vocabulary:
        preferred = [w for w in candidates if w in vocabulary]
        if len(preferred) >= n:
            candidates = preferred
    candidates.sort(key=lambda word: hash_fraction(word, salt=key))
    return tuple(candidates[:n])


def build_validation_requests(
    books: pd.DataFrame, n: int = 30, vocabulary: frozenset[str] | None = None
) -> list[ValidationRequest]:
    """`n` requests for ages 3-5 from the **val** split, balanced across sources, deterministic.

    Only books whose whole-story FK grade is within the Learn to Read range qualify. Ages cycle 3, 4, 5.
    Requests are taken round-robin across sources in hash order, so one source cannot fill the set.
    """
    val = books[books["split"] == "val"].copy()
    val["level"] = val["text"].map(fk_grade)
    val = val[val["level"] <= LEARN_TO_READ.max_level]
    val["_order"] = val["book_key"].map(lambda key: hash_fraction(key, salt="requests:"))
    queues = {source: list(group.sort_values("_order").itertuples(index=False))
              for source, group in val.groupby("source", sort=True)}
    picked: list[Any] = []
    while len(picked) < n and any(queues.values()):
        for source in sorted(queues):
            if queues[source] and len(picked) < n:
                picked.append(queues[source].pop(0))
    requests = []
    for index, book in enumerate(picked):
        target = target_words_of(book.text, book.book_key, 2 + index % 2, vocabulary)
        if len(target) < 2:
            continue
        requests.append(ValidationRequest(
            request_id=hashlib.sha1(book.book_key.encode()).hexdigest()[:12],
            book_key=book.book_key, source=book.source, age=3 + index % 3, target_words=target,
            character_name=character_of(book.text), setting=setting_of(book.text),
            reading_level=float(book.level), level_source=FK_VERSION,
            license=book.license, attribution=book.attribution,
        ))
    return requests


def write_requests(requests: list[ValidationRequest], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(asdict(request), sort_keys=True) + "\n")


def read_requests(path: Path) -> list[ValidationRequest]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            raw["target_words"] = tuple(raw["target_words"])
            out.append(ValidationRequest(**raw))
    return out


def read_dataset(version: str, split: str = "val", store: Any = None) -> pd.DataFrame:
    """The books of one split at a frozen data version. Reads the bucket: person-run, never in tests."""
    from primer.data.catalog import read_dataset as read

    books = books_from_frames(read("storybook_pages", store, version=version),
                              read("stories_text", store, sources=["tinystories"], version=version))
    return books[books["split"] == split].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Build the validation requests: `python -m primer.story.data --data-uri gs://... --out ...`."""
    import argparse

    from primer.data.catalog import read_dataset as read
    from primer.data.storage import Store

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data-uri", required=True, help="where the frozen version lives (read only)")
    parser.add_argument("--version", default="v1.1")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path(".local-data/story/validation_requests.jsonl"))
    args = parser.parse_args(argv)

    store = Store(args.data_uri)
    books = books_from_frames(read("storybook_pages", store, version=args.version),
                              read("stories_text", store, sources=["tinystories"], version=args.version))
    lists = read("word_lists", store, sources=["sight_words"], version=args.version)
    vocabulary = frozenset(str(w).lower() for w in lists["word"]) if not lists.empty else None
    requests = build_validation_requests(books, n=args.n, vocabulary=vocabulary)
    write_requests(requests, args.out)
    print(f"{len(books)} books ({books.attrs['dropped_duplicates']} cross-source duplicates dropped), "
          f"{(books['split'] == 'val').sum()} in val; wrote {len(requests)} requests to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
