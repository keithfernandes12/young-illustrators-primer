"""Text cleaning and counting shared by the text sources."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_SPACES = re.compile(r"[ \t\f\v ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


def normalize_text(text: str | None) -> str:
    """Unicode-normalize, collapse runs of spaces, keep paragraph breaks."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def sentence_count(text: str) -> int:
    if not text:
        return 0
    return max(len(_SENTENCE_END.findall(text)), 1)


def text_hash(text: str) -> str:
    """Hash for exact-duplicate detection, ignoring case and whitespace."""
    key = " ".join(text.lower().split())
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def record_id(source: str, source_id: str) -> str:
    return hashlib.sha1(f"{source}:{source_id}".encode()).hexdigest()[:16]
