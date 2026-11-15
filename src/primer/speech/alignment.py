"""Align what a child said to the page text and decide which words were misread.

Pure string work, no model: it runs on a transcript from any ``SpeechRecognizer``. Matching is
lenient on purpose -- case, punctuation and near-misses count as correct -- so an accent or a
young child's pronunciation is not marked wrong (scoping section 9, failure analysis).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from primer.speech.interface import ReadingResult, Transcript, WordResult

#: Two words at least this similar (difflib ratio) count as the same word.
NEAR_MISS = 0.8

_NOT_WORD = re.compile(r"[^\w']+")


def normalize(text: str) -> list[str]:
    """Lower-cased words with punctuation removed; apostrophes inside words are kept."""
    return [w.strip("'") for w in _NOT_WORD.sub(" ", text.lower()).split() if w.strip("'")]


def _same_word(expected: str, heard: str) -> bool:
    return expected == heard or SequenceMatcher(None, expected, heard).ratio() >= NEAR_MISS


def align(page_index: int, page_text: str, transcript: Transcript) -> ReadingResult:
    """Compare `transcript` with `page_text` word by word."""
    expected = normalize(page_text)
    heard = normalize(transcript.text)
    results: list[WordResult] = []
    matcher = SequenceMatcher(None, expected, heard, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            results += [WordResult(e, h, "correct") for e, h in zip(expected[i1:i2], heard[j1:j2])]
        elif op == "delete":
            results += [WordResult(e, None, "skipped") for e in expected[i1:i2]]
        elif op == "insert":
            results += [WordResult(None, h, "inserted") for h in heard[j1:j2]]
        else:  # replace: pair words up in order, then whatever is left over
            exp, got = expected[i1:i2], heard[j1:j2]
            for e, h in zip(exp, got):
                results.append(WordResult(e, h, "correct" if _same_word(e, h) else "misread"))
            results += [WordResult(e, None, "skipped") for e in exp[len(got) :]]
            results += [WordResult(None, h, "inserted") for h in got[len(exp) :]]
    return ReadingResult(page_index=page_index, words=tuple(results))
