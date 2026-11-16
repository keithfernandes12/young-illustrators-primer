"""The story stage's own safety guardrail: an age-appropriate avoid-list.

`AvoidListGuardrail` satisfies the safety area's `SafetyClassifier` protocol, so the writer can take
either it or member 3's classifier, and serving's existing check after `write` stays an independent
second line. It matches whole words only, using the same tokenisation as `levels.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from primer.safety.interface import SafetyVerdict
from primer.story.levels import words

DEFAULT_AVOID_LIST = Path(__file__).parent / "configs" / "avoid_list.yaml"


@dataclass(frozen=True)
class AvoidList:
    version: str
    #: category -> the topic phrase the prompt uses
    prompts: dict[str, str]
    #: lower-cased word -> its category
    words: dict[str, str]


def load_avoid_list(path: Path = DEFAULT_AVOID_LIST) -> AvoidList:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts: dict[str, str] = {}
    lookup: dict[str, str] = {}
    for category, spec in raw["categories"].items():
        prompts[category] = str(spec["prompt"])
        for word in spec["words"]:
            lookup[str(word).lower()] = category
    return AvoidList(version=str(raw["version"]), prompts=prompts, words=lookup)


class AvoidListGuardrail:
    """Flags text that uses any avoid-list word. The category is the first match's."""

    def __init__(self, avoid_list: AvoidList | None = None) -> None:
        self.avoid_list = avoid_list or load_avoid_list()

    def check(self, text: str) -> SafetyVerdict:
        for token in words(text):
            category = self.avoid_list.words.get(token.lower())
            if category is not None:
                # The reason names the category, not the word: verdicts can end up in logs.
                return SafetyVerdict(flagged=True, category=category, reason=f"avoid-list: {category}")
        return SafetyVerdict(flagged=False)
