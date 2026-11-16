"""Stub models for the story tests. No weights, no network, no torch.

Named `story_fakes`, not `fakes`: `tests/` has no `__init__.py`, so a bare `fakes` module would
collide with another area's the moment one exists (see `tests/drawing/conftest.py`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from primer.story.writer import GenerationSettings


def story_json(pages: Sequence[str], title: str = "A Good Day") -> str:
    """A reply in exactly the format the writer asks for."""
    return json.dumps({"title": title, "pages": [{"text": page} for page in pages]})


GOOD_PAGES = ("Pip sees a red ball.", "Pip and the cat play.", "They nap in the sun.", "Pip is happy.")


class ScriptedBackend:
    """Replies with `replies` in order (the last one repeats), and records every call's settings."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.calls: list[GenerationSettings] = []
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        self.calls.append(settings)
        self.messages.append(messages)
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


class WordEchoBackend:
    """A deterministic 'model' that writes a 4-page story using every word in the prompt's word list.

    `invalid_first` makes its first reply to each request broken JSON, to exercise the retry path. "First"
    is per request *and* per grid point (temperature, token budget): the grid reuses one backend per
    model across grid points, and a stub that only fumbled on the first grid point would make the
    grid's tie-break test pass for the wrong reason.
    """

    def __init__(self, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self._seen: set[tuple[str, float, int]] = set()

    def generate(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str:
        user = messages[-1]["content"]
        # The word list ends at its full stop; a reading-level sentence may follow it.
        words = user.split("Word list: ", 1)[1].split(".", 1)[0].split(", ") if "Word list: " in user else []
        key = (user, settings.temperature, settings.max_new_tokens)
        if self.invalid_first and key not in self._seen:
            self._seen.add(key)
            return "Once upon a time {not json"
        pages = [f"The {word} is here." for word in words]
        while len(pages) < 4:
            pages.append("We play in the sun.")
        return story_json(pages)
