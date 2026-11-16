"""The JSON the model is asked for, and the parser that turns it into a `Story`.

The model must answer with exactly::

    {"title": "...", "pages": [{"text": "..."}, ...]}

A reply counts as **valid JSON** when, after stripping whitespace and one optional Markdown code fence
(```json ... ```), it parses as that object: a non-empty string title of at most 80 characters, and a
list of pages that each have a non-empty string `text` of at most 400 characters and no other keys.
Page-count limits are not checked here; they belong to the age-mode preset and are measured, not used
to reject the JSON.
"""

from __future__ import annotations

import json
import re

from primer.story.interface import Story

MAX_TITLE = 80
MAX_PAGE = 400

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class InvalidStoryJSON(ValueError):
    """The model's reply is not the story JSON. The message says why, for the attempt log."""


def parse_story(reply: str) -> Story:
    """Parse the model's reply into a `Story`, or raise `InvalidStoryJSON`."""
    text = reply.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidStoryJSON(f"not JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise InvalidStoryJSON("top level is not an object")
    extra = set(data) - {"title", "pages"}
    if extra:
        raise InvalidStoryJSON(f"unexpected keys: {sorted(extra)}")
    title = data.get("title")
    if not isinstance(title, str) or not title.strip() or len(title) > MAX_TITLE:
        raise InvalidStoryJSON("title must be a non-empty string of at most 80 characters")
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        raise InvalidStoryJSON("pages must be a non-empty list")
    texts = []
    for index, page in enumerate(pages):
        if not isinstance(page, dict) or set(page) != {"text"}:
            raise InvalidStoryJSON(f"page {index} must be an object with only a 'text' key")
        body = page["text"]
        if not isinstance(body, str) or not body.strip() or len(body) > MAX_PAGE:
            raise InvalidStoryJSON(f"page {index} text must be a non-empty string of at most 400 characters")
        texts.append(body.strip())
    return Story(pages=tuple(texts), title=title.strip())
