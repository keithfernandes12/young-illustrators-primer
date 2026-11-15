"""The prompt for one story: rules for ages 3-5, the avoid-list topics, and the JSON format.

The avoid-list goes in as **topics** ("do not include: violence, weapons, ..."), not as its word list;
the words are used only to check the output (`guardrails.py`). The target reading level goes in too,
so the model is told what `fk_within1_rate` measures it against. The character's name goes into the
prompt, never into a log.
"""

from __future__ import annotations

from primer.story.guardrails import AvoidList
from primer.story.interface import StoryRequest
from primer.story.presets import AgeMode

Message = dict[str, str]

FORMAT = '{"title": "...", "pages": [{"text": "..."}, {"text": "..."}]}'


def build_messages(request: StoryRequest, mode: AgeMode, avoid_list: AvoidList) -> list[Message]:
    """Chat messages for one request: a system message with the rules, a user message with the ask."""
    avoid = "; ".join(avoid_list.prompts.values())
    system = (
        f"You write very short picture-book stories for children aged {mode.min_age} to {mode.max_age} "
        "who are learning to read.\n"
        "Rules:\n"
        f"- Write {mode.min_pages} to {mode.max_pages} pages.\n"
        f"- Each page has at most {mode.max_sentences_per_page} short sentences of at most "
        f"{mode.max_words_per_sentence} words.\n"
        "- Use simple, everyday words a young child knows.\n"
        "- Use every word in the word list at least once, spelled exactly as given.\n"
        f"- Keep the story gentle and kind, with a happy ending. Do not include: {avoid}.\n"
        f"- Answer with JSON only, in exactly this shape: {FORMAT}"
    )
    character = request.character_name or request.subject or "a little friend"
    parts = [f"Write a story about {character}"]
    if request.setting:
        parts.append(f" in {request.setting}")
    parts.append(".")
    if request.prompt:
        parts.append(f" The child asked for: {request.prompt}.")
    if request.description:
        parts.append(f" Their drawing shows: {request.description}.")
    if request.target_words:
        parts.append(" Word list: " + ", ".join(request.target_words) + ".")
    if request.reading_level is not None:
        parts.append(f" Reading level: {level_phrase(request.reading_level)} or easier.")
    return [{"role": "system", "content": system}, {"role": "user", "content": "".join(parts)}]


def level_phrase(level: float) -> str:
    """The target FK grade as words a model understands: below grade 1 is "kindergarten"."""
    if level < 1.0:
        return "kindergarten"
    return f"US grade {level:g}"
