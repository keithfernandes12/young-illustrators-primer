"""The LLM story writer: prompt, generate, parse, check, and retry.

`LLMStoryWriter` satisfies `StoryWriter`. It is model-agnostic: anything with a `generate(messages,
settings) -> str` method is a backend, which is how the tests drive it with a scripted stub and how
`model.py` plugs in a Hugging Face model.

**What triggers a retry** (plan Decisions, 2026-11-14, scope change for the Mac submission): exactly
three things --

1. the reply is not the story JSON (`schema.parse_story`);
2. a target word is missing;
3. the story fails the safety check (the story stage's own avoid-list guardrail by default).

Page and sentence limits and the reading level are **measured, not retried on** (`checks.py`, and
`evaluate.py` reports them). After `max_attempts` failures the writer raises `StoryGenerationError`;
it never returns a story that failed one of the three checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from primer.safety.interface import SafetyClassifier
from primer.story.checks import missing_target_words
from primer.story.guardrails import AvoidListGuardrail, load_avoid_list
from primer.story.interface import Story, StoryRequest
from primer.story.presets import mode_for
from primer.story.prompts import Message, build_messages
from primer.story.schema import InvalidStoryJSON, parse_story


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.7
    max_new_tokens: int = 384
    top_p: float = 0.9
    #: The first attempt's seed; attempt i uses seed + i, so a retry is not a replay.
    seed: int = 0


class ChatBackend(Protocol):
    def generate(self, messages: list[Message], settings: GenerationSettings) -> str:
        """The model's reply to a chat, as raw text."""
        ...


@dataclass(frozen=True)
class Attempt:
    #: "ok", "invalid_json", "missing_words" or "unsafe"
    outcome: str
    #: why, without the story text or the character name (safe to log)
    detail: str = ""


@dataclass(frozen=True)
class WriteReport:
    """One request's outcome: the returned story (or None) and every attempt's outcome."""

    story: Story | None
    attempts: tuple[Attempt, ...]
    #: The last story that parsed, even if it then failed a check (None if nothing ever parsed).
    #: For debugging a failed request only: the evaluation never scores it, because a request that
    #: returned no story reached no child (review F4/F19).
    last_parsed: Story | None = None

    @property
    def ok(self) -> bool:
        return self.story is not None


class StoryGenerationError(RuntimeError):
    """No attempt produced a story that passed all three checks. Serving handles this (KAN-121)."""

    def __init__(self, report: WriteReport) -> None:
        outcomes = ", ".join(attempt.outcome for attempt in report.attempts)
        super().__init__(f"no valid story after {len(report.attempts)} attempts ({outcomes})")
        self.report = report


@dataclass
class LLMStoryWriter:
    backend: ChatBackend
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    classifier: SafetyClassifier | None = None
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._avoid_list = load_avoid_list()
        if self.classifier is None:
            self.classifier = AvoidListGuardrail(self._avoid_list)

    def write(self, request: StoryRequest) -> Story:
        report = self.write_with_report(request)
        if report.story is None:
            raise StoryGenerationError(report)
        return report.story

    def write_with_report(self, request: StoryRequest) -> WriteReport:
        mode = mode_for(request.age, request.age_mode)
        messages = build_messages(request, mode, self._avoid_list)
        attempts: list[Attempt] = []
        last_parsed: Story | None = None
        assert self.classifier is not None  # set in __post_init__
        for index in range(self.max_attempts):
            settings = GenerationSettings(
                temperature=self.settings.temperature,
                max_new_tokens=self.settings.max_new_tokens,
                top_p=self.settings.top_p,
                seed=self.settings.seed + index,
            )
            reply = self.backend.generate(messages, settings)
            try:
                story = parse_story(reply)
            except InvalidStoryJSON as exc:
                attempts.append(Attempt("invalid_json", str(exc)))
                continue
            last_parsed = story
            missing = missing_target_words(story, request.target_words)
            if missing:
                attempts.append(Attempt("missing_words", f"{len(missing)} of {len(request.target_words)}"))
                continue
            verdict = self.classifier.check(" ".join((story.title, *story.pages)))
            if verdict.flagged:
                attempts.append(Attempt("unsafe", verdict.category or "flagged"))
                continue
            attempts.append(Attempt("ok"))
            return WriteReport(story=story, attempts=tuple(attempts), last_parsed=story)
        return WriteReport(story=None, attempts=tuple(attempts), last_parsed=last_parsed)
