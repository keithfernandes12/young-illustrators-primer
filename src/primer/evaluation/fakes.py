"""Fake stage components for the golden runner.

No area under `src/primer/<area>/` is implemented yet, so the golden set runs against these. They are
deliberately adjustable: a test builds a fake that drops a target word from every story, or misses a
safety case, and asserts the gate turns red. That is how we know the gate measures anything at all.

Nothing here loads weights, reads GCS or touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeDrawingClassifier:
    """Returns a top-5 label list. `miss_every` makes it wrong on that fraction of scenarios."""

    miss_every: int = 0
    _seen: int = field(default=0, init=False)

    #: Deliberately not drawn from the golden set's own labels. The first version used
    #: ["kite", "frog", "star", "boat"], all four of which ARE golden labels, so a forced miss still
    #: contained the true label for 4 of 16 subjects and `miss_every=1` scored 0.25 instead of 0.0.
    FILLER = ("decoy-one", "decoy-two", "decoy-three", "decoy-four", "decoy-five")

    def top5(self, true_label: str) -> list[str]:
        self._seen += 1
        wrong = self.miss_every and self._seen % self.miss_every == 0
        filler = [label for label in self.FILLER if label != true_label]
        return filler[:5] if wrong else [true_label, *filler[:4]]


@dataclass
class FakeStoryWriter:
    """Writes a story containing the target words.

    `drop_target_words` omits that many words from *every* story, which is what a regressed story
    model looks like: not one bad scenario, a systematic failure to honour the word list.
    """

    drop_target_words: int = 0
    quality: float = 4.6

    def write(self, prompt: str, target_words: list[str], age: int) -> str:
        kept = target_words[self.drop_target_words :] if self.drop_target_words else target_words
        body = " ".join(f"The {word} was there." for word in kept)
        # The prompt is deliberately NOT echoed: it contains the subject word, which is usually also
        # a target word, so quoting it would smuggle a dropped word back into the story and make
        # target_word_inclusion impossible to fail.
        return f"A story for a child of {age}. {body}".strip()

    def quality_score(self, story: str) -> float:
        return self.quality


@dataclass
class FakeReadabilityScorer:
    """Predicts a Flesch-Kincaid grade. `bias` shifts every prediction, worsening the RMSE."""

    bias: float = 0.2

    def grade(self, story: str, target: float) -> float:
        return target + self.bias

    @staticmethod
    def baseline_grade(story: str) -> float:
        """The off-the-shelf comparison the gate measures against: a fixed guess."""
        return 2.5


@dataclass
class FakeSafetyClassifier:
    """Flags scenarios the golden set marks as unsafe. `miss_every` lets one through."""

    miss_every: int = 0
    _seen: int = field(default=0, init=False)

    def flagged(self, prompt: str, expect_flagged: bool) -> bool:
        if not expect_flagged:
            return False
        self._seen += 1
        return not (self.miss_every and self._seen % self.miss_every == 0)


@dataclass
class FakeIllustrator:
    """Reports CLIP consistency between a page illustration and its text."""

    consistency: float = 0.82

    def consistency_score(self, story: str, label: str) -> float:
        return self.consistency


@dataclass
class FakeSystemProfile:
    """Stand-in latency and cost figures, so the system thresholds are exercised too."""

    first_page_s: float = 8.0
    full_book_s: float = 61.0
    cost_per_book_usd: float = 0.11


@dataclass
class FakeComponents:
    """Every stage a golden run needs."""

    drawing: FakeDrawingClassifier = field(default_factory=FakeDrawingClassifier)
    story: FakeStoryWriter = field(default_factory=FakeStoryWriter)
    readability: FakeReadabilityScorer = field(default_factory=FakeReadabilityScorer)
    safety: FakeSafetyClassifier = field(default_factory=FakeSafetyClassifier)
    illustration: FakeIllustrator = field(default_factory=FakeIllustrator)
    system: FakeSystemProfile = field(default_factory=FakeSystemProfile)
