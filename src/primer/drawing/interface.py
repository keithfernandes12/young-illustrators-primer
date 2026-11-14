"""The drawing contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.

The recogniser returns a **character sheet per character** plus **one description for the whole
drawing** (KAN-38). The sheet is what KAN-42 reposes and conditions on, and what KAN-64 shows the
parent, so every field below has a named consumer:

- ``label`` + ``confidence`` + ``candidates`` -- what the character is. ``candidates`` is the
  ranked guess list for **this one character**, which is what `drawing_top5_acc` measures.
- ``description`` (on the analysis, not the character) -- what the whole drawing shows, in words
  the story writer can use.
- ``body_type`` -- KAN-42 picks its reposing branch from this, so it is required, not decorative.
- ``crop_ref`` / ``mask_ref`` -- the character cut out of the child's own drawing.
- ``keypoints`` -- humanoids only; drives OpenPose-style reposing.
- ``reference_image_ref`` + ``approved`` -- the storybook-style render the parent signed off.
  IP-Adapter conditions on **this**, never on the crop.
- ``embedding`` -- the identity cache key (KAN-71). *Not* the consistency signal: KAN-48 measures
  consistency with DINOv2 + CLIP, a different space from this encoder's.

``crop_ref`` and ``reference_image_ref`` are two different images and are never interchangeable.
The crop is the child's own crayon lines, extracted. The reference is that character redrawn in
storybook style and approved. Conditioning IP-Adapter on the crop would make every page look like
crayon, which defeats the scoping doc's "consistent storybook style that keeps the child's design".

**Who fills what.** ``analyse()`` produces everything except ``reference_image_ref`` and
``approved``: rendering a character in storybook style is a diffusion call, which belongs to the
illustration pipeline, and approval is a person's decision taken in KAN-64. Those two fields ride
on the character sheet so KAN-42 has one place to read them from; they are set later with
``Character.approve()``, not by the recogniser. Serving has no confirmation step today, so nothing
yet enforces that an unapproved reference never reaches the illustrator -- that gate belongs in
KAN-64.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

#: The joints the Amateur Drawings annotations carry, in order. Confirmed against the ingestion
#: that landed in #12: `data/schemas.py` stores `keypoints` as "17 [x, y, visibility] joints in
#: COCO order" and `data/sources/amateur_drawings.py` pins `N_KEYPOINTS = 17`. Keep this list in
#: that order -- a silent reordering is invisible in a metric and wrong in every reposed page.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class BodyType(StrEnum):
    """How a character is built, which is how KAN-42 decides to repose it.

    A closed set rather than a free string: KAN-42 branches on this value, and a typo would pick
    the wrong reposing path silently rather than failing.
    """

    #: A person or person-like figure. Reposed with keypoints (OpenPose).
    HUMANOID = "humanoid"
    #: A dragon, a monster, an animal. Reposed by prompt, no skeleton.
    CREATURE = "creature"
    #: A house, a car, a tree. Placed rather than reposed.
    OBJECT = "object"
    #: Not determined. KAN-42 must treat this as "do not repose", never as a default branch.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Candidate:
    """One ranked guess at what a character is."""

    label: str
    confidence: float


@dataclass(frozen=True)
class Keypoint:
    """One joint of a humanoid character.

    ``x`` and ``y`` are normalised to the character's crop, 0..1, so they survive a resize. The
    source annotations are in the pixel frame of Meta's resized PNG (`data/schemas.py`: `bbox`,
    `keypoints` and `segmentation` share that frame), so whoever fills this divides by it.

    ``visible`` is COCO's visibility flag reduced to ``> 0``, which is how `amateur_drawings.py`
    already counts `n_keypoints`. It does not distinguish COCO's 1 (labelled but occluded) from 2
    (labelled and visible); nothing downstream needs that yet, and KAN-42 only asks whether a joint
    can be used. A joint that was not seen is kept with ``visible`` False rather than dropped, so
    the list stays positional -- index 3 is the same joint on every character.

    Derived from a child's drawing: treat it as child data, like the embedding.
    """

    name: str
    x: float
    y: float
    visible: bool = True


@dataclass(frozen=True)
class Character:
    """One character found in a drawing: the character sheet of KAN-38.

    Several fields here are derived from the child's drawing -- ``embedding``, ``crop_ref``,
    ``mask_ref``, ``keypoints``. Never log them, never put them in a URL or an object name, and do
    not persist them outside the paths the parent opted into.
    """

    label: str
    confidence: float
    embedding: tuple[float, ...] = ()
    #: This character's ranked guesses, most confident first, including the winning one. Defaults
    #: to the single ``label``/``confidence`` pair so a one-guess recogniser needs no extra work.
    candidates: tuple[Candidate, ...] = ()
    body_type: BodyType = BodyType.UNKNOWN
    #: The character cut out of the child's drawing, as a URI -- never image bytes.
    crop_ref: str = ""
    #: The cutout's mask, as a URI.
    mask_ref: str = ""
    #: Humanoids only. Empty for creatures and objects, which KAN-42 reposes another way.
    keypoints: tuple[Keypoint, ...] = ()
    #: The storybook-style render of this character. Set by the stylise step, not by ``analyse()``.
    reference_image_ref: str = ""
    #: Whether a parent signed the reference off (KAN-64). Set by ``approve()``, never by a model.
    approved: bool = False

    def __post_init__(self) -> None:
        if not self.candidates:
            # A recogniser that offers one guess still has a ranked list; it is one long. This
            # keeps `top_labels` meaningful for every Character, however it was constructed.
            object.__setattr__(self, "candidates", (Candidate(self.label, self.confidence),))
            return
        if self.candidates[0].label != self.label:
            raise ValueError(
                f"label {self.label!r} is not the top candidate {self.candidates[0].label!r}; "
                "drawing_top5_acc is scored from the candidate list, so the two must agree"
            )
        confidences = [candidate.confidence for candidate in self.candidates]
        if confidences != sorted(confidences, reverse=True):
            raise ValueError("candidates must be ranked most confident first")

    @property
    def has_embedding(self) -> bool:
        return bool(self.embedding)

    @property
    def has_approved_reference(self) -> bool:
        """Whether IP-Adapter may condition on this character's reference image.

        Both halves are required: a reference nobody approved is not one we may put in a book.
        """
        return self.approved and bool(self.reference_image_ref)

    @property
    def identity_key(self) -> str:
        """An opaque, stable handle for this character's identity.

        Downstream code needs to know that two pages were conditioned on the *same* character; it
        does not need the vector to answer that. Passing this instead keeps the child-derived
        embedding in the one place that conditions on it, rather than copying it onto every page
        of the finished book.

        Only the label and the embedding go into the key. Approval state and the reference image
        deliberately do not: approving a character, or regenerating its reference once KAN-42's
        style LoRA exists, must not make page four stop matching page one.
        """
        digest = hashlib.sha256(repr((self.label, self.embedding)).encode()).hexdigest()
        return f"{self.label}:{digest[:16]}"

    def top_labels(self, k: int = 5) -> list[str]:
        """This character's k most confident labels, for `drawing_top5_acc`."""
        return [candidate.label for candidate in self.candidates[:k]]

    def approve(self, reference_image_ref: str) -> Character:
        """Return this character with an approved storybook-style reference attached.

        The recogniser never calls this. The confirmation step (KAN-64) does, once a parent has
        looked at the render and said yes.
        """
        if not reference_image_ref:
            raise ValueError("cannot approve a character without a reference image")
        return replace(self, reference_image_ref=reference_image_ref, approved=True)


@dataclass(frozen=True)
class DrawingAnalysis:
    """Everything the recogniser returns for one drawing."""

    #: What the whole drawing shows, in words the story writer can use.
    description: str = ""
    #: The main characters, most confident first.
    characters: tuple[Character, ...] = field(default_factory=tuple)

    @property
    def main(self) -> Character | None:
        """The most confident character, or None for a drawing with nothing in it."""
        return self.characters[0] if self.characters else None

    @property
    def humanoids(self) -> tuple[Character, ...]:
        """The characters KAN-42 reposes with keypoints."""
        return tuple(c for c in self.characters if c.body_type is BodyType.HUMANOID)

    def top_labels(self, k: int = 5) -> list[str]:
        """The main character's k most confident labels, for `drawing_top5_acc`.

        Ranking is **within** one character, not across them (KAN-38). Quick, Draw! records carry
        exactly one label, so the evaluation set is single-character and this is the character
        being scored. Reading one label from each of five characters would quietly turn "five
        guesses at one thing" into "one guess at five things" the moment a drawing has more than
        one character in it.
        """
        return self.main.top_labels(k) if self.main else []


@runtime_checkable
class DrawingRecognizer(Protocol):
    """Recognise what a child drew. Gated on `drawing_top5_acc`."""

    def analyse(self, image: bytes) -> DrawingAnalysis:
        """Describe the drawing and extract its main characters, most confident first.

        Fills every field of the character sheet except ``reference_image_ref`` and ``approved``,
        which the stylise-and-approve step sets later (see the module docstring).
        """
        ...


#: Labels the Fake treats as people, so a test can exercise the humanoid reposing branch without
#: a model. Real body-type inference is KAN-38's job; this is only enough to drive the branch.
_FAKE_HUMANOIDS = frozenset({"child", "girl", "boy", "person", "man", "woman", "mum", "dad"})
_FAKE_OBJECTS = frozenset({"house", "car", "tree", "boat", "sun", "flower"})


class FakeDrawingRecognizer:
    """Returns `labels` as a full character sheet each. No weights, no network.

    Deterministic throughout, so two runs of the same test compare equal. ``approved=True`` fills
    the reference image and approval flag that a real pipeline only gets after KAN-64 -- that is
    how KAN-42 exercises IP-Adapter conditioning before any of this is built.
    """

    def __init__(
        self,
        label: str = "cat",
        also: tuple[str, ...] = (),
        dimensions: int = 4,
        approved: bool = False,
    ) -> None:
        self.labels = (label, *also)
        self.dimensions = dimensions
        self.approved = approved

    def _embedding(self, label: str) -> tuple[float, ...]:
        """Stable per label, so two pages of the same character compare equal."""
        seed = sum(ord(character) for character in label)
        return tuple(round(((seed + index) % 100) / 100, 2) for index in range(self.dimensions))

    def _body_type(self, label: str) -> BodyType:
        if label in _FAKE_HUMANOIDS:
            return BodyType.HUMANOID
        if label in _FAKE_OBJECTS:
            return BodyType.OBJECT
        return BodyType.CREATURE

    def _candidates(self, label: str, confidence: float) -> tuple[Candidate, ...]:
        """The winner plus two deterministic runners-up, so top-5 has something to rank."""
        runners = (f"{label}-like", f"not-a-{label}")
        return (
            Candidate(label, confidence),
            *(
                Candidate(runner, round(max(confidence - 0.2 * (index + 1), 0.0), 2))
                for index, runner in enumerate(runners)
            ),
        )

    def _keypoints(self, label: str) -> tuple[Keypoint, ...]:
        """17 joints on a deterministic diagonal. Humanoids only, like the real thing."""
        seed = sum(ord(character) for character in label)
        return tuple(
            Keypoint(
                name=name,
                x=round(((seed + index) % 100) / 100, 2),
                y=round(index / len(KEYPOINT_NAMES), 2),
                # One joint hidden, so a consumer that assumes every joint is visible fails here
                # rather than on real data (KAN-90).
                visible=name != "left_ankle",
            )
            for index, name in enumerate(KEYPOINT_NAMES)
        )

    def _character(self, label: str, index: int) -> Character:
        confidence = round(1.0 - index * 0.1, 2)
        body_type = self._body_type(label)
        reference = f"fake://reference/{label}" if self.approved else ""
        return Character(
            label=label,
            confidence=confidence,
            embedding=self._embedding(label),
            candidates=self._candidates(label, confidence),
            body_type=body_type,
            crop_ref=f"fake://crop/{label}",
            mask_ref=f"fake://mask/{label}",
            keypoints=self._keypoints(label) if body_type is BodyType.HUMANOID else (),
            reference_image_ref=reference,
            approved=self.approved,
        )

    def analyse(self, image: bytes) -> DrawingAnalysis:
        characters = tuple(
            self._character(label, index) for index, label in enumerate(self.labels)
        )
        described = " and a ".join(self.labels)
        return DrawingAnalysis(description=f"a child's drawing of a {described}", characters=characters)
