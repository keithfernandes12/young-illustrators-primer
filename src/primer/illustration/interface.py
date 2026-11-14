"""The illustration contract: the Protocol serving imports, plus a Fake for tests.

A model area never imports ``primer.serving``; serving imports the area. Importing this module must
stay cheap and must not load weights -- ``model.py`` does that, lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Illustration depends on drawing, one way and by design: the consistency model consumes the
# recogniser's output. Nothing in drawing imports illustration, so there is no cycle.
from primer.drawing.interface import Character


@dataclass(frozen=True)
class Illustration:
    """One page illustration. `image_ref` is a URI or cache key, never image bytes."""

    page_index: int
    image_ref: str
    alt_text: str
    #: What the page was conditioned on, so consistency can be scored against the drawing.
    subject: str | None = None
    #: Whether the child's sketch reached the model for this page (ControlNet structure).
    conditioned_on_sketch: bool = False
    #: Which character identity this page was conditioned on (IP-Adapter), as the opaque
    #: `Character.identity_key` rather than the embedding itself. A book has one of these per page,
    #: and the embedding is child-derived, so the vector stays with the illustrator that conditions
    #: on it instead of being copied onto every page.
    identity: str | None = None


@runtime_checkable
class Illustrator(Protocol):
    """Illustrate pages and keep the character consistent. Gated on `clip_consistency`.

    KAN-42 conditions on two different things, and they do different jobs:

    - ``sketch`` -- the child's drawing itself, for ControlNet (scribble/lineart). This is the
      **structure**: what the page should be shaped like.
    - ``characters`` -- the recogniser's output, carrying one embedding per main character, for
      IP-Adapter. This is the **identity**: who the character is, so page four looks like page one.

    Both are optional in the signature because an illustrator may implement only one of them, and a
    page may have no recognised character at all. Neither is optional in practice for a good book:
    without the sketch the page ignores the drawing's shape, and without the embeddings
    `clip_consistency` has nothing to hold a character to.

    ``sketch`` is raw image bytes from a child. Keep it in memory, pass it to the model, and do not
    log it, name an object after it, or persist it outside a path the parent opted into.

    ``drawing_ref`` is a **URI of the original upload that the illustrator can read** -- a
    ``gs://`` object in the drawings bucket, or a ``file://`` path in local development. It is the
    durable handle for the same image ``sketch`` carries in memory: use it to re-read the drawing
    on a retry, to attribute a page to its source, or to avoid holding bytes across a queue.
    ``build_book`` requires it whenever a drawing was given, so an implementation may assume it is
    present and non-empty. The ``fixture://`` scheme appears only in test fixtures and resolves to
    nothing; do not emit it from product code.
    """

    def illustrate(
        self,
        page_index: int,
        page_text: str,
        style: str,
        subject: str | None = None,
        drawing_ref: str | None = None,
        sketch: bytes | None = None,
        characters: tuple[Character, ...] = (),
    ) -> Illustration:
        """One illustration for one page."""
        ...

    def consistency(
        self, illustrations: list[Illustration], characters: tuple[Character, ...] = ()
    ) -> float:
        """How alike the illustrations are, 0..1. The whole point of a picture book.

        `characters` is what they are supposed to be alike *to*: a real scorer compares each page
        against the character embeddings it was conditioned on, not only against the other pages.
        """
        ...


class FakeIllustrator:
    """Deterministic refs and a fixed consistency score. Produces no images."""

    def __init__(self, consistency_score: float = 0.9) -> None:
        self.consistency_score = consistency_score

    def illustrate(
        self,
        page_index: int,
        page_text: str,
        style: str,
        subject: str | None = None,
        drawing_ref: str | None = None,
        sketch: bytes | None = None,
        characters: tuple[Character, ...] = (),
    ) -> Illustration:
        return Illustration(
            page_index=page_index,
            image_ref=f"fake://{style}/{subject or 'unknown'}/page-{page_index}",
            alt_text=page_text[:60],
            subject=subject,
            # Recorded, not stored: a test can see whether serving actually conditioned the page.
            conditioned_on_sketch=sketch is not None,
            identity=characters[0].identity_key if characters and characters[0].has_embedding else None,
        )

    def consistency(
        self, illustrations: list[Illustration], characters: tuple[Character, ...] = ()
    ) -> float:
        """0 unless every page was conditioned on the same character identity.

        A real scorer compares images. This compares what each page was conditioned on, so a
        serving bug that drops the embedding shows up as a failing metric rather than silently.
        """
        if not illustrations:
            return 0.0
        if len({illustration.subject for illustration in illustrations}) != 1:
            return 0.0
        identities = {illustration.identity for illustration in illustrations}
        if len(identities) != 1 or not next(iter(identities)):
            return 0.0
        if characters and next(iter(identities)) != characters[0].identity_key:
            return 0.0
        return self.consistency_score
