"""The synthetic books for `consistency()` mean what they claim.

`IllustratorModel.consistency()` still raises until it calls the stage 6 scorer (KAN-105). These
tests pin the fixtures that wiring will be tested on, so that when a score comes out wrong it is the
scorer under suspicion and not the book. They check pixels, not embeddings: no weights, no network.
"""

import io
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from PIL import Image

from primer.illustration.model import IllustratorModel

#: (expect_pass, weakest_page) for every book. None for expect_pass: undecided until KAN-105.
VERDICTS = {
    "consistent": (True, None),
    "wrong_character": (False, None),
    "one_page_drifts": (False, 2),
    "two_characters": (True, None),
    "unapproved_reference": (None, None),
    "no_reference": (None, None),
}


def _path(ref: str) -> Path:
    """The local path of a ``file://`` URI, decoded: ``%20`` is a space, and ``/C:/`` is ``C:\\``."""
    parsed = urlparse(ref)
    assert parsed.scheme == "file"
    return Path(url2pathname(parsed.path))


def _read(ref: str) -> bytes:
    return _path(ref).read_bytes()


def _coverage(data: bytes, colour: tuple[int, int, int]) -> float:
    """Share of pixels drawn in exactly ``colour``: which character a page shows, without a model."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    counts = {rgb: n for n, rgb in image.getcolors(maxcolors=image.width * image.height) or []}
    return counts.get(colour, 0) / (image.width * image.height)


@pytest.fixture
def shows(looks):
    """Which of the two looks an image shows, by its body colour."""
    pip, zog = looks

    def which(data: bytes):
        on_pip, on_zog = _coverage(data, pip.body), _coverage(data, zog.body)
        assert max(on_pip, on_zog) > 0.02, "the image shows no character"
        return pip if on_pip > on_zog else zog

    return which


def test_six_books_cover_the_cases(synthetic_books):
    assert set(synthetic_books) == {
        "consistent",
        "wrong_character",
        "one_page_drifts",
        "two_characters",
        "unapproved_reference",
        "no_reference",
    }
    for book in synthetic_books.values():
        assert len(book.illustrations) == 4
        assert [page.page_index for page in book.illustrations] == [0, 1, 2, 3]


def test_every_page_shows_the_character_it_claims(synthetic_books, shows):
    for name, book in synthetic_books.items():
        for page, look in zip(book.illustrations, book.page_looks, strict=True):
            assert shows(_read(page.image_ref)) is look, f"{name} page {page.page_index}"


def test_references_show_their_character_and_carry_the_approval_state(synthetic_books, looks, shows):
    for name, book in synthetic_books.items():
        for character in book.characters:
            if not character.reference_image_ref:
                continue
            expected = looks[0] if character.label == "pip" else looks[1]
            assert shows(_read(character.reference_image_ref)) is expected, name
    assert all(c.has_approved_reference for c in synthetic_books["consistent"].characters)
    assert all(c.has_approved_reference for c in synthetic_books["two_characters"].characters)
    unapproved = synthetic_books["unapproved_reference"].characters
    assert unapproved and unapproved[0].reference_image_ref and not unapproved[0].has_approved_reference
    (missing,) = synthetic_books["no_reference"].characters
    assert missing.reference_image_ref == "" and not missing.has_approved_reference


def test_identity_is_what_a_page_was_conditioned_on_not_what_it_shows(synthetic_books):
    # Metadata must not give the answer away: a drifted page was still conditioned on Pip.
    for name, book in synthetic_books.items():
        if name == "two_characters":
            continue
        pip = book.characters[0].identity_key
        assert [page.identity for page in book.illustrations] == [pip] * 4, name
    book = synthetic_books["two_characters"]
    pip, zog = (c.identity_key for c in book.characters)
    assert [page.identity for page in book.illustrations] == [pip, zog, pip, zog]


def test_pages_of_one_character_are_posed_not_copied(synthetic_books):
    # A scorer must recognise the character, not match pixels; identical pages would let it cheat.
    pages = {_read(page.image_ref) for page in synthetic_books["consistent"].illustrations}
    reference = _read(synthetic_books["consistent"].characters[0].reference_image_ref)
    assert len(pages) == 4
    assert reference not in pages


def test_wrong_character_book_has_perfect_cohesion(synthetic_books, looks):
    # The failure page-to-page similarity cannot see: every page alike, all of the wrong character.
    book = synthetic_books["wrong_character"]
    assert len({_read(page.image_ref) for page in book.illustrations}) == 1
    assert book.page_looks == (looks[1],) * 4 and book.characters[0].label == "pip"
    assert book.expect_pass is False


def test_exactly_one_page_drifts_and_it_is_the_weakest(synthetic_books, looks):
    book = synthetic_books["one_page_drifts"]
    wrong = [i for i, look in enumerate(book.page_looks) if look is not looks[0]]
    assert wrong == [book.weakest_page] == [2]


def test_the_whole_verdict_table(synthetic_books):
    # Every verdict pinned, so flipping one is a failing test. The two None rows are filled in once the
    # team answers the KAN-105 question; a guess here would become the contract.
    assert {n: (b.expect_pass, b.weakest_page) for n, b in synthetic_books.items()} == VERDICTS


def test_rendering_is_deterministic(looks, render_look):
    pip, zog = looks
    assert render_look(pip, 80, 160, 0.8, "grass") == render_look(pip, 80, 160, 0.8, "grass")
    assert render_look(pip) != render_look(zog)


def test_refs_are_local_files_with_neutral_names(synthetic_books, tmp_path):
    for book in synthetic_books.values():
        refs = [page.image_ref for page in book.illustrations]
        refs += [c.reference_image_ref for c in book.characters if c.reference_image_ref]
        for ref in refs:
            path = _path(ref)
            assert path.parent == tmp_path
            assert len(path.stem) == 32 and "pip" not in path.stem and "zog" not in path.stem


def test_file_uris_with_encoded_characters_resolve(tmp_path):
    # A tmp or home directory can hold spaces or non-ASCII; `as_uri()` percent-encodes them.
    path = tmp_path / "a folder" / "pâge 1.png"
    path.parent.mkdir()
    path.write_bytes(b"page")
    ref = path.as_uri()
    assert "%20" in ref and _path(ref) == path and _read(ref) == b"page"


def test_consistency_still_raises_until_the_scorer_is_wired(synthetic_books, tmp_path):
    # The books are ready; the score is not. This flips when KAN-105 lands.
    model = IllustratorModel(tmp_path / "pages", pipeline=lambda request: None)
    book = synthetic_books["consistent"]
    with pytest.raises(NotImplementedError, match="stage 6"):
        model.consistency(book.illustrations, book.characters)
