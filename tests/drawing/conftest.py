"""Fixtures for the drawing tests. The recording encoder itself lives in `drawing_fakes.py`.

That name is deliberately not `fakes.py`: `tests/` has no `__init__.py`, so pytest's prepend
import mode puts this directory on `sys.path` and a bare `fakes` would collide with any other
area's fakes module the moment one is added.

Nothing here touches a checkpoint, the network or `PIL`. The "drawings" are arbitrary bytes: the
fake encoder never decodes them, and nothing in this slice does either beyond handing them to the
tower, so a real PNG would only buy a dependency CI does not install.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from drawing_fakes import TEST_LABELS

from primer.drawing.model import DrawingConfig


@pytest.fixture
def labels_file(tmp_path):
    """Writes a labels file in the real format: a comment header, a blank line, then one per line."""

    def write(labels: Sequence[str] = TEST_LABELS, name: str = "test_categories.txt"):
        path = tmp_path / name
        path.write_text("# a test vocabulary\n\n" + "\n".join(labels) + "\n", encoding="utf-8")
        return path

    return write


@pytest.fixture
def config(labels_file):
    """A `DrawingConfig` over `TEST_LABELS`, with the real prompt template and top_k.

    `config_dir` is the folder the labels file is in, which is how a config resolves its own
    vocabulary: the shipped config stays a self-contained folder rather than a set of absolute
    paths, and a test config does not have to reach into the module to redirect a global.
    """
    path = labels_file()
    return DrawingConfig(
        name="test-baseline",
        encoder="fake/encoder",
        encoder_revision="0" * 40,
        encoder_license="apache-2.0",
        labels_file=path.name,
        prompt_template="This is a photo of {label}.",
        text_max_length=64,
        text_padding="max_length",
        top_k=5,
        config_dir=path.parent,
    )


@pytest.fixture
def drawing() -> bytes:
    """A synthetic drawing. Never a child's: these bytes are not even an image."""
    return b"\x89PNG\r\n\x1a\n synthetic strokes"
