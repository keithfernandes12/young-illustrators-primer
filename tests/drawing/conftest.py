"""Fixtures for the drawing tests. The recording encoder itself lives in `fakes.py`.

Nothing here touches a checkpoint, the network or `PIL`. The "drawings" are arbitrary bytes: the
fake encoder never decodes them, and nothing in this slice does either beyond handing them to the
tower, so a real PNG would only buy a dependency CI does not install.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from fakes import TEST_LABELS

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
def config(labels_file, monkeypatch):
    """A `DrawingConfig` over `TEST_LABELS`, with the real prompt template and top_k.

    `CONFIG_DIR` is redirected because `labels_file` is resolved relative to it, which is what
    keeps the shipped config a single self-contained folder rather than a set of absolute paths.
    """
    path = labels_file()
    monkeypatch.setattr("primer.drawing.model.CONFIG_DIR", path.parent)
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
    )


@pytest.fixture
def drawing() -> bytes:
    """A synthetic drawing. Never a child's: these bytes are not even an image."""
    return b"\x89PNG\r\n\x1a\n synthetic strokes"
