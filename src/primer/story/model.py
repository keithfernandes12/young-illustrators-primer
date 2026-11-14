"""The real story implementation. Weights load lazily, never at import time.

Importing this module must stay cheap: the serving layer imports every area, and a test that
imports it must not pull a checkpoint off disk or out of GCS.
"""

from __future__ import annotations


class StoryWriterModel:
    """The production story implementation. Will satisfy ``interface.StoryWriter``."""

    def __init__(self, model_uri: str | None = None) -> None:
        self._model_uri = model_uri
        self._loaded: object | None = None  # populated on first use, not here

    def _load(self) -> object:
        raise NotImplementedError("story/model.py is a stub")
