"""The real readability implementation. Weights load lazily, never at import time.

Importing this module must stay cheap: the serving layer imports every area, and a test that
imports it must not pull a checkpoint off disk or out of GCS.
"""

from __future__ import annotations


class ReadabilityScorerModel:
    """The production readability implementation. Will satisfy ``interface.ReadabilityScorer``."""

    def __init__(self, model_uri: str | None = None) -> None:
        self._model_uri = model_uri
        self._loaded: object | None = None  # populated on first use, not here

    def _load(self) -> object:
        raise NotImplementedError("readability/model.py is a stub")
