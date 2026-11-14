"""Stage 6, character consistency: a pretrained-embedder score for how alike a book's pages are.

``scorer`` is the request-path part and imports neither torch nor ``primer.data``. ``embedders`` loads
the pretrained weights lazily, and ``experiment`` is the offline run that picks the embedder and sets
the threshold. See ``CARD.md``.
"""

from primer.illustration.consistency.scorer import (
    ConsistencyResult,
    ConsistencyScorer,
    FakeEmbedder,
    ImageEmbedder,
    summarise,
)

__all__ = ["ConsistencyResult", "ConsistencyScorer", "FakeEmbedder", "ImageEmbedder", "summarise"]
