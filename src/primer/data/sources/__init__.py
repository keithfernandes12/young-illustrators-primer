"""Registry of implemented sources, keyed as in configs/sources.yaml."""

from __future__ import annotations

from primer.data.config import SourceConfig
from primer.data.sources.base import Source
from primer.data.sources.tinystories import TinyStories
from primer.data.storage import Store

SOURCES: dict[str, type[Source]] = {
    TinyStories.key: TinyStories,
}


def get_source(config: SourceConfig, store: Store) -> Source:
    try:
        return SOURCES[config.key](config, store)
    except KeyError:
        raise KeyError(
            f"No pipeline implemented for source {config.key!r} yet. Implemented: {', '.join(SOURCES)}"
        ) from None
