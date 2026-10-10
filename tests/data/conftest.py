import importlib.util
from pathlib import Path

import pytest

from primer.data.config import SourceConfig
from primer.data.storage import Store


@pytest.fixture
def store(tmp_path):
    """A Store on a temporary local folder, so tests never touch GCS."""
    return Store(tmp_path.as_posix())


@pytest.fixture
def tinystories_config():
    return SourceConfig(
        key="tinystories",
        name="TinyStories",
        license="CDLA-Sharing-1.0",
        attribution="TinyStories, Eldan & Li (2023)",
        ingest={"hf_dataset": "roneneldan/TinyStories", "batch_size": 4, "min_words": 3, "max_words": 50},
    )


@pytest.fixture
def dashboard_loaders():
    """The tracker dashboard's loaders module, to check it reads what the pipeline writes."""
    path = Path(__file__).resolve().parents[2] / "dashboards" / "tracker" / "loaders.py"
    spec = importlib.util.spec_from_file_location("tracker_loaders", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
