"""The autouse guard in tests/conftest.py really blocks GCS."""

import fsspec
import pytest

from primer.data.storage import Store


def test_opening_a_gcs_filesystem_fails():
    with pytest.raises(RuntimeError, match="must not touch GCS"):
        fsspec.filesystem("gs")


def test_a_store_on_the_bucket_cannot_be_used():
    with pytest.raises(RuntimeError, match="must not touch GCS"):
        Store("gs://young-illustrators-primer-data").exists("metadata/state/tinystories.json")
