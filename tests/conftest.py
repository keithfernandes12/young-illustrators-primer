"""Applies to every test: nothing may reach the shared GCS bucket.

Tests use the fake fetcher and the temporary local `store` fixture (tests/data/conftest.py). This
makes that rule enforced instead of only written down: the pipeline's default data root points at
a temporary folder, and opening any gs:// filesystem raises.
"""

import gcsfs
import pytest


@pytest.fixture(autouse=True)
def no_gcs(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIMER_DATA_URI", (tmp_path / "data-root").as_posix())

    def refuse(*args, **kwargs):
        raise RuntimeError("Tests must not touch GCS. Use the `store` fixture (a temporary local folder).")

    gcsfs.GCSFileSystem.clear_instance_cache()
    monkeypatch.setattr(gcsfs.GCSFileSystem, "__init__", refuse)
