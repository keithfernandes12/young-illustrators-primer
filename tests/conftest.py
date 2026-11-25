"""Applies to every test: nothing may reach the shared GCS bucket.

Tests use the fake fetcher and the temporary local `store` fixture (tests/data/conftest.py). This
makes that rule enforced instead of only written down: the pipeline's default data root points at
a temporary folder, and opening any gs:// filesystem raises.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import gcsfs
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def load_script():
    """Import a file that is not part of the package: a script, or a hook with no .py extension.

    An explicit loader is needed because git hooks have no .py extension. The module is registered
    in ``sys.modules`` before it runs, because a dataclass looks its module up there while it is
    being defined.
    """

    def load(relative_path: str, name: str):
        path = str(REPO_ROOT / relative_path)
        loader = importlib.machinery.SourceFileLoader(name, path)
        spec = importlib.util.spec_from_file_location(name, path, loader=loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module

    return load


@pytest.fixture(autouse=True)
def no_gcs(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIMER_DATA_URI", (tmp_path / "data-root").as_posix())

    def refuse(*args, **kwargs):
        raise RuntimeError("Tests must not touch GCS. Use the `store` fixture (a temporary local folder).")

    gcsfs.GCSFileSystem.clear_instance_cache()
    monkeypatch.setattr(gcsfs.GCSFileSystem, "__init__", refuse)
