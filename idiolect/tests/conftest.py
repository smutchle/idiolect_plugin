import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = ROOT / "tests" / "fixtures"
CORPORA = FIXTURES / "corpora"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Never write to the real ~/.idiolect during tests."""
    monkeypatch.setenv("IDIOLECT_HOME", str(tmp_path / "idhome"))
    yield
