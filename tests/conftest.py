import pytest

import minithesis.core as core


@pytest.fixture(autouse=True)
def _isolate_database(tmp_path, monkeypatch):
    """Ensure each test gets a fresh default database directory
    so tests don't leak state via .minithesis-cache."""
    monkeypatch.setattr(core, "_DEFAULT_DATABASE_PATH", str(tmp_path / "cache"))
