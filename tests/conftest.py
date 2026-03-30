import os

import pytest

try:
    from minithesis.features import DISABLED_MODULES
except (ImportError, AttributeError):
    # Compiled mode: features module not available.
    # Read disabled modules from env var directly.
    DISABLED_MODULES = frozenset(
        m for m in os.environ.get("MINITHESIS_DISABLED", "").split(",") if m
    )

import minithesis.core as core


def pytest_configure(config):
    config.addinivalue_line("markers", "requires(module): skip if module is disabled")


def pytest_collection_modifyitems(items):
    if not DISABLED_MODULES:
        return
    for item in items:
        for mark in item.iter_markers("requires"):
            for mod in mark.args:
                if mod in DISABLED_MODULES:
                    item.add_marker(
                        pytest.mark.skip(reason=f"minithesis.{mod} is disabled")
                    )
                    break


@pytest.fixture(autouse=True)
def _isolate_database(tmp_path, monkeypatch):
    """Ensure each test gets a fresh default database directory
    so tests don't leak state via .minithesis-cache."""
    monkeypatch.setattr(core, "_DEFAULT_DATABASE_PATH", str(tmp_path / "cache"))
