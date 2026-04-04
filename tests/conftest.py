import os

import pytest

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

try:
    from pbtkit.features import DISABLED_MODULES
except (ImportError, AttributeError):
    # Compiled mode: features module not available.
    # Read disabled modules from env var directly.
    DISABLED_MODULES = frozenset(
        m for m in os.environ.get("PBTKIT_DISABLED", "").split(",") if m
    )

try:
    import pbtkit.database as _db_mod
except (ImportError, AttributeError):
    _db_mod = None


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
                        pytest.mark.skip(reason=f"pbtkit.{mod} is disabled")
                    )
                    break


@pytest.fixture(autouse=True)
def _isolate_database(tmp_path, monkeypatch):
    """Ensure each test gets a fresh default database directory
    so tests don't leak state via .pbtkit-cache."""
    if _db_mod is not None and "database" not in DISABLED_MODULES:
        monkeypatch.setattr(_db_mod, "_DEFAULT_DATABASE_PATH", str(tmp_path / "cache"))
