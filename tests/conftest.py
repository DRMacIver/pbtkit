import os
import sys
import types

import pytest

# Poison disabled modules BEFORE importing minithesis, so that
# __init__.py's `import minithesis.floats` etc. finds the dummy
# and skips loading the real code.
assert "minithesis" not in sys.modules, (
    "minithesis was imported before conftest.py could set up module poisoning"
)

DISABLED_MODULES = frozenset(
    m for m in os.environ.get("MINITHESIS_DISABLED", "").split(",") if m
)

for _name in DISABLED_MODULES:
    _full = f"minithesis.{_name}"
    _dummy = types.ModuleType(_full)
    _dummy.DISABLED = True  # type: ignore[attr-defined]
    sys.modules[_full] = _dummy


def module_disabled(name: str) -> bool:
    """Check if a minithesis submodule has been disabled."""
    return name in DISABLED_MODULES


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
