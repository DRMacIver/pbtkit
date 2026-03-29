import os
import sys
import types

import pytest

# Poison disabled modules BEFORE importing minithesis, so that
# __init__.py's `import minithesis.floats` etc. finds the dummy
# and skips loading the real code.  Importing any symbol from a
# disabled module succeeds, but calling the symbol raises
# NotImplementedError.
assert "minithesis" not in sys.modules, (
    "minithesis was imported before conftest.py could set up module poisoning"
)

DISABLED_MODULES = frozenset(
    m for m in os.environ.get("MINITHESIS_DISABLED", "").split(",") if m
)


class _DisabledSymbol:
    """Placeholder for a symbol imported from a disabled module.
    Raises NotImplementedError when called."""

    def __init__(self, module_name: str, symbol_name: str):
        self._module_name = module_name
        self._symbol_name = symbol_name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self._module_name}.{self._symbol_name} is not available:"
            f" minithesis.{self._module_name} is disabled"
        )

    def __repr__(self):
        return f"<disabled: minithesis.{self._module_name}.{self._symbol_name}>"


class _DisabledModule(types.ModuleType):
    """A dummy module whose attributes are all _DisabledSymbol instances."""

    def __init__(self, module_name: str, full_name: str):
        super().__init__(full_name)
        self._module_name = module_name
        self.DISABLED = True

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _DisabledSymbol(self._module_name, name)


for _name in DISABLED_MODULES:
    _full = f"minithesis.{_name}"
    sys.modules[_full] = _DisabledModule(_name, _full)

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
