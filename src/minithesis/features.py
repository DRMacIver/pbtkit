"""Feature management for minithesis.

Reads the MINITHESIS_DISABLED environment variable and prevents
disabled extension modules from loading, replacing them with dummy
modules that raise NotImplementedError when any symbol is called.

This module must be imported before the extension modules (floats,
bytes, text, collections, targeting).
"""

from __future__ import annotations

import os
import sys
import types


DISABLED_MODULES: frozenset[str] = frozenset(
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


def disable_modules(modules: frozenset[str]) -> None:
    """Poison sys.modules entries for the given module names."""
    for name in modules:
        full = f"minithesis.{name}"
        sys.modules[full] = _DisabledModule(name, full)


disable_modules(DISABLED_MODULES)
