"""Feature management for pbtkit.

Reads the PBTKIT_DISABLED environment variable and prevents
disabled extension modules from loading, replacing them with dummy
modules that raise NotImplementedError when any symbol is called.

This module must be imported before the extension modules (floats,
bytes, text, collections, targeting).
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

DISABLED_MODULES: frozenset[str] = frozenset(
    m for m in os.environ.get("PBTKIT_DISABLED", "").split(",") if m
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
            f" pbtkit.{self._module_name} is disabled"
        )

    def __repr__(self):
        return f"<disabled: pbtkit.{self._module_name}.{self._symbol_name}>"


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
        full = f"pbtkit.{name}"
        sys.modules[full] = _DisabledModule(name, full)


def needed_for(feature: str) -> Any:
    """Mark a function or method as needed for a specific feature.

    This is a no-op at runtime — it returns the decorated function
    unchanged. The single-file compiler uses this annotation to
    strip decorated items when the feature is disabled."""

    def decorator(fn: Any) -> Any:
        return fn

    return decorator


def feature_enabled(feature: str) -> bool:
    """Return True if *feature* is not disabled.

    At runtime this always returns True for non-disabled features.
    In compiled single-file output, ``if feature_enabled("X"):``
    blocks (with a ``# needed_for("X")`` comment) are stripped
    entirely when X is disabled."""
    return feature not in DISABLED_MODULES


disable_modules(DISABLED_MODULES)
