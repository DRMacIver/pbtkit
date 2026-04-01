import sys

import pytest

pbtkit_features = pytest.importorskip(
    "pbtkit.features", reason="not available in compiled mode"
)
_DisabledModule = pbtkit_features._DisabledModule
_DisabledSymbol = pbtkit_features._DisabledSymbol
disable_modules = pbtkit_features.disable_modules


def test_disabled_symbol_raises():
    sym = _DisabledSymbol("floats", "draw_float")
    with pytest.raises(NotImplementedError, match="not available"):
        sym()


def test_disabled_symbol_repr():
    sym = _DisabledSymbol("floats", "draw_float")
    assert repr(sym) == "<disabled: pbtkit.floats.draw_float>"


def test_disabled_module_returns_symbols():
    mod = _DisabledModule("floats", "pbtkit.floats")
    assert mod.DISABLED is True
    sym = mod.draw_float
    assert isinstance(sym, _DisabledSymbol)


def test_disabled_module_raises_for_dunders():
    mod = _DisabledModule("floats", "pbtkit.floats")
    with pytest.raises(AttributeError):
        mod.__iter__


def test_disable_modules():
    disable_modules(frozenset(["_test_fake"]))
    full = "pbtkit._test_fake"
    assert full in sys.modules
    assert isinstance(sys.modules[full], _DisabledModule)
    del sys.modules[full]


def test_disable_dotted_modules():
    disable_modules(frozenset(["_test_pkg._test_sub"]))
    assert "pbtkit._test_pkg._test_sub" in sys.modules
    assert isinstance(sys.modules["pbtkit._test_pkg._test_sub"], _DisabledModule)
    # Parent is NOT created — only the leaf is poisoned.
    assert "pbtkit._test_pkg" not in sys.modules
    del sys.modules["pbtkit._test_pkg._test_sub"]
