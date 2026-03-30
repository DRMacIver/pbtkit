import sys

import pytest

minithesis_features = pytest.importorskip(
    "minithesis.features", reason="not available in compiled mode"
)
_DisabledModule = minithesis_features._DisabledModule
_DisabledSymbol = minithesis_features._DisabledSymbol
disable_modules = minithesis_features.disable_modules


def test_disabled_symbol_raises():
    sym = _DisabledSymbol("floats", "draw_float")
    with pytest.raises(NotImplementedError, match="not available"):
        sym()


def test_disabled_symbol_repr():
    sym = _DisabledSymbol("floats", "draw_float")
    assert repr(sym) == "<disabled: minithesis.floats.draw_float>"


def test_disabled_module_returns_symbols():
    mod = _DisabledModule("floats", "minithesis.floats")
    assert mod.DISABLED is True
    sym = mod.draw_float
    assert isinstance(sym, _DisabledSymbol)


def test_disabled_module_raises_for_dunders():
    mod = _DisabledModule("floats", "minithesis.floats")
    with pytest.raises(AttributeError):
        mod.__iter__


def test_disable_modules():
    disable_modules(frozenset(["_test_fake"]))
    full = "minithesis._test_fake"
    assert full in sys.modules
    assert isinstance(sys.modules[full], _DisabledModule)
    del sys.modules[full]
