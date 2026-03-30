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


def test_disable_dotted_modules():
    disable_modules(frozenset(["_test_pkg._test_sub"]))
    assert "minithesis._test_pkg" in sys.modules
    assert isinstance(sys.modules["minithesis._test_pkg"], _DisabledModule)
    assert "minithesis._test_pkg._test_sub" in sys.modules
    assert isinstance(sys.modules["minithesis._test_pkg._test_sub"], _DisabledModule)
    # Disabling a second child under the same parent should not
    # overwrite the existing parent entry.
    disable_modules(frozenset(["_test_pkg._test_other"]))
    assert "minithesis._test_pkg._test_other" in sys.modules
    del sys.modules["minithesis._test_pkg._test_other"]
    del sys.modules["minithesis._test_pkg._test_sub"]
    del sys.modules["minithesis._test_pkg"]
