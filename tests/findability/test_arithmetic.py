"""Arithmetic findability tests.

Ported from Hypothesis's test_testdecorators.py (@fails tests).
These verify the engine can find counterexamples to false
arithmetic properties.
"""

import pytest

import pbtkit.generators as gs
from pbtkit import run_test

pytestmark = pytest.mark.requires("collections")


def test_float_addition_is_not_associative():
    """Engine can find floats where (x + y) + z != x + (y + z)."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=2000)
        def _(tc):
            x = tc.draw(gs.floats())
            y = tc.draw(gs.floats())
            z = tc.draw(gs.floats())
            assert x + (y + z) == (x + y) + z


def test_float_addition_does_not_cancel():
    """Engine can find floats where x + (y - x) != y."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=2000)
        def _(tc):
            x = tc.draw(gs.floats())
            y = tc.draw(gs.floats())
            assert x + (y - x) == y


@pytest.mark.requires("text")
def test_string_addition_is_not_commutative():
    """Engine can find non-empty strings where x + y != y + x."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x = tc.draw(gs.text(min_size=1))
            y = tc.draw(gs.text(min_size=1))
            assert x + y == y + x


@pytest.mark.requires("bytes")
def test_bytes_addition_is_not_commutative():
    """Engine can find non-empty byte strings where x + y != y + x."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x = tc.draw(gs.binary(min_size=1))
            y = tc.draw(gs.binary(min_size=1))
            assert x + y == y + x


def test_integer_bound_can_be_exceeded():
    """Engine can find an integer >= t for various thresholds."""
    for t in [1, 10, 100, 1000]:
        with pytest.raises(AssertionError):

            @run_test(database={}, max_examples=10000)
            def _(tc):
                x = tc.draw(gs.integers(-(2**63), 2**63 - 1))
                assert x < t


def test_int_is_not_always_negative():
    """Engine can find a non-negative integer."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x = tc.draw(gs.integers(-(2**63), 2**63 - 1))
            assert x < 0


def test_int_addition_is_commutative():
    """This property is TRUE — verify it doesn't spuriously fail."""

    @run_test(database={})
    def _(tc):
        x = tc.draw(gs.integers(-(2**63), 2**63 - 1))
        y = tc.draw(gs.integers(-(2**63), 2**63 - 1))
        assert x + y == y + x


def test_int_addition_is_associative():
    """This property is TRUE — verify it doesn't spuriously fail."""

    @run_test(database={})
    def _(tc):
        x = tc.draw(gs.integers(-(2**63), 2**63 - 1))
        y = tc.draw(gs.integers(-(2**63), 2**63 - 1))
        z = tc.draw(gs.integers(-(2**63), 2**63 - 1))
        assert x + (y + z) == (x + y) + z


def test_reversing_preserves_integer_addition():
    """This property is TRUE — verify it doesn't spuriously fail."""

    @run_test(database={})
    def _(tc):
        xs = tc.draw(gs.lists(gs.integers(-(2**63), 2**63 - 1)))
        assert sum(xs) == sum(reversed(xs))


def test_integer_division_preserves_order():
    """This property is TRUE for positive integers: n/2 < n."""

    @run_test(database={})
    def _(tc):
        n = tc.draw(gs.integers(1, 2**63 - 1))
        assert n / 2 < n
