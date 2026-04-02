"""Float findability tests.

Ported from Hypothesis's nocover/test_floating.py (@fails tests) and
cover/test_float_nastiness.py (find_any tests).
These verify the engine can find various categories of interesting
floating-point values.
"""

import math
import sys

import pytest

import pbtkit.generators as gs
from pbtkit import run_test

from .conftest import finds

pytestmark = pytest.mark.requires("floats")


def test_inversion_is_imperfect():
    """Engine can find a float where x * (1/x) != 1."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            x = tc.draw(gs.floats())
            if x == 0.0:
                return
            y = 1.0 / x
            assert x * y == 1.0


@pytest.mark.requires("collections")
def test_can_find_nan_in_list():
    """Engine can find a list of floats containing NaN."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            xs = tc.draw(gs.lists(gs.floats()))
            assert not any(math.isnan(x) for x in xs)


@pytest.mark.requires("edge_case_boosting")
def test_can_find_positive_infinity():
    """Engine can find a positive infinite float."""
    finds(gs.floats(), lambda x: x > 0 and math.isinf(x))


@pytest.mark.requires("edge_case_boosting")
def test_can_find_negative_infinity():
    """Engine can find a negative infinite float."""
    finds(gs.floats(), lambda x: x < 0 and math.isinf(x))


def test_can_find_non_integer_float():
    """Engine can find a finite float that is not an integer."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            x = tc.draw(gs.floats(allow_nan=False, allow_infinity=False))
            assert x == int(x)


def test_can_find_integer_float():
    """Engine can find a finite float that IS an integer (but not trivially 0)."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            x = tc.draw(gs.floats(allow_nan=False, allow_infinity=False))
            assert x != int(x)


def test_can_find_float_outside_exact_int_range():
    """Engine can find a finite float so large that x + 1 == x."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            x = tc.draw(gs.floats(allow_nan=False, allow_infinity=False))
            assert x + 1 != x


def test_can_find_float_that_does_not_round_trip_through_str():
    """Engine can find a float where float(str(x)) != x."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            x = tc.draw(gs.floats())
            assert float(str(x)) == x


def test_can_find_float_that_does_not_round_trip_through_repr():
    """Engine can find a float where float(repr(x)) != x."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            x = tc.draw(gs.floats())
            assert float(repr(x)) == x


@pytest.mark.requires("edge_case_boosting")
def test_half_bounded_generates_zero():
    """Engine can find zero from half-bounded float ranges."""
    finds(gs.floats(-1.0, allow_nan=False), lambda x: x == 0.0)
    finds(gs.floats(max_value=1.0, allow_nan=False), lambda x: x == 0.0)


# --- True properties that should NOT be falsified ---


def test_is_float():
    """All drawn floats are actually float instances."""

    @run_test(database={}, max_examples=1000)
    def _(tc):
        x = tc.draw(gs.floats())
        assert isinstance(x, float)


def test_negation_is_self_inverse():
    """For non-NaN floats, -(-x) == x."""

    @run_test(database={}, max_examples=1000)
    def _(tc):
        x = tc.draw(gs.floats(allow_nan=False))
        y = -x
        assert -y == x


def test_largest_range_has_no_infinities():
    """Floats bounded by float_info.max contain no infinities."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        x = tc.draw(gs.floats(-sys.float_info.max, sys.float_info.max, allow_nan=False))
        assert not math.isinf(x)
