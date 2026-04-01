"""Float shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/floats.rs.
"""

import math

import pytest

import minithesis.generators as gs

from .conftest import minimal

pytestmark = [pytest.mark.requires("floats"), pytest.mark.requires("collections")]


@pytest.mark.requires("shrinking.mutation")
def test_shrinks_to_simple_float_above_1():
    # Under the (exponent_rank, mantissa, sign) ordering, the simplest
    # float > 1.0 is the next representable float (same exponent, mantissa+1).
    result = minimal(gs.floats(allow_nan=False), lambda x: x > 1.0)
    assert result > 1.0
    assert result < 1.0 + 1e-10  # very close to 1.0


def test_shrinks_to_simple_float_above_0():
    assert minimal(gs.floats(allow_nan=False), lambda x: x > 0.0) == 1.0


@pytest.mark.parametrize("n", [1, 2, 3, 8, 10])
def test_can_shrink_in_variable_sized_context(n):
    x = minimal(
        gs.lists(gs.floats(allow_nan=False, allow_infinity=False), min_size=n),
        lambda x: any(f != 0.0 for f in x),
    )
    assert len(x) == n
    assert x.count(0.0) == n - 1
    assert 1.0 in x


def test_can_find_nan():
    x = minimal(gs.floats(), lambda x: math.isnan(x))
    assert math.isnan(x)


def test_can_find_nans():
    x = minimal(gs.lists(gs.floats()), lambda x: math.isnan(sum(x)))
    if len(x) == 1:
        assert math.isnan(x[0])
    else:
        assert 2 <= len(x) <= 3
