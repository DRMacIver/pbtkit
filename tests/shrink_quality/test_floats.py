"""Float shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/floats.rs.
"""

import math

import pytest

from minithesis.generators import floats, lists

from .conftest import minimal

pytestmark = [pytest.mark.requires("floats"), pytest.mark.requires("collections")]


def test_shrinks_to_simple_float_above_1():
    assert minimal(floats(allow_nan=False), lambda x: x > 1.0) == 2.0


def test_shrinks_to_simple_float_above_0():
    assert minimal(floats(allow_nan=False), lambda x: x > 0.0) == 1.0


_xfail_list_context = pytest.mark.xfail(
    reason="float shrinker doesn't simplify within list context"
)


@pytest.mark.parametrize(
    "n",
    [
        1,
        pytest.param(2, marks=_xfail_list_context),
        pytest.param(3, marks=_xfail_list_context),
        pytest.param(8, marks=_xfail_list_context),
        pytest.param(10, marks=_xfail_list_context),
    ],
)
def test_can_shrink_in_variable_sized_context(n):
    x = minimal(
        lists(floats(allow_nan=False, allow_infinity=False), min_size=n),
        lambda x: any(f != 0.0 for f in x),
    )
    assert len(x) == n
    assert x.count(0.0) == n - 1
    assert 1.0 in x


def test_can_find_nan():
    x = minimal(floats(), lambda x: math.isnan(x))
    assert math.isnan(x)


def test_can_find_nans():
    x = minimal(lists(floats()), lambda x: math.isnan(sum(x)))
    if len(x) == 1:
        assert math.isnan(x[0])
    else:
        assert 2 <= len(x) <= 3
