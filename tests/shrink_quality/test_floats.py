"""Float shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/floats.rs.
"""

import math
from random import Random

import pytest

import minithesis.generators as gs
from minithesis.core import MinithesisState as State
from minithesis.core import Status

from .conftest import minimal

pytestmark = pytest.mark.requires("floats")


@pytest.mark.requires("shrinking.mutation")
def test_shrinks_to_simple_float_above_1():
    # Under the (exponent_rank, mantissa, sign) ordering, the simplest
    # float > 1.0 is the next representable float (same exponent, mantissa+1).
    result = minimal(gs.floats(allow_nan=False), lambda x: x > 1.0)
    assert result > 1.0
    assert result < 1.0 + 1e-10  # very close to 1.0


def test_shrinks_to_simple_float_above_0():
    assert minimal(gs.floats(allow_nan=False), lambda x: x > 0.0) == 1.0


@pytest.mark.requires("collections")
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


@pytest.mark.requires("collections")
def test_can_find_nans():
    x = minimal(gs.lists(gs.floats()), lambda x: math.isnan(sum(x)))
    if len(x) == 1:
        assert math.isnan(x[0])
    else:
        assert 2 <= len(x) <= 3


@pytest.mark.requires("shrinking.index_passes")
def test_float_increment_shortens_via_negative():
    """Making a float negative can trigger an earlier check and shorten
    the overall choice sequence. try_shortening_via_increment should
    try negative float values.
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.booleans())
        v1 = tc.any(gs.floats(allow_nan=False, allow_infinity=False))
        tc.any(gs.booleans())
        if v1 < 0.0:
            tc.mark_status(Status.INTERESTING)
        tc.any(gs.booleans())
        if v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    assert len(state.result) == 3


@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_with_float_source_gaps():
    """lower_and_bump must handle from_index returning None when the
    float source has index gaps (bounded range with interleaved signs)."""

    def tf(tc):
        v0 = tc.any(gs.floats(min_value=1.0, max_value=2.0, allow_nan=False))
        v1 = tc.any(gs.booleans())
        if v0 > 1.5 and v1:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None


@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_with_bounded_float_target():
    """lower_and_bump must skip invalid float bump values when the
    float's range doesn't include 1.0 or -1.0."""

    def tf(tc):
        v0 = tc.any(gs.integers(0, 5))
        v1 = tc.any(gs.floats(min_value=0.0, max_value=0.5, allow_nan=False))
        if v0 >= 3 and v1 > 0.0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None


@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_negative_zero_decrement_target():
    """lower_and_bump skips -0.0 when 0.0 is already a decrement target.

    For a float with to_index=2 (e.g. 1.0), from_index(0)=0.0 is added to
    decrement_targets. Then from_index(1)=-0.0 satisfies -0.0 in [0.0]
    (Python equality), covering the 'case v if v in decrement_targets' branch.

    The float is placed before the integer so it is the source node
    processed by lower_and_bump (gap=1 requires at least two indexed nodes)."""

    def tf(tc):
        v = tc.any(gs.floats(allow_nan=False, allow_infinity=False))
        a = tc.draw_integer(0, 10)
        if v > 0.5 and a > 0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    assert state.result[0].value == 1.0


@pytest.mark.requires("collections")
def test_negative_zero_shrinks_to_positive_zero():
    """The shrinker should prefer 0.0 over -0.0 since sort_key(0.0) <
    sort_key(-0.0). The cache must distinguish them despite 0.0 == -0.0
    in Python.
    Regression for shrink quality found by minismith."""

    @gs.composite
    def pair(tc):
        a = tc.any(gs.booleans())
        b = tc.any(gs.booleans())
        return (a, b)

    def tf(tc):
        tc.any(pair())
        tc.any(pair())
        v2 = tc.any(
            gs.one_of(
                gs.floats(allow_nan=False, allow_infinity=False),
                gs.floats(allow_nan=False, allow_infinity=False),
                gs.nothing(),
            )
        )
        tc.any(gs.booleans())
        v4 = tc.any(gs.booleans())
        if not (((v4) or (v2 > 0.0)) and (v2 >= 0.0)):
            tc.mark_status(Status.INTERESTING)

    state = State(Random(120), tf, 100)
    state.run()
    assert state.result is not None
    float_val = state.result[5].value
    assert isinstance(float_val, float)
    assert math.copysign(1.0, float_val) == 1.0, f"Expected 0.0 but got {float_val!r}"
