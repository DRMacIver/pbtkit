"""Composite generator shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/composite.rs.
"""

from random import Random

import pytest

from minithesis.core import MinithesisState, Status
from minithesis.generators import (
    binary,
    booleans,
    composite,
    floats,
    integers,
    lists,
    one_of,
    text,
    tuples,
)

from .conftest import minimal


@composite
def int_pair(tc, lo, hi):
    a = tc.any(integers(lo, hi))
    b = tc.any(integers(lo, hi))
    return (a, b)


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_positive_sum_of_pair():
    assert minimal(int_pair(0, 1000), lambda x: x[0] + x[1] > 1000) == (1, 1000)


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_negative_sum_of_pair():
    assert minimal(int_pair(-1000, 1000), lambda x: x[0] + x[1] < -1000) == (-1, -1000)


@pytest.mark.requires("text")
@composite
def separated_sum(tc):
    n1 = tc.any(integers(0, 1000))
    tc.any(text())
    tc.any(booleans())
    tc.any(integers(-(2**63), 2**63 - 1))
    n2 = tc.any(integers(0, 1000))
    return (n1, n2)


@pytest.mark.requires("text")
@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_sum_of_pair_separated():
    assert minimal(separated_sum(), lambda x: x[0] + x[1] > 1000) == (1, 1000)


def test_minimize_dict_of_booleans():
    result = minimal(tuples(booleans(), booleans()), lambda x: x[0] or x[1])
    assert not (result[0] and result[1])
    assert result[0] or result[1]


@composite
def int_struct(tc):
    a = tc.any(integers(-(2**63), 2**63 - 1))
    b = tc.any(integers(-(2**63), 2**63 - 1))
    return (a, b)


def test_minimize_namedtuple():
    tab = minimal(int_struct(), lambda x: x[0] < x[1])
    assert tab[1] == tab[0] + 1


def test_earlier_exit_produces_shorter_sequence():
    """When v0=True triggers an early exit with fewer choices than
    v0=False followed by more draws, the shrinker should prefer the
    shorter path. Found by shrink comparison test."""

    @composite
    def pair_of_bools(tc):
        a = tc.any(booleans())
        b = tc.any(booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.any(booleans())
        v1 = tc.any(pair_of_bools())
        v2 = tc.any(pair_of_bools())
        # First exit: when v0=True (5 choices used)
        if v0:
            tc.mark_status(Status.INTERESTING)
        # More draws only reached when v0=False
        tc.any(booleans())
        # Second exit: len(v1) is always 2, so this always fires (6 choices)
        if len(v1) != 0:
            tc.mark_status(Status.INTERESTING)

    state = MinithesisState(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    # v0=True exits after 5 choices; v0=False needs 6.
    # The shrinker should find the shorter 5-choice path.
    vals = [n.value for n in state.result]
    assert len(vals) == 5
    assert vals[0]  # v0 is truthy (triggers early exit)


def test_one_of_shrinks_branch_selector():
    """one_of should shrink toward branch 0 even when a higher branch
    also produces an interesting result. Found by shrink comparison test."""
    result = minimal(
        one_of(booleans(), floats(allow_nan=False, allow_infinity=False)),
        lambda v: bool(v),
    )
    # Branch 0 (booleans) with value True is simpler than
    # branch 1 (floats) with any truthy float.
    assert result is True


def test_early_exit_via_flag_with_preceding_draws():
    """When v0=True triggers an early exit but v1 and v2 are drawn
    BEFORE the v0 check, the shrinker must coordinate: shorten v1/v2
    AND flip v0 simultaneously. Hypothesis finds [True, b'', False]
    (3 choices) but minithesis gets stuck at 5 choices with v0=False.

    Found by the Hypothesis shrink comparison test."""

    @composite
    def test_data(tc):
        v0 = tc.any(booleans())
        v1 = tc.any(binary(max_size=20))
        v2 = tc.any(lists(integers(0, 0), max_size=10))
        return (v0, v1, v2)

    result = minimal(
        test_data(),
        lambda t: t[0] or len(t[1]) + len(t[2]) >= 20,
    )
    # Optimal: v0=True with minimal v1 and v2 → 3 choices total.
    # Suboptimal: v0=False with long v1/v2 → 5+ choices.
    assert result[0] is True


def test_one_of_branch_switch_with_trailing_draws():
    """When one_of(booleans(), floats()) at branch=1 produces a truthy
    float, switching to branch=0 (boolean True) is simpler but requires
    fixing the downstream kind AND keeping the trailing composite draws.

    Found by the Hypothesis shrink comparison test."""

    @composite
    def gen_pair(tc):
        a = tc.any(booleans())
        b = tc.any(booleans())
        return (a, b)

    @composite
    def test_data(tc):
        v0 = tc.any(one_of(booleans(), floats(allow_nan=False, allow_infinity=False)))
        tc.any(gen_pair())
        return v0

    result = minimal(test_data(), lambda v: bool(v))
    assert result is True


def test_shorter_path_via_later_assertion():
    """When emptying a list removes one failure but a later assertion
    still fails with fewer total choices, the shrinker should prefer
    the shorter path. Found by the Hypothesis shrink comparison test."""

    @composite
    def pair(tc):
        a = tc.any(booleans())
        b = tc.any(floats(allow_nan=False, allow_infinity=False))
        return (a, b)

    @composite
    def test_data(tc):
        v0 = tc.any(pair())
        v1 = tc.any(lists(integers(0, 20), max_size=10, unique=True))
        tc.any(pair())
        return (v0, v1)

    result = minimal(
        test_data(),
        # Fails when v1 is non-empty (7 choices) OR always via len(v0)==2
        # (v0 is a tuple, 6 choices). The shorter path is v1 empty.
        lambda t: len(t[1]) > 0 or len(t[0]) != 0,
    )
    assert len(result[1]) == 0


def test_one_of_branch_switch_to_float():
    """When one_of(floats(), booleans()) starts at branch=1 (booleans),
    switching to branch=0 (floats) is simpler but requires replacing
    the boolean value with a valid float. Found by shrink comparison test."""
    result = minimal(
        one_of(floats(allow_nan=False, allow_infinity=False), booleans()),
        lambda _: True,
    )
    # Branch 0 (floats) with simplest value 0.0 is simpler than
    # branch 1 (booleans) with simplest value False.
    assert isinstance(result, float)
    assert result == 0.0


def test_one_of_shorter_branch_needs_non_simplest_value():
    """When one_of(tuples(booleans(), booleans()), booleans()) starts at
    branch=0 (tuple, 3 choices), branch=1 with True (2 choices) is shorter.
    But the increment + pun produces branch=1 with False (not interesting).
    Found by the Hypothesis shrink comparison test."""
    result = minimal(
        one_of(tuples(booleans(), booleans()), booleans()),
        lambda v: bool(v),
    )
    assert result is True


def test_shorter_path_when_guard_precedes_expensive_draw():
    """A guard check (v0 > 0) comes after a cheap draw but before an
    expensive draw (a list). When v0=1 triggers the guard, the expensive
    draw is skipped, producing a shorter sequence. But the shrinker
    reduces v0 to 0 (simplest) first, which forces the longer path.

    Found by the Hypothesis shrink comparison test."""

    @composite
    def test_data(tc):
        v0 = tc.any(integers(0, 10))
        v1 = tc.any(booleans())
        v2 = tc.any(lists(integers(0, 100), max_size=10))
        return (v0, v1, v2)

    result = minimal(
        test_data(),
        lambda t: t[0] > 0 or len(t[2]) >= 3,
    )
    # Optimal: v0=1 exits early → 3 choices (v0, v1, list_stop).
    # Suboptimal: v0=0 needs len(v2) >= 3 → 8+ choices.
    assert result[0] > 0
