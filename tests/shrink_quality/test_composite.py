"""Composite generator shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/composite.rs.
"""

from random import Random

import pytest

from minithesis.core import MinithesisState, Status
from minithesis.generators import booleans, composite, integers, text, tuples

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
