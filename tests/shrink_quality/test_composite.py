"""Composite generator shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/composite.rs.
"""

import pytest

from minithesis.generators import booleans, composite, integers, text, tuples

from .conftest import minimal


@composite
def int_pair(tc, lo, hi):
    a = tc.any(integers(lo, hi))
    b = tc.any(integers(lo, hi))
    return (a, b)


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_sum_of_pair():
    assert minimal(int_pair(0, 1000), lambda x: x[0] + x[1] > 1000) == (1, 1000)


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
