"""Integer shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/integers.rs.
"""

import pytest

from minithesis.generators import composite, integers, lists

from .conftest import minimal

pytestmark = pytest.mark.requires("collections")


def test_integers_from_minimizes_leftwards():
    assert minimal(integers(101, 2**63)) == 101


def test_minimize_bounded_integers_to_zero():
    assert minimal(integers(-10, 10)) == 0


def test_minimize_bounded_integers_to_positive():
    assert minimal(integers(-10, 10), lambda x: x != 0) == 1


def test_minimize_single_element_in_silly_large_int_range():
    hi = 2**63 - 1
    lo = -(2**63)
    result = minimal(integers(lo // 2, hi // 2), lambda x: x >= lo // 4)
    assert result == 0


def test_minimize_multiple_elements_in_silly_large_int_range():
    hi = 2**63 - 1
    lo = -(2**63)
    result = minimal(
        lists(integers(lo // 2, hi // 2)),
        lambda x: len(x) >= 20,
        max_examples=10000,
    )
    assert result == [0] * 20


def test_minimize_multiple_elements_min_is_not_dupe():
    @composite
    def bounded_int_list(tc):
        return tc.any(lists(integers(0, 2**62)))

    target = list(range(20))
    result = minimal(
        bounded_int_list(),
        lambda x: len(x) >= 20 and all(x[i] >= target[i] for i in range(20)),
        max_examples=10000,
    )
    assert result == list(range(20))


def test_can_find_an_int():
    assert minimal(integers(-(2**63), 2**63 - 1)) == 0


def test_can_find_an_int_above_13():
    assert minimal(integers(-(2**63), 2**63 - 1), lambda x: x >= 13) == 13


def test_minimizes_towards_zero():
    assert minimal(integers(-1000, 50), lambda x: x < 0) == -1
