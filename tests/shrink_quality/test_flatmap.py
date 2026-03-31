"""Flatmap shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/flatmap.rs.
"""

import pytest

from minithesis.generators import booleans, integers, just, lists, sampled_from

from .conftest import minimal

pytestmark = pytest.mark.requires("collections")


def test_can_simplify_flatmap_with_bounded_left_hand_size():
    assert (
        minimal(
            booleans().flat_map(lambda x: lists(just(x))),
            lambda x: len(x) >= 10,
        )
        == [False] * 10
    )


def test_can_simplify_across_flatmap_of_just():
    assert minimal(integers(-(2**63), 2**63 - 1).flat_map(just)) == 0


def test_can_simplify_on_right_hand_strategy_of_flatmap():
    result = minimal(
        integers(-(2**63), 2**63 - 1).flat_map(lambda x: lists(just(x))),
    )
    assert result == []


def test_can_ignore_left_hand_side_of_flatmap():
    assert (
        minimal(
            integers(-(2**63), 2**63 - 1).flat_map(
                lambda _: lists(integers(-(2**63), 2**63 - 1))
            ),
            lambda x: len(x) >= 10,
        )
        == [0] * 10
    )


def test_can_simplify_on_both_sides_of_flatmap():
    assert (
        minimal(
            integers(-(2**63), 2**63 - 1).flat_map(lambda x: lists(just(x))),
            lambda x: len(x) >= 10,
        )
        == [0] * 10
    )


def test_flatmap_rectangles():
    result = minimal(
        integers(0, 10).flat_map(
            lambda w: lists(lists(sampled_from(["a", "b"]), min_size=w, max_size=w))
        ),
        lambda x: ["a", "b"] in x,
        max_examples=2000,
    )
    assert result == [["a", "b"]]


# From nocover/test_flatmap.py


@pytest.mark.requires("shrinking.sorting")
@pytest.mark.parametrize("n", [1, 3, 5, 9])
def test_can_shrink_through_a_binding(n):
    result = minimal(
        integers(0, 100).flat_map(lambda k: lists(booleans(), min_size=k, max_size=k)),
        lambda x: sum(x) >= n,
    )
    assert result == [True] * n


@pytest.mark.requires("shrinking.bind_deletion")
@pytest.mark.parametrize("n", [1, 3, 5, 9])
def test_can_delete_in_middle_of_a_binding(n):
    result = minimal(
        integers(1, 100).flat_map(lambda k: lists(booleans(), min_size=k, max_size=k)),
        lambda x: len(x) >= 2 and x[0] and x[-1] and x.count(False) >= n,
    )
    expected = [True] + [False] * n + [True]
    assert result == expected
