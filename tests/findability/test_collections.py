"""Collection findability tests.

Ported from Hypothesis's test_shrink_quality.py. These use unbounded
integers (unlike the shrink_quality versions which cap at 0-100) to
verify the engine can find results in a much larger search space.
"""

import pytest

import pbtkit.generators as gs

from .conftest import finds

pytestmark = pytest.mark.requires("collections")


@gs.composite
def list_and_int(tc):
    v = tc.draw(gs.lists(gs.integers(-(2**63), 2**63 - 1)))
    i = tc.draw(gs.integers(-(2**63), 2**63 - 1))
    return (v, i)


@pytest.mark.requires("edge_case_boosting")
@pytest.mark.parametrize("n", [0, 1, 10, 100, 1000])
def test_containment(n):
    """Engine can find (list, int) where int >= n and int is in the list,
    using unbounded integers."""
    result = finds(
        list_and_int(),
        lambda x: x[1] >= n and x[1] in x[0],
    )
    ls, i = result
    assert i >= n
    assert i in ls


@pytest.mark.requires("span_mutation")
def test_duplicate_containment():
    """Engine can find (list, int) where int appears more than once in the list,
    using unbounded integers."""
    result = finds(
        list_and_int(),
        lambda x: x[0].count(x[1]) > 1,
    )
    ls, i = result
    assert ls.count(i) > 1


def test_can_find_list_with_sum():
    """Engine can find a list of unbounded integers with sum >= 10."""
    result = finds(
        gs.lists(gs.integers(-(2**63), 2**63 - 1)),
        lambda x: sum(x) >= 10,
    )
    assert sum(result) >= 10


def test_can_find_dictionary_with_key_gt_value():
    """Engine can find a dictionary where some key > its value,
    using unbounded integers."""
    result = finds(
        gs.dictionaries(
            gs.integers(-(2**63), 2**63 - 1), gs.integers(-(2**63), 2**63 - 1)
        ),
        lambda xs: any(k > v for k, v in xs.items()),
    )
    assert any(k > v for k, v in result.items())


def test_can_find_sorted_list():
    """Engine can find a list of integers that is NOT sorted."""
    finds(
        gs.lists(gs.integers(-(2**63), 2**63 - 1)),
        lambda xs: sorted(xs) != xs,
    )


def test_can_find_large_sum_list():
    """Engine can find a list with sum >= 100."""
    result = finds(
        gs.lists(gs.integers(0, 100)),
        lambda xs: sum(xs) >= 100,
    )
    assert sum(result) >= 100
