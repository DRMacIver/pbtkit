"""Collection shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/collections.rs.
"""

from random import Random

import pytest

import pbtkit.core as core
import pbtkit.generators as gs
from pbtkit import run_test
from pbtkit.core import ChoiceNode, IntegerChoice, Status
from pbtkit.core import PbtkitState as State

from .conftest import minimal

pytestmark = pytest.mark.requires("collections")

# --- Sets (using unique lists as proxy) ---


@pytest.mark.requires("shrinking.sorting")
def test_minimize_3_set():
    result = minimal(
        gs.lists(gs.integers(-(2**63), 2**63 - 1), unique=True),
        lambda x: len(x) >= 3,
    )
    assert result == [0, 1, -1]


@pytest.mark.requires("shrinking.sorting")
def test_minimize_sets_sampled_from():
    items = list(range(10))
    assert minimal(
        gs.lists(gs.sampled_from(items), min_size=3, unique=True),
    ) == [0, 1, 2]


# --- Containment tests ---


@gs.composite
def list_and_int(tc):
    v = tc.draw(gs.lists(gs.integers(0, 100)))
    i = tc.draw(gs.integers(0, 100))
    return (v, i)


_requires_duplication = pytest.mark.requires("shrinking.duplication_passes")


@pytest.mark.requires("shrinking.sorting")
@pytest.mark.parametrize(
    "n",
    [
        0,
        pytest.param(1, marks=_requires_duplication),
        pytest.param(10, marks=_requires_duplication),
        pytest.param(50, marks=_requires_duplication),
    ],
)
def test_containment(n):
    result = minimal(
        list_and_int(), lambda x: x[1] >= n and x[1] in x[0], max_examples=1000
    )
    assert result == ([n], n)


@pytest.mark.requires("shrinking.duplication_passes")
def test_duplicate_containment():
    ls, i = minimal(list_and_int(), lambda x: x[0].count(x[1]) > 1)
    assert ls == [0, 0]
    assert i == 0


# --- List ordering and structure ---


@pytest.mark.requires("shrinking.sorting")
def test_reordering_bytes():
    ls = minimal(
        gs.lists(gs.integers(0, 1000)),
        lambda x: sum(x) >= 10 and len(x) >= 3,
    )
    assert ls == sorted(ls)


def test_minimize_long_list():
    assert (
        minimal(gs.lists(gs.booleans(), min_size=50), lambda x: len(x) >= 70)
        == [False] * 70
    )


@pytest.mark.requires("shrinking.sorting")
def test_minimize_list_of_longish_lists():
    size = 5
    xs = minimal(
        gs.lists(gs.lists(gs.booleans())),
        lambda x: len([t for t in x if any(t) and len(t) >= 2]) >= size,
    )
    assert len(xs) == size
    for x in xs:
        assert x == [False, True]


def test_minimize_list_of_fairly_non_unique_ints():
    xs = minimal(
        gs.lists(gs.integers(0, 100)),
        lambda x: len(set(x)) < len(x),
    )
    assert len(xs) == 2


def test_list_with_complex_sorting_structure():
    xs = minimal(
        gs.lists(gs.lists(gs.booleans())),
        lambda x: list(reversed([list(reversed(t)) for t in x])) > x and len(x) > 3,
    )
    assert len(xs) == 4


def test_list_with_wide_gap():
    xs = minimal(
        gs.lists(gs.integers(-(2**63), 2**63 - 1)),
        lambda x: len(x) > 0 and max(x) > min(x) + 10 and min(x) + 10 > 0,
    )
    assert len(xs) == 2
    s = sorted(xs)
    assert s[1] == 11 + s[0]


# --- Lists of collections ---


def test_minimize_list_of_lists():
    result = minimal(
        gs.lists(gs.lists(gs.integers(-(2**63), 2**63 - 1))),
        lambda x: len([s for s in x if s]) >= 3,
    )
    assert result == [[0]] * 3


def test_minimize_list_of_tuples():
    result = minimal(
        gs.lists(
            gs.tuples(
                gs.integers(-(2**63), 2**63 - 1), gs.integers(-(2**63), 2**63 - 1)
            )
        ),
        lambda x: len(x) >= 2,
    )
    assert result == [(0, 0), (0, 0)]


# --- Lists forced near top ---


@pytest.mark.parametrize("n", [0, 1, 5, 10])
def test_lists_forced_near_top(n):
    assert minimal(
        gs.lists(gs.integers(-(2**63), 2**63 - 1), min_size=n, max_size=n + 2),
        lambda t: len(t) == n + 2,
    ) == [0] * (n + 2)


# --- Dictionaries ---


@pytest.mark.requires("text")
def test_dictionary_minimizes_to_empty():
    result = minimal(gs.dictionaries(gs.integers(-(2**63), 2**63 - 1), gs.text()))
    assert result == {}


@pytest.mark.requires("text")
def test_dictionary_minimizes_values():
    result = minimal(
        gs.dictionaries(gs.integers(-(2**63), 2**63 - 1), gs.text()),
        lambda t: len(t) >= 3,
    )
    assert len(result) >= 3
    assert set(result.values()) == {""}
    for k in result:
        if k < 0:
            assert k + 1 in result
        if k > 0:
            assert k - 1 in result


def test_minimize_multi_key_dicts():
    result = minimal(
        gs.dictionaries(gs.booleans().map(str), gs.booleans()),
        lambda x: len(x) > 0,
    )
    assert len(result) == 1
    assert result == {"False": False}


# --- Find tests ---


def test_find_dictionary():
    smallest = minimal(
        gs.dictionaries(
            gs.integers(-(2**63), 2**63 - 1), gs.integers(-(2**63), 2**63 - 1)
        ),
        lambda xs: any(k > v for k, v in xs.items()),
    )
    assert len(smallest) == 1


def test_can_find_list():
    x = minimal(
        gs.lists(gs.integers(-(2**63), 2**63 - 1)),
        lambda x: sum(x) >= 10,
    )
    assert sum(x) == 10


# --- Collectively minimize ---


def test_can_collectively_minimize_integers():
    n = 10
    xs = minimal(
        gs.lists(gs.integers(-(2**63), 2**63 - 1), min_size=n, max_size=n),
        lambda x: len(set(x)) >= 2,
        max_examples=2000,
    )
    assert len(xs) == n
    assert 2 <= len(set(xs)) <= 3


def test_can_collectively_minimize_booleans():
    n = 10
    xs = minimal(
        gs.lists(gs.booleans(), min_size=n, max_size=n),
        lambda x: len(set(x)) >= 2,
        max_examples=2000,
    )
    assert len(xs) == n
    assert len(set(xs)) == 2


@pytest.mark.requires("text")
def test_can_collectively_minimize_text():
    n = 10
    xs = minimal(
        gs.lists(gs.text(), min_size=n, max_size=n),
        lambda x: len(set(x)) >= 2,
        max_examples=2000,
    )
    assert len(xs) == n
    assert 2 <= len(set(xs)) <= 3


# --- Sorting pass regressions ---


def test_sorting_pass_survives_type_changes_from_lists():
    """Sorting insertion-sort must not crash when a successful swap
    changes the result so that choice types at pre-computed indices
    shift. Regression for AssertionError in sorting.py found by pbtsmith."""

    with pytest.raises(AssertionError):

        @run_test(max_examples=1, database={}, quiet=True, random=Random(0))
        def _(tc):
            v0 = tc.draw(gs.lists(gs.booleans(), max_size=10))
            v1 = tc.draw(gs.lists(gs.integers(0, 0), max_size=10))
            assert len(v0) == len(v1)


def test_sorting_full_sort_survives_stale_indices():
    """Sorting full-sort path must not crash when a prior group's
    sort shortens the result, making indices for the next group
    invalid. Regression for IndexError in sorting.py found by pbtsmith."""

    try:

        @run_test(max_examples=1, database={}, quiet=True, random=Random(1))
        def _(tc):
            v0 = tc.draw(gs.lists(gs.integers(0, 12), max_size=10))
            tc.draw(gs.booleans())
            if not (len(v0) == 0 or v0[0] > 0):
                raise AssertionError
            if len(v0) > 2:
                if not (len(v0) == 0):
                    raise AssertionError
    except AssertionError:
        pass


def test_sorting_stale_filter_with_punning():
    """Sorting stale-index filter must handle the case where punning
    changes node types so that a group has fewer than 2 members.
    Regression for AssertionError in sorting.py found by shrink comparison."""

    @gs.composite
    def pair(tc):
        a = tc.draw(gs.booleans())
        b = tc.draw(gs.booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.draw(gs.lists(gs.integers(0, 0), max_size=10))
        v1 = tc.draw(
            gs.integers(0, 0).flat_map(lambda _: gs.lists(gs.booleans(), max_size=1))
        )
        tc.draw(pair())
        if len(v0) != len(v1):
            tc.mark_status(Status.INTERESTING)

    for seed in range(5):
        state = State(Random(seed), tf, 200)
        state.run()


def test_unique_list_shrinks_using_negative_values():
    """Unique signed integer lists should shrink to use negative values
    when that gives smaller absolute values (e.g. [0,1,-1,2,-2] not [0,1,2,3,4]).
    Regression for shrink quality found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.lists(gs.integers(-10, 10), max_size=5, unique=True))
        if len(v0) >= 5:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    # Extract the integer values from the list choices (skip booleans)
    int_values = [n.value for n in state.result if isinstance(n.kind, IntegerChoice)]
    assert int_values == [0, 1, -1, 2, -2]


def test_redistribute_stale_indices_after_type_change():
    """redistribute_integers must handle stale indices when previous passes
    change the result structure, causing a node that was IntegerChoice to
    become BooleanChoice. Regression for AssertionError found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.booleans())
        tc.draw(gs.booleans().map(lambda x: int(x)))
        tc.draw(gs.integers(1, 7).filter(lambda x: x % 2 == 0))
        tc.draw(gs.booleans())
        tc.draw(gs.one_of(gs.integers(0, 0), gs.booleans()))
        if v0:
            tc.mark_status(Status.INTERESTING)

    # Should not crash.
    state = State(Random(0), tf, 1000)
    state.run()


@pytest.mark.requires("bytes")
@pytest.mark.requires("text")
def test_sort_insertion_stale_indices():
    """Sorting insertion sort must handle stale indices when a swap
    changes the result structure (e.g. shortening via value punning).
    Regression for IndexError in sorting.py found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.lists(gs.integers(0, 20), max_size=10, unique=True))
        tc.draw(
            gs.dictionaries(
                gs.text(min_codepoint=32, max_codepoint=126, max_size=5),
                gs.booleans(),
                max_size=5,
            )
        )
        v2 = tc.draw(gs.lists(gs.booleans(), max_size=10))
        v3 = tc.draw(gs.binary(max_size=20))
        tc.draw(gs.booleans())
        if len(v0) != 0:
            tc.mark_status(Status.INTERESTING)
        if len(v2) != len(v3):
            tc.mark_status(Status.INTERESTING)

    # Should not crash. Try multiple seeds to exercise sorting edge cases.
    for seed in range(5):
        state = State(Random(seed), tf, 1000)
        state.run()


@pytest.mark.requires("shrinking.sorting")
def test_sort_values_insertion_natural_exit():
    """sort_values insertion sort j-decrement and natural while-exit paths.

    Calls sort_values directly (bypassing earlier passes like redistribute_integers
    that would otherwise find the result first). Starting from [1, 0, 0] with
    a+b>c: full sort [0,0,1] fails (0+0>1=False), insertion sort at pos=1 swaps
    [1,0]→[0,1] (j→0, natural while-exit), covering both j-decrement and
    2088→2086 branch."""

    def tf(tc):
        a = tc.draw_integer(0, 10)
        b = tc.draw_integer(0, 10)
        c = tc.draw_integer(0, 10)
        if a + b > c:
            tc.mark_status(Status.INTERESTING)

    sort_fn = next(p for p in core.SHRINK_PASSES if p.__name__ == "sort_values")
    ic = IntegerChoice(0, 10)
    state = State(Random(0), tf, 1000)
    state.result = [
        ChoiceNode(kind=ic, value=1, was_forced=False),
        ChoiceNode(kind=ic, value=0, was_forced=False),
        ChoiceNode(kind=ic, value=0, was_forced=False),
    ]
    sort_fn(state)
    assert state.result is not None
    assert [n.value for n in state.result] == [0, 1, 0]


@pytest.mark.requires("shrinking.sorting")
def test_sort_stale_indices_after_punning():
    """Sorting handles indices becoming stale when a swap changes types
    via value punning (e.g. one_of branch switch)."""

    def tf(tc):
        v0 = tc.draw(gs.one_of(gs.integers(0, 10), gs.integers(0, 10)))
        v1 = tc.draw(gs.one_of(gs.integers(0, 10), gs.integers(0, 10)))
        if v0 + v1 > 10:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None


@pytest.mark.requires("shrinking.sorting")
def test_swap_adjacent_blocks_identical():
    """swap_adjacent_blocks skips identical adjacent blocks.
    With a==b==c==d and a>0, minimum is [1,1,1,1]. Starting directly
    from [1,1,1,1], block_size=2 gives block_a=[1,1]==block_b=[1,1],
    covering the identical-block skip guard at every shrink iteration."""

    def tf(tc):
        a = tc.draw_integer(0, 10)
        b = tc.draw_integer(0, 10)
        c = tc.draw_integer(0, 10)
        d = tc.draw_integer(0, 10)
        if a == b == c == d and a > 0:
            tc.mark_status(Status.INTERESTING)

    ic = IntegerChoice(0, 10)
    state = State(Random(0), tf, 1000)
    state.result = [
        ChoiceNode(kind=ic, value=1, was_forced=False),
        ChoiceNode(kind=ic, value=1, was_forced=False),
        ChoiceNode(kind=ic, value=1, was_forced=False),
        ChoiceNode(kind=ic, value=1, was_forced=False),
    ]
    state.shrink()
    assert state.result is not None
    assert [n.value for n in state.result] == [1, 1, 1, 1]
