"""Collection shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/collections.rs.
"""

import pytest

import minithesis.generators as gs

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
    v = tc.any(gs.lists(gs.integers(0, 100)))
    i = tc.any(gs.integers(0, 100))
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
