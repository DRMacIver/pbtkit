"""String shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/strings.rs.
"""

from random import Random

import pytest

import minithesis.generators as gs
from minithesis.core import MinithesisState as State
from minithesis.core import Status

from .conftest import minimal

pytestmark = pytest.mark.requires("text")


def test_minimize_string_to_empty():
    assert minimal(gs.text()) == ""


def test_minimize_longer_string():
    result = minimal(gs.text(max_size=50), lambda x: len(x) >= 10)
    assert result == "0" * 10


@pytest.mark.requires("collections")
def test_minimize_longer_list_of_strings():
    assert minimal(gs.lists(gs.text()), lambda x: len(x) >= 10) == [""] * 10


def test_string_sorts_characters_when_possible():
    """String shrinking should sort characters by codepoint.
    Sorting '0e0' produces '00e' (smaller codepoints first)."""

    def tf(tc):
        v0 = tc.draw(gs.text(min_codepoint=32, max_codepoint=126, max_size=20))
        if len(v0) >= 3 and "e" in v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    assert state.result[0].value == "00e"


def test_string_insertion_sort_swap_succeeds():
    """String shrinker insertion sort swaps out-of-order adjacent chars,
    covering j -= 1 and natural while-exit (j reaches 0).

    Uses a fixed-length 2-char string where the condition requires both
    'a' and 'b'. Starting from "ba" (out of order), insertion sort swaps
    to "ab" (still satisfies the condition), covering both paths."""

    def tf(tc):
        s = tc.draw_string(ord("a"), ord("b"), 2, 2)
        # Permutation-invariant condition: needs both 'a' and 'b'.
        if "a" in s and "b" in s:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    assert state.result[0].value == "ab"


@pytest.mark.requires("shrinking.advanced_string_passes")
def test_string_length_redistribution():
    """When two strings share a total length constraint (len(v0)+len(v1) >= N),
    the shrinker should redistribute length to make the first string as short
    as possible, even though shortening v0 requires lengthening v1.
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.draw(gs.text(min_codepoint=32, max_codepoint=126, max_size=20))
        v1 = tc.draw(gs.text(min_codepoint=32, max_codepoint=126, max_size=20))
        if len(v0) + len(v1) >= 30:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    v0_len = len(state.result[0].value)
    # Optimal: v0 as short as possible (10 chars, since v1 max is 20).
    assert v0_len == 10
