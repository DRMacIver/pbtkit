"""Type and structural findability tests.

Ported from Hypothesis's test_testdecorators.py (@fails tests).
These verify the engine can find structural counterexamples —
type mismatches, unsorted lists, etc.
"""

import pytest

import pbtkit.generators as gs
from pbtkit import run_test

pytestmark = pytest.mark.requires("collections")


@pytest.mark.requires("floats")
def test_one_of_produces_different_types():
    """Engine can find values from one_of with different types."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x = tc.draw(gs.one_of(gs.floats(), gs.booleans()))
            y = tc.draw(gs.one_of(gs.floats(), gs.booleans()))
            assert type(x) == type(y)


def test_list_is_not_always_sorted():
    """Engine can find an unsorted list."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            xs = tc.draw(gs.lists(gs.integers(0, 100)))
            assert sorted(xs) == xs


@pytest.mark.requires("floats")
def test_float_is_not_always_an_endpoint():
    """Engine can find a float in (1.0, 2.0) that is not 1.0 or 2.0."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x = tc.draw(gs.floats(1.0, 2.0, allow_nan=False))
            assert x in {1.0, 2.0}


@pytest.mark.xfail(reason="text generation doesn't find duplicates in 200 examples")
@pytest.mark.requires("text")
def test_can_find_string_with_duplicate_characters():
    """Engine can find a string where not all characters are unique."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            s = tc.draw(gs.text(min_size=2))
            assert len(set(s)) == len(s)


@pytest.mark.requires("text")
def test_can_find_non_ascii_text():
    """Engine can find text that cannot be encoded as ASCII."""
    with pytest.raises(UnicodeEncodeError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            x = tc.draw(gs.text())
            x.encode("ascii")


def test_removing_element_from_non_unique_list():
    """Engine can find a list where removing an element still leaves it present."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            xs = tc.draw(gs.lists(gs.integers(0, 10), min_size=2))
            y = tc.draw(gs.sampled_from(xs))
            xs.remove(y)
            assert y not in xs
