"""Integer shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/integers.rs.
"""

from random import Random

import pytest

import pbtkit.generators as gs
from pbtkit import run_test
from pbtkit.core import PbtkitState as State
from pbtkit.core import Status

from .conftest import minimal


def test_integers_from_minimizes_leftwards():
    assert minimal(gs.integers(101, 2**63)) == 101


def test_minimize_bounded_integers_to_zero():
    assert minimal(gs.integers(-10, 10)) == 0


def test_minimize_bounded_integers_to_positive():
    assert minimal(gs.integers(-10, 10), lambda x: x != 0) == 1


def test_minimize_single_element_in_silly_large_int_range():
    hi = 2**63 - 1
    lo = -(2**63)
    result = minimal(gs.integers(lo // 2, hi // 2), lambda x: x >= lo // 4)
    assert result == 0


@pytest.mark.requires("collections")
def test_minimize_multiple_elements_in_silly_large_int_range():
    hi = 2**63 - 1
    lo = -(2**63)
    result = minimal(
        gs.lists(gs.integers(lo // 2, hi // 2)),
        lambda x: len(x) >= 20,
        max_examples=10000,
    )
    assert result == [0] * 20


@pytest.mark.requires("collections")
def test_minimize_multiple_elements_min_is_not_dupe():
    @gs.composite
    def bounded_int_list(tc):
        return tc.draw(gs.lists(gs.integers(0, 2**62)))

    target = list(range(20))
    result = minimal(
        bounded_int_list(),
        lambda x: len(x) >= 20 and all(x[i] >= target[i] for i in range(20)),
        max_examples=10000,
    )
    assert result == list(range(20))


def test_can_find_an_int():
    assert minimal(gs.integers(-(2**63), 2**63 - 1)) == 0


def test_can_find_an_int_above_13():
    assert minimal(gs.integers(-(2**63), 2**63 - 1), lambda x: x >= 13) == 13


def test_minimizes_towards_zero():
    assert minimal(gs.integers(-1000, 50), lambda x: x < 0) == -1


def test_integer_shrinks_negative():
    """Negative integers shrink toward zero via swap_integer_sign
    and binary_search_integer_towards_zero."""

    def tf(tc):
        n = tc.draw_integer(-1000, 1000)
        if n < 0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    assert state.result[0].value == -1


def test_integer_shrinks_via_binary_search():
    """Large integers shrink via binary search toward zero."""

    def tf(tc):
        n = tc.draw_integer(0, 10000)
        if n > 100:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    assert state.result[0].value == 101


def test_integer_shrinks_negative_only_range():
    """Shrinking in a range with max_value <= 0 exercises the
    early exit in binary_search_integer_towards_zero."""

    def tf(tc):
        n = tc.draw_integer(-100, -1)
        if n <= -10:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    assert state.result[0].value == -10


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_reduces_additive_pairs(capsys):

    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=10000)
        def _(test_case):
            m = test_case.choice(1000)
            n = test_case.choice(1000)
            assert m + n <= 1000

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 1",
        "choice(1000): 1000",
    ]


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_redistribute_stale_indices():
    """redistribute_integers must handle indices shrinking mid-loop
    when redistribution changes the path (e.g. reducing count)."""

    def tf(tc):
        # The boolean-controlled branch means redistribution can
        # change the number of integer nodes.
        b = tc.weighted(0.5)
        a = tc.draw_integer(0, 100)
        c = tc.draw_integer(0, 100)
        if b:
            d = tc.draw_integer(0, 100)
            if a + c + d > 200:
                tc.mark_status(Status.INTERESTING)
        else:
            if a + c > 100:
                tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 2000)
    state.run()
    assert state.result is not None


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_redistribute_stale_indices_at_gap_two():
    """redistribute_integers must handle stale indices when gap=1 redistribution
    shortens the result, making the outer gap=2 loop's pre-computed range stale.
    The condition pair_idx - 1 + gap >= len(indices) fires when the replacement
    shrinks the result from 3 integer nodes to 2, then gap=2 iterates pair_idx=1
    which needs indices[2] that no longer exists.
    Regression for stale index guard found by pbtsmith."""

    def tf(tc):
        gate = tc.draw_integer(0, 138)
        base = tc.draw_integer(0, 100)
        if gate > 46:
            extra = tc.draw_integer(0, 100)
            if base + extra > 30:
                tc.mark_status(Status.INTERESTING)
        else:
            if base > 27:
                tc.mark_status(Status.INTERESTING)

    state = State(Random(117), tf, 3000)
    state.run()
    assert state.result is not None
    # Should shrink to: gate=0 (short path), base=28 (> 27)
    assert [n.value for n in state.result] == [0, 28]
