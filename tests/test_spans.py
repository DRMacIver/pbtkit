"""Tests for span tracking and span-based mutation."""

from random import Random

import pytest

import pbtkit.generators as gs
from pbtkit import run_test
from pbtkit.core import PbtkitState, Status, StopTest, TestCase

pytestmark = [
    pytest.mark.requires("spans"),
    pytest.mark.requires("collections"),
]


def test_draw_records_spans():
    """Each draw() call creates a span covering the choices it used."""
    tc = TestCase.for_choices([3, 5])
    tc.draw(gs.integers(0, 10))
    tc.draw(gs.integers(0, 10))
    assert len(tc.spans) == 2
    label0, start0, stop0 = tc.spans[0]
    label1, start1, stop1 = tc.spans[1]
    assert start0 == 0
    assert stop0 == 1
    assert start1 == 1
    assert stop1 == 2
    assert "integers" in label0
    assert "integers" in label1


def test_nested_spans():
    """Composite generators create nested spans."""

    @gs.composite
    def pair(tc):
        a = tc.draw(gs.integers(0, 5))
        b = tc.draw(gs.integers(0, 5))
        return (a, b)

    tc = TestCase.for_choices([1, 2])
    tc.draw(pair())
    # Should have 3 spans: outer (pair) + 2 inner (integers)
    assert len(tc.spans) == 3
    # Inner spans close before outer
    inner1 = tc.spans[0]
    inner2 = tc.spans[1]
    outer = tc.spans[2]
    assert inner1[1] == 0 and inner1[2] == 1  # first integer
    assert inner2[1] == 1 and inner2[2] == 2  # second integer
    assert outer[1] == 0 and outer[2] == 2  # pair covers both


def test_list_draw_has_spans():
    """Drawing a list creates a span for the list and sub-spans for elements."""
    tc = TestCase.for_choices([True, 3, False])  # 1 element, value 3, stop
    tc.draw(gs.lists(gs.integers(0, 10), max_size=5))
    # Should have at least the outer list span
    assert len(tc.spans) >= 1
    outer = tc.spans[-1]
    assert outer[1] == 0  # starts at beginning
    assert outer[2] == len(tc.nodes)  # ends at end


@pytest.mark.requires("span_mutation")
def test_span_mutation_finds_duplicate():
    """Span mutation can find duplicate compound elements in a list."""
    # Use a small range so mutation is effective
    gen = gs.lists(gs.tuples(gs.integers(0, 3), gs.integers(0, 3)), max_size=10)

    best = [None]

    def test_fn(tc: TestCase) -> None:
        ls = tc.draw(gen)
        if len(ls) != len(set(ls)):
            best[0] = ls
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), test_fn, 2000)
    state.run()
    assert state.result is not None
    assert best[0] is not None


@pytest.mark.requires("span_mutation")
def test_span_mutation_noop_without_spans():
    """Span mutation returns early when base case has no spans."""
    from pbtkit.span_mutation import span_mutation

    # A test that draws nothing produces no spans.
    state = PbtkitState(Random(0), lambda tc: None, 100)
    span_mutation(state)


@pytest.mark.requires("span_mutation")
def test_span_mutation_exercises_swaps():
    """Span mutation generates a test case and tries span swaps."""
    from pbtkit.span_mutation import span_mutation

    # Use min_size=2 to guarantee at least 2 tuple elements → multiple spans
    gen = gs.lists(
        gs.tuples(gs.integers(0, 3), gs.integers(0, 3)), min_size=2, max_size=5
    )
    calls = [0]

    def test_fn(tc: TestCase) -> None:
        calls[0] += 1
        tc.draw(gen)

    # Try multiple seeds until we get one that produces multi-span output
    for seed in range(20):
        calls[0] = 0
        state = PbtkitState(Random(seed), test_fn, 10000)
        span_mutation(state)
        if calls[0] > 1:
            return
    assert False, "no seed produced span mutation swaps"
