"""Shared utilities for findability tests."""

from __future__ import annotations

from collections.abc import Callable
from random import Random
from typing import TypeVar

from pbtkit.core import Generator, PbtkitState, Status, StopTest, TestCase

T = TypeVar("T")


def finds(
    generator: Generator[T],
    condition: Callable[[T], bool] = lambda _: True,
    max_examples: int = 1000,
) -> T:
    """Assert that ``generator`` can find a value satisfying ``condition``.

    Returns the (shrunk) value found. Unlike the shrink_quality ``minimal``
    helper, the caller should NOT assert on the exact shape of the result —
    these tests only care that the engine is capable of finding *something*."""

    best = [None]

    def test_function(tc: TestCase) -> None:
        value = tc.draw(generator)
        if condition(value):
            best[0] = value
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), test_function, max_examples)
    state.run()
    assert state.result is not None, "No example found"
    # Replay to capture the final shrunk value.
    tc = TestCase.for_choices([n.value for n in state.result])
    try:
        test_function(tc)
    except StopTest:
        pass
    return best[0]
