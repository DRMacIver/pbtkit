"""Shared utilities for shrink quality tests.

Ported from Hypothesis's shrink quality test suite via hegel-rust.
"""

from __future__ import annotations

from collections.abc import Callable
from random import Random
from typing import TypeVar

from minithesis.core import Generator, MinithesisState, Status, StopTest, TestCase

T = TypeVar("T")


def minimal(
    generator: Generator[T],
    condition: Callable[[T], bool] = lambda _: True,
    max_examples: int = 1000,
) -> T:
    """Find the minimal value from ``generator`` that satisfies ``condition``.

    Runs generation + shrinking and returns the smallest counterexample."""

    best = [None]

    def test_function(tc: TestCase) -> None:
        value = tc.any(generator)
        if condition(value):
            best[0] = value
            tc.mark_status(Status.INTERESTING)

    state = MinithesisState(Random(0), test_function, max_examples)
    state.run()
    assert state.result is not None, "No example found"
    # Replay to capture the final shrunk value.
    tc = TestCase.for_choices([n.value for n in state.result])
    try:
        test_function(tc)
    except StopTest:
        pass
    return best[0]
