"""Generator functions for minithesis.

This module provides the user-facing API for generating test data,
analogous to hypothesis.strategies in Hypothesis.
"""

from __future__ import annotations

from typing import Any, List, NoReturn, TypeVar

from minithesis import Generator, TestCase

T = TypeVar("T", covariant=True)
U = TypeVar("U")


def integers(min_value: int, max_value: int) -> Generator[int]:
    """Generates an integer in the range [min_value, max_value]."""
    return Generator(
        lambda tc: tc.draw_integer(min_value, max_value),
        name=f"integers(min_value={min_value}, max_value={max_value})",
    )


def binary(min_size: int = 0, max_size: int = 8) -> Generator[bytes]:
    """Any byte string with length in [min_size, max_size] is possible."""
    return Generator(
        lambda tc: tc.draw_bytes(min_size, max_size),
        name=f"binary(min_size={min_size}, max_size={max_size})",
    )


def lists(
    elements: Generator[U],
    min_size: int = 0,
    max_size: float = float("inf"),
) -> Generator[List[U]]:
    """Generates lists whose elements are drawn from ``elements``."""

    def produce(test_case: TestCase) -> List[U]:
        result: List[U] = []
        while True:
            if len(result) < min_size:
                test_case.weighted(0.9, forced=True)
            elif len(result) + 1 >= max_size:
                test_case.weighted(0.9, forced=False)
                break
            elif not test_case.weighted(0.9):
                break
            result.append(test_case.any(elements))
        return result

    return Generator(
        produce,
        name=f"lists({elements.name}, min_size={min_size}, max_size={max_size})",
    )


def just(value: U) -> Generator[U]:
    """Only ``value`` is possible."""
    return Generator[U](lambda tc: value, name=f"just({value})")


def nothing() -> Generator[NoReturn]:
    """No possible values. i.e. Any call to ``any`` will reject
    the test case."""

    def produce(tc: TestCase) -> NoReturn:
        tc.reject()

    return Generator(produce)


def one_of(*generators: Generator[T]) -> Generator[T]:
    """Randomly picks one of the given generators and draws from it."""
    if not generators:
        return nothing()
    if len(generators) == 1:
        return generators[0]
    return Generator(
        lambda tc: tc.any(generators[tc.choice(len(generators) - 1)]),
        name=f"one_of({', '.join(g.name for g in generators)})",
    )


def tuples(*generators: Generator[Any]) -> Generator[Any]:
    """Generates a tuple of length len(generators) where element i
    is drawn from generators[i]."""
    return Generator(
        lambda tc: tuple(tc.any(g) for g in generators),
        name=f"tuples({', '.join(g.name for g in generators)})",
    )
