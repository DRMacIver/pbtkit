"""Generator functions for minithesis.

This module provides the user-facing API for generating test data,
analogous to hypothesis.strategies in Hypothesis.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, List, NoReturn, Sequence, TypeVar

from minithesis import Generator, Status, TestCase

T = TypeVar("T", covariant=True)
U = TypeVar("U")


class many:
    """Utility class for drawing variable-length collections. Handles
    the "should I keep drawing more values?" logic using a geometric
    distribution with forced results for min/max constraints.

    Ported from Hypothesis's conjecture.utils.many.

    Usage:
        elements = many(test_case, min_size=0, max_size=10)
        while elements.more():
            result.append(draw_something())
    """

    def __init__(
        self,
        tc: TestCase,
        min_size: int,
        max_size: float,
    ) -> None:
        assert min_size <= max_size
        average_size = min(
            max(min_size * 2, min_size + 5),
            0.5 * (min_size + max_size),
        )
        self.tc = tc
        self.min_size = min_size
        self.max_size = max_size
        desired_extra = average_size - min_size
        max_extra = max_size - min_size
        if desired_extra >= max_extra:
            # Slightly short of 1.0 so as to not force this to always
            # be the maximum size.
            self.p_continue = 0.99
        elif math.isinf(max_size):
            # Geometric distribution with the right expected size.
            self.p_continue = 1.0 - 1.0 / (1 + desired_extra)
        else:
            # If we have a finite max size we slightly undershoot
            # with the geometric estimate. We'd rather produce
            # slightly too large than slightly too small collections,
            # and the exact probability doesn't matter very much, so
            # we bump it up slightly to compensate for the undershoot.
            self.p_continue = 1.0 - 1.0 / (2 + desired_extra)
        self.count = 0
        self.rejections = 0
        self.force_stop = False

    def more(self) -> bool:
        """Should we draw another element?"""
        if self.min_size == self.max_size:
            # Fixed size: draw exactly min_size elements.
            should_continue = self.count < self.min_size
        else:
            if self.force_stop:
                forced = False
            elif self.count < self.min_size:
                forced = True
            elif self.count >= self.max_size:
                forced = False
            else:
                forced = None
            should_continue = self.tc.weighted(self.p_continue, forced=forced)

        if should_continue:
            self.count += 1
            return True
        return False

    def reject(self) -> None:
        """Reject the last drawn element (e.g. because it was a
        duplicate). Decrements the count and may force the
        collection to stop if too many rejections occur."""
        assert self.count > 0
        self.count -= 1
        self.rejections += 1
        if self.rejections > max(3, 2 * self.count):
            if self.count < self.min_size:
                self.tc.mark_status(Status.INVALID)
            else:
                self.force_stop = True


def floats(
    min_value: float = -float("inf"),
    max_value: float = float("inf"),
    *,
    allow_nan: bool = True,
    allow_infinity: bool = True,
) -> Generator[float]:
    """Generates floating-point numbers.

    For bounded ranges, generates uniformly. For unbounded ranges,
    explores the full float space including special values."""
    return Generator(
        lambda tc: tc.draw_float(
            min_value,
            max_value,
            allow_nan=allow_nan,
            allow_infinity=allow_infinity,
        ),
        name=f"floats(min_value={min_value}, max_value={max_value}, allow_nan={allow_nan}, allow_infinity={allow_infinity})",
    )


def text(
    min_codepoint: int = 0,
    max_codepoint: int = 0x10FFFF,
    min_size: int = 0,
    max_size: int = 10,
) -> Generator[str]:
    """Generates strings with codepoints in [min_codepoint, max_codepoint]
    and length in [min_size, max_size]. Surrogates are excluded."""
    defaults = {
        "min_codepoint": 0,
        "max_codepoint": 0x10FFFF,
        "min_size": 0,
        "max_size": 10,
    }
    params = {
        "min_codepoint": min_codepoint,
        "max_codepoint": max_codepoint,
        "min_size": min_size,
        "max_size": max_size,
    }
    args = ", ".join(f"{k}={v}" for k, v in params.items() if v != defaults[k])
    return Generator(
        lambda tc: tc.draw_string(min_codepoint, max_codepoint, min_size, max_size),
        name=f"text({args})" if args else "text()",
    )


def integers(min_value: int, max_value: int) -> Generator[int]:
    """Generates an integer in the range [min_value, max_value]."""
    return Generator(
        lambda tc: tc.draw_integer(min_value, max_value),
        name=f"integers(min_value={min_value}, max_value={max_value})",
    )


def binary(min_size: int = 0, max_size: int = 8) -> Generator[bytes]:
    """Generates a byte string with length in [min_size, max_size]."""
    return Generator(
        lambda tc: tc.draw_bytes(min_size, max_size),
        name=f"binary(min_size={min_size}, max_size={max_size})",
    )


def lists(
    elements: Generator[U],
    min_size: int = 0,
    max_size: float = float("inf"),
) -> Generator[List[U]]:
    """Generates lists whose elements are drawn from ``elements``.

    Uses a geometric distribution for list length, with forced
    results to respect min_size and max_size bounds."""

    def produce(test_case: TestCase) -> List[U]:
        result: List[U] = []
        elems = many(test_case, min_size=min_size, max_size=max_size)
        while elems.more():
            result.append(test_case.any(elements))
        return result

    return Generator[List[U]](
        produce,
        name=f"lists({elements.name}, min_size={min_size}, max_size={max_size})",
    )


def just(value: U) -> Generator[U]:
    """Always generates ``value``."""
    return Generator[U](lambda tc: value, name=f"just({value})")


def nothing() -> Generator[NoReturn]:
    """No possible values. Any call to ``any`` will reject the test case."""

    def produce(tc: TestCase) -> NoReturn:
        tc.reject()

    return Generator(produce)


def sampled_from(elements: Sequence[U]) -> Generator[U]:
    """Generates values by picking uniformly from ``elements``.
    Shrinks toward earlier elements in the sequence."""
    if not elements:
        return nothing()
    if len(elements) == 1:
        return just(elements[0])
    return Generator(
        lambda tc: elements[tc.draw_integer(0, len(elements) - 1)],
        name=f"sampled_from({list(elements)!r})",
    )


def booleans() -> Generator[bool]:
    """Generates True or False. Shrinks toward False."""
    return Generator(
        lambda tc: tc.weighted(0.5),
        name="booleans()",
    )


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


def composite(
    fn: Callable[..., U],
) -> Callable[..., Generator[U]]:
    """Decorator for writing generators as functions. The decorated
    function receives a ``TestCase`` as its first argument and can
    use it to draw values. Additional arguments become parameters
    of the returned generator.

    Usage::

        @composite
        def pairs(tc):
            x = tc.any(integers(0, 10))
            y = tc.any(integers(x, 10))
            return (x, y)

        @run_test()
        def _(tc):
            x, y = tc.any(pairs())
            assert x <= y
    """

    @functools.wraps(fn)
    def accept(*args: Any, **kwargs: Any) -> Generator[U]:
        def produce(tc: TestCase) -> U:
            return fn(tc, *args, **kwargs)

        return Generator(produce, name=fn.__name__)

    return accept
