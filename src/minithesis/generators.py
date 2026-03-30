"""Generator functions for minithesis.

This module provides the user-facing API for generating test data,
analogous to hypothesis.strategies in Hypothesis.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import (
    Any,
    NoReturn,
    TypeVar,
)

from minithesis.collections import many
from minithesis.core import Generator, TestCase

T = TypeVar("T", covariant=True)
U = TypeVar("U")


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
    unique: bool = False,
    unique_by: Callable[[U], Any] | None = None,
) -> Generator[list[U]]:
    """Generates lists whose elements are drawn from ``elements``.

    Uses a geometric distribution for list length, with forced
    results to respect min_size and max_size bounds.

    If ``unique`` is True, all elements will be distinct.
    If ``unique_by`` is provided, elements will be distinct
    by the key function."""

    needs_unique = unique or unique_by is not None
    key_fn = unique_by if unique_by is not None else (lambda x: x)

    if needs_unique:

        def produce(test_case: TestCase) -> list[U]:
            result: list[U] = []
            seen: set = set()
            elems = many(test_case, min_size=min_size, max_size=max_size)
            while elems.more():
                value = test_case.any(elements)
                key = key_fn(value)
                if key in seen:
                    elems.reject()
                else:
                    seen.add(key)
                    result.append(value)
            return result

    else:

        def produce(test_case: TestCase) -> list[U]:
            result: list[U] = []
            elems = many(test_case, min_size=min_size, max_size=max_size)
            while elems.more():
                result.append(test_case.any(elements))
            return result

    return Generator[list[U]](
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


V = TypeVar("V")


def dictionaries(
    keys: Generator[U],
    values: Generator[V],
    min_size: int = 0,
    max_size: float = float("inf"),
) -> Generator[dict[U, V]]:
    """Generates dictionaries with keys from ``keys`` and values
    from ``values``. Duplicate keys are rejected using many()'s
    rejection mechanism."""

    def produce(test_case: TestCase) -> dict[U, V]:
        result: dict[U, V] = {}
        elems = many(test_case, min_size=min_size, max_size=max_size)
        while elems.more():
            k = test_case.any(keys)
            v = test_case.any(values)
            if k in result:
                elems.reject()
            else:
                result[k] = v
        return result

    return Generator[dict[U, V]](
        produce,
        name=f"dictionaries({keys.name}, {values.name}, min_size={min_size}, max_size={max_size})",
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
