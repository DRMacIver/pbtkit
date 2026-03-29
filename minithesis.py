# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
This file implements a simple property-based testing library called
minithesis. It's not really intended to be used as is, but is instead
a proof of concept that implements as much of the core ideas of
Hypothesis in a simple way that is designed for people who want to
implement a property-based testing library for non-Python languages.

minithesis is always going to be self-contained in a single file
and consist of < 1000 sloc (as measured by cloc). This doesn't
count comments and I intend to comment on it extensively.


=============
PORTING NOTES
=============

minithesis supports roughly the following features, more or less
in order of most to least important:

1. Test case generation.
2. Test case reduction ("shrinking")
3. A small library of primitive generators and combinators.
4. A Test case database for replay between runs.
5. Targeted property-based testing
6. A caching layer for mapping choice sequences to outcomes


Anything that supports 1 and 2 is a reasonably good first porting
goal. You'll probably want to port most of the generators library
because it's easy and it helps you write tests, but don't worry
too much about the specifics.

The test case database is *very* useful and I strongly encourage
you to support it, but if it's fiddly feel free to leave it out
of a first pass.

Targeted property-based testing is very much a nice to have. You
probably don't need it, but it's a rare enough feature that supporting
it gives you bragging rights and who doesn't love bragging rights?

The caching layer you can skip. It's used more heavily in Hypothesis
proper, but in minithesis you only really need it for shrinking
performance, so it's mostly a nice to have.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum
from random import Random

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)


T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)
U = TypeVar("U")


@dataclass(frozen=True)
class ChoiceType(Generic[U]):
    """Base class for typed choices. The type parameter U is the
    type of value this choice produces."""

    @property
    def simplest(self) -> U:
        """The simplest value for this choice type, used as the
        shrink target."""
        raise NotImplementedError

    def validate(self, value: U) -> bool:
        """Return True if value is valid for this choice type."""
        raise NotImplementedError

    def sort_key(self, value: U) -> Any:
        """Returns a comparable key for ordering values during
        shrinking. By default just returns the value itself."""
        return value


@dataclass(frozen=True)
class IntegerChoice(ChoiceType[int]):
    min_value: int
    max_value: int

    @property
    def simplest(self) -> int:
        return self.min_value

    def validate(self, value: int) -> bool:
        return self.min_value <= value <= self.max_value


@dataclass(frozen=True)
class BooleanChoice(ChoiceType[bool]):
    p: float

    @property
    def simplest(self) -> bool:
        return False

    def validate(self, value: bool) -> bool:
        return value in (0, 1)


@dataclass(frozen=True)
class BytesChoice(ChoiceType[bytes]):
    min_size: int
    max_size: int

    @property
    def simplest(self) -> bytes:
        return b"\x00" * self.min_size

    def validate(self, value: bytes) -> bool:
        return isinstance(value, bytes) and self.min_size <= len(value) <= self.max_size

    def sort_key(self, value: bytes) -> Any:
        """Shortlex ordering: shorter is simpler, then lexicographic."""
        return (len(value), value)


@dataclass(frozen=True)
class StringChoice(ChoiceType[str]):
    min_codepoint: int
    max_codepoint: int
    min_size: int
    max_size: int

    @property
    def simplest(self) -> str:
        return chr(self.min_codepoint) * self.min_size

    def validate(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        if not (self.min_size <= len(value) <= self.max_size):
            return False
        return all(
            self.min_codepoint <= ord(c) <= self.max_codepoint
            and not (0xD800 <= ord(c) <= 0xDFFF)
            for c in value
        )

    def sort_key(self, value: str) -> Any:
        """Shortlex ordering: shorter is simpler, then by codepoints."""
        return (len(value), value)


def _draw_unbounded_float(random: Random) -> float:
    """Generate a random float from the full float space,
    excluding NaN (via rejection sampling). NaN is ~0.05%
    of bit patterns so this rarely loops."""
    while True:
        result = _lex_to_float(random.getrandbits(64))
        if not math.isnan(result):
            return result


def _draw_nan(random: Random) -> float:
    """Generate a random NaN value."""
    # Set exponent to all 1s and a random non-zero mantissa.
    exponent = 0x7FF << 52
    sign = random.getrandbits(1) << 63
    mantissa = random.getrandbits(52) or 1  # ensure non-zero
    return struct.unpack("!d", struct.pack("!Q", sign | exponent | mantissa))[0]


def _lex_to_float(bits: int) -> float:
    """Convert a lexicographically ordered 64-bit integer to a float.
    Used by the unbounded float generator to produce floats from
    random bit patterns, covering the full float space."""
    if bits >> 63:
        bits = bits ^ (1 << 63)
    else:
        bits = bits ^ ((1 << 64) - 1)
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


def _shortlex(s: str) -> Tuple[int, str]:
    """Shortlex key: shorter strings are simpler, then lexicographic."""
    return (len(s), s)


def _parse_float_string(value: float) -> Tuple[str, str, str, str]:
    """Parse a finite float's string representation into components.

    Returns (exp_part, frac_part, int_part, sign) where:
    - exp_part is the exponent string (e.g. "+20", "-10", or "")
    - frac_part is the fractional digits (e.g. "5", "0", or "")
    - int_part is the integer digits (e.g. "1", "100")
    - sign is "" for positive, "-" for negative
    """
    s = str(value)
    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:]
    if "e" in s:
        mantissa, exp_part = s.split("e")
    else:
        mantissa = s
        exp_part = ""
    if "." in mantissa:
        int_part, frac_part = mantissa.split(".")
    else:
        int_part = mantissa
        frac_part = ""
    return exp_part, frac_part, int_part, sign


def _float_string_key(value: float) -> Tuple:
    """Sort key for finite floats based on their string representation.

    Compares the (exponent, fractional, integer) triple in shortlex
    order, with positive preferred over negative. For the exponent,
    positive exponents are simpler than negative ones."""
    exp_part, frac_part, int_part, sign = _parse_float_string(value)
    # For the exponent, split sign from magnitude.
    if exp_part.startswith("-"):
        exp_sign = 1  # negative exponents are less simple
        exp_abs = exp_part[1:]
    elif exp_part.startswith("+"):
        exp_sign = 0
        exp_abs = exp_part[1:]
    else:
        exp_sign = 0
        exp_abs = exp_part
    exp_key = (_shortlex(exp_abs), exp_sign)
    frac_key = _shortlex(frac_part)
    int_key = _shortlex(int_part)
    # Prefer positive over negative (0 for positive, 1 for negative)
    sign_key = 0 if sign == "" else 1
    return (exp_key, frac_key, int_key, sign_key)


@dataclass(frozen=True)
class FloatChoice(ChoiceType[float]):
    min_value: float
    max_value: float
    allow_nan: bool
    allow_infinity: bool

    @property
    def simplest(self) -> float:
        # Prefer 0.0 if in range, otherwise the bound closest to 0.
        if self.min_value <= 0.0 <= self.max_value:
            return 0.0
        elif abs(self.min_value) <= abs(self.max_value):
            return self.min_value
        else:
            return self.max_value

    def validate(self, value: float) -> bool:
        if not isinstance(value, float):
            return False
        if math.isnan(value):
            return self.allow_nan
        if math.isinf(value):
            return self.allow_infinity
        return self.min_value <= value <= self.max_value

    def sort_key(self, value: float) -> Any:
        """Order floats by human-readable simplicity.

        Finite < inf < -inf < NaN. Among finite floats, we compare
        by their string representation split into (exponent, fractional,
        integer) parts in shortlex order. Positive is preferred to
        negative."""
        if math.isnan(value):
            return (3,)
        if math.isinf(value):
            return (1,) if value > 0 else (2,)
        return (0, _float_string_key(value))


@dataclass(frozen=True)
class ChoiceNode(Generic[U]):
    """A single choice made during test case generation. Each choice
    carries its kind (with constraints) and value, plus whether it
    was forced (i.e. deterministic, not random)."""

    kind: ChoiceType[U]
    value: U
    was_forced: bool

    def with_value(self, value: U) -> ChoiceNode[U]:
        """Return a copy of this node with a different value."""
        return ChoiceNode(kind=self.kind, value=value, was_forced=self.was_forced)

    @property
    def sort_key(self) -> Any:
        """A comparable key for this node's value, delegated to
        the choice type."""
        return self.kind.sort_key(self.value)


class Database(Protocol):
    def __setitem__(self, key: str, value: bytes) -> None: ...

    def get(self, key: str) -> Optional[bytes]: ...

    def __delitem__(self, key: str) -> None: ...


_TAG_INTEGER = 0
_TAG_BOOLEAN = 1
_TAG_BYTES = 2
_TAG_FLOAT = 3
_TAG_STRING = 4


def _serialize_choices(nodes: Sequence[ChoiceNode]) -> bytes:
    """Serialize a choice sequence to bytes for database storage."""
    parts: List[bytes] = []
    for n in nodes:
        if isinstance(n.kind, IntegerChoice):
            parts.append(bytes([_TAG_INTEGER]) + n.value.to_bytes(8, "big"))
        elif isinstance(n.kind, BooleanChoice):
            parts.append(bytes([_TAG_BOOLEAN, int(n.value)]))
        elif isinstance(n.kind, BytesChoice):
            parts.append(
                bytes([_TAG_BYTES]) + len(n.value).to_bytes(4, "big") + n.value
            )
        elif isinstance(n.kind, StringChoice):
            encoded = n.value.encode("utf-8")
            parts.append(
                bytes([_TAG_STRING]) + len(encoded).to_bytes(4, "big") + encoded
            )
        else:
            assert isinstance(n.kind, FloatChoice)
            parts.append(bytes([_TAG_FLOAT]) + struct.pack("!d", n.value))
    return b"".join(parts)


def _deserialize_choices(data: bytes) -> Optional[List]:
    """Deserialize a choice sequence from bytes. Returns None if
    the data is malformed (e.g. from an old format)."""
    values: List = []
    i = 0
    try:
        while i < len(data):
            tag = data[i]
            i += 1
            if tag == _TAG_INTEGER:
                if i + 8 > len(data):
                    return None
                values.append(int.from_bytes(data[i : i + 8], "big"))
                i += 8
            elif tag == _TAG_BOOLEAN:
                values.append(bool(data[i]))
                i += 1
            elif tag == _TAG_BYTES:
                if i + 4 > len(data):
                    return None
                length = int.from_bytes(data[i : i + 4], "big")
                i += 4
                if i + length > len(data):
                    return None
                values.append(data[i : i + length])
                i += length
            elif tag == _TAG_STRING:
                if i + 4 > len(data):
                    return None
                length = int.from_bytes(data[i : i + 4], "big")
                i += 4
                if i + length > len(data):
                    return None
                values.append(data[i : i + length].decode("utf-8"))
                i += length
            elif tag == _TAG_FLOAT:
                if i + 8 > len(data):
                    return None
                values.append(struct.unpack("!d", data[i : i + 8])[0])
                i += 8
            else:
                return None
    except (IndexError, ValueError):
        return None
    return values


def run_test(
    max_examples: int = 100,
    random: Optional[Random] = None,
    database: Optional[Database] = None,
    quiet: bool = False,
) -> Callable[[Callable[[TestCase], None]], None]:
    """Decorator to run a test. Usage is:

    .. code-block: python

        @run_test()
        def _(test_case):
            n = test_case.choice(1000)
            ...

    The decorated function takes a ``TestCase`` argument,
    and should raise an exception to indicate a test failure.
    It will either run silently or print drawn values and then
    fail with an exception if minithesis finds some test case
    that fails.

    The test will be run immediately, unlike in Hypothesis where
    @given wraps a function to expose it to the test runner.
    If you don't want it to be run immediately wrap it inside a
    test function yourself.

    Arguments:

    * max_examples: the maximum number of valid test cases to run for.
      Note that under some circumstances the test may run fewer test
      cases than this.
    * random: An instance of random.Random that will be used for all
      nondeterministic choices.
    * database: A dict-like object in which results will be cached and resumed
      from, ensuring that if a test is run twice it fails in the same way.
    * quiet: Will not print anything on failure if True.
    """

    def accept(test: Callable[[TestCase], None]) -> None:
        def mark_failures_interesting(test_case: TestCase) -> None:
            try:
                test(test_case)
            except Exception:
                if test_case.status is not None:
                    raise
                test_case.mark_status(Status.INTERESTING)

        state = TestingState(
            random or Random(), mark_failures_interesting, max_examples
        )

        if database is None:
            # If the database is not set, use a standard cache directory
            # location to persist examples.
            db: Database = DirectoryDB(_DEFAULT_DATABASE_PATH)
        else:
            db = database

        previous_failure = db.get(test.__name__)

        if previous_failure is not None:
            values = _deserialize_choices(previous_failure)
            if values is not None:
                state.test_function(TestCase.for_choices(values))

        if state.result is None:
            state.run()

        if state.valid_test_cases == 0:
            raise Unsatisfiable()

        if state.result is None:
            try:
                del db[test.__name__]
            except KeyError:
                pass
        else:
            db[test.__name__] = _serialize_choices(state.result)

        if state.result is not None:
            test(
                TestCase.for_choices(
                    [n.value for n in state.result], print_results=not quiet
                )
            )

    return accept


class TestCase:
    """Represents a single generated test case, which consists
    of an underlying sequence of typed choices."""

    @classmethod
    def for_choices(
        cls,
        choices: Sequence[Any],
        print_results: bool = False,
    ) -> TestCase:
        """Returns a test case that makes this series of choices."""
        return TestCase(
            prefix=choices,
            random=None,
            max_size=len(choices),
            print_results=print_results,
        )

    def __init__(
        self,
        prefix: Sequence[Any],
        random: Optional[Random],
        max_size: float = float("inf"),
        print_results: bool = False,
    ):
        self.prefix = prefix
        self._random = random
        self.max_size = max_size
        self.nodes: List[ChoiceNode] = []
        self.status: Optional[Status] = None
        self.print_results = print_results
        self.depth = 0
        self.targeting_score: Optional[int] = None

    def draw_integer(self, min_value: int, max_value: int) -> int:
        """Returns a number in the range [min_value, max_value]."""
        n = max_value - min_value
        if n.bit_length() > 64 or n < 0:
            raise ValueError(f"Invalid range [{min_value}, {max_value}]")
        return self.__make_choice(
            IntegerChoice(min_value, max_value),
            lambda: self.random.randint(min_value, max_value),
        )

    def draw_float(
        self,
        min_value: float = -math.inf,
        max_value: float = math.inf,
        *,
        allow_nan: bool = True,
        allow_infinity: bool = True,
    ) -> float:
        """Returns a random float in [min_value, max_value].

        NaN is disallowed whenever any bound is set (since NaN is
        not comparable). For bounded ranges, generates uniformly.
        For half-bounded ranges, generates uniformly in the finite
        portion and occasionally returns the infinite bound. For
        fully unbounded ranges, generates via random bit patterns."""
        # Disallow NaN when any bound is set, since NaN is not
        # comparable to numeric bounds.
        if min_value != -math.inf or max_value != math.inf:
            allow_nan = False
        kind = FloatChoice(min_value, max_value, allow_nan, allow_infinity)

        bounded = math.isfinite(min_value) and math.isfinite(max_value)
        half_bounded = not bounded and (
            math.isfinite(min_value) or math.isfinite(max_value)
        )

        if bounded:

            def generate() -> float:
                return self.random.uniform(min_value, max_value)

        elif half_bounded:
            # One bound is finite, the other is ±inf. Generate a
            # non-negative float and add/subtract from the bound.

            def generate() -> float:
                if allow_infinity and self.random.random() < 0.05:
                    return math.inf if max_value == math.inf else -math.inf
                magnitude = abs(_draw_unbounded_float(self.random))
                if math.isfinite(min_value):
                    return min_value + magnitude
                else:
                    return max_value - magnitude

        elif allow_nan:
            # Fully unbounded with NaN allowed. NaN is only ~0.05%
            # of bit patterns, so boost it to ~NAN_DRAW_PROBABILITY.
            def generate() -> float:
                if self.random.random() < NAN_DRAW_PROBABILITY:
                    return _draw_nan(self.random)
                return _draw_unbounded_float(self.random)

        else:
            # Fully unbounded, no NaN.
            def generate() -> float:
                return _draw_unbounded_float(self.random)

        return self.__make_choice(kind, generate)

    def draw_string(
        self,
        min_codepoint: int = 0,
        max_codepoint: int = 0x10FFFF,
        min_size: int = 0,
        max_size: int = 10,
    ) -> str:
        """Returns a random string with length in [min_size, max_size]
        and characters with codepoints in [min_codepoint, max_codepoint].
        Surrogates (0xD800-0xDFFF) are excluded."""
        # Compute the valid codepoint range excluding surrogates.
        if min_codepoint > max_codepoint:
            raise ValueError(
                f"Invalid codepoint range [{min_codepoint}, {max_codepoint}]"
            )
        kind = StringChoice(min_codepoint, max_codepoint, min_size, max_size)

        def generate() -> str:
            length = self.random.randint(min_size, max_size)
            chars: List[str] = []
            for _ in range(length):
                # Rejection-sample to avoid surrogates (0xD800-0xDFFF).
                while True:
                    cp = self.random.randint(min_codepoint, max_codepoint)
                    if not (0xD800 <= cp <= 0xDFFF):
                        break
                chars.append(chr(cp))
            return "".join(chars)

        return self.__make_choice(kind, generate)

    def draw_bytes(self, min_size: int, max_size: int) -> bytes:
        """Returns a random byte string with length in [min_size, max_size]."""
        return self.__make_choice(
            BytesChoice(min_size, max_size),
            lambda: bytes(
                self.random.randint(0, 255)
                for _ in range(self.random.randint(min_size, max_size))
            ),
        )

    def choice(self, n: int) -> int:
        """Returns a number in the range [0, n]"""
        result = self.draw_integer(0, n)
        if self.__should_print():
            print(f"choice({n}): {result}")
        return result

    def weighted(self, p: float, *, forced: Optional[bool] = None) -> bool:
        """Return True with probability ``p``. If ``forced`` is
        provided, the result is forced to that value (no randomness)."""
        if p <= 0:
            forced = False
        elif p >= 1:
            forced = True
        result = bool(
            self.__make_choice(
                BooleanChoice(p),
                lambda: self.random.random() <= p,
                forced=forced,
            )
        )
        if self.__should_print():
            print(f"weighted({p}): {result}")
        return result

    def forced_choice(self, n: int) -> int:
        """Inserts a forced integer choice into the choice sequence,
        as if some call to choice() had returned ``n``. You almost
        never need this, but sometimes it can be a useful hint to
        the shrinker."""
        if n.bit_length() > 64 or n < 0:
            raise ValueError(f"Invalid choice {n}")
        return self.__make_choice(
            IntegerChoice(0, n),
            lambda: n,
            forced=n,
        )

    def reject(self) -> NoReturn:
        """Mark this test case as invalid."""
        self.mark_status(Status.INVALID)

    def assume(self, precondition: bool) -> None:
        """If this precondition is not met, abort the test and
        mark this test case as invalid."""
        if not precondition:
            self.reject()

    def target(self, score: int) -> None:
        """Set a score to maximize. Multiple calls to this function
        will override previous ones.

        The name and idea come from Löscher, Andreas, and Konstantinos
        Sagonas. "Targeted property-based testing." ISSTA. 2017, but
        the implementation is based on that found in Hypothesis,
        which is not that similar to anything described in the paper.
        """
        self.targeting_score = score

    def any(self, generator: Generator[U]) -> U:
        """Return a value from ``generator``."""
        try:
            self.depth += 1
            result = generator.produce(self)
        finally:
            self.depth -= 1

        if self.__should_print():
            print(f"any({generator}): {result}")
        return result

    def mark_status(self, status: Status) -> NoReturn:
        """Set the status and raise StopTest."""
        if self.status is not None:
            raise Frozen()
        self.status = status
        raise StopTest()

    @property
    def random(self) -> Random:
        """The Random instance for this test case. Only available
        when the test case was created with a Random (not from
        for_choices)."""
        assert self._random is not None
        return self._random

    def __should_print(self) -> bool:
        return self.print_results and self.depth == 0

    def __make_choice(
        self,
        kind: ChoiceType[U],
        rnd_method: Callable[[], U],
        *,
        forced: Optional[U] = None,
    ) -> U:
        """Core method for recording a choice. Uses the forced value,
        prefix, or rnd_method, then validates against kind."""
        if self.status is not None:
            raise Frozen()
        if len(self.nodes) >= self.max_size:
            self.mark_status(Status.OVERRUN)
        if forced is not None:
            value = forced
        elif len(self.nodes) < len(self.prefix):
            value = self.prefix[len(self.nodes)]
        else:
            value = rnd_method()
        self.nodes.append(ChoiceNode(kind, value, forced is not None))
        if forced is None and not kind.validate(value):
            self.mark_status(Status.INVALID)
        return value


class Generator(Generic[T]):
    """Represents some range of values that might be used in
    a test, that can be requested from a ``TestCase``.

    Pass one of these to TestCase.any to get a concrete value.
    """

    def __init__(self, produce: Callable[[TestCase], T], name: Optional[str] = None):
        self.produce = produce
        self.name = produce.__name__ if name is None else name

    def __repr__(self) -> str:
        return self.name

    def map(self, f: Callable[[T], S]) -> Generator[S]:
        """Returns a ``Generator`` where values come from
        applying ``f`` to some possible value for ``self``."""
        return Generator(
            lambda test_case: f(test_case.any(self)),
            name=f"{self.name}.map({f.__name__})",
        )

    def flat_map(self, f: Callable[[T], Generator[S]]) -> Generator[S]:
        """Returns a ``Generator`` where values come from
        drawing a value from ``self``, passing it to ``f`` to
        get a new ``Generator``, then drawing from that."""

        def produce(test_case: TestCase) -> S:
            return test_case.any(f(test_case.any(self)))

        return Generator[S](
            produce,
            name=f"{self.name}.flat_map({f.__name__})",
        )

    def filter(self, f: Callable[[T], bool]) -> Generator[T]:
        """Returns a ``Generator`` whose values are drawn from
        ``self`` and satisfy ``f``. Retries up to 3 times,
        then rejects the test case."""

        def produce(test_case: TestCase) -> T:
            for _ in range(3):
                candidate = test_case.any(self)
                if f(candidate):
                    return candidate
            test_case.reject()

        return Generator[T](produce, name=f"{self.name}.filter({f.__name__})")


# We cap the maximum amount of entropy a test case can use.
# This prevents cases where the generated test case size explodes
# by effectively rejecting test cases that use too many choices.
BUFFER_SIZE = 8 * 1024

_DEFAULT_DATABASE_PATH = ".minithesis-cache"

NAN_DRAW_PROBABILITY = 0.01


def sort_key(nodes: Sequence[ChoiceNode]) -> Tuple:
    """Returns a key that can be used for the shrinking order
    of test cases. Shorter choice sequences are simpler, and
    among equal lengths we prefer smaller values.

    This comparison is safe because in a non-flaky test, the choice
    type at each position is determined by previous values, so two
    sequences always have the same type at the first index they differ."""
    return (len(nodes), [n.sort_key for n in nodes])


class CachedTestFunction:
    """Returns a cached version of a function that maps
    a choice sequence to the status of calling a test function
    on a test case populated with it. Is able to take advantage
    of the structure of the test function to predict the result
    even if exact sequence of choices has not been seen
    previously.

    You can safely omit implementing this at the cost of
    somewhat increased shrinking time.
    """

    def __init__(self, test_function: Callable[[TestCase], None]):
        self.test_function = test_function

        # Tree nodes are either a point at which a choice occurs
        # in which case they map the result of the choice to the
        # tree node we are in after, or a Status object indicating
        # mark_status was called at this point and all future
        # choices are irrelevant.
        #
        # Note that a better implementation of this would use
        # a Patricia trie, which implements long non-branching
        # paths as an array inline. For simplicity we don't
        # do that here.
        self.tree: Dict[Any, Union[Status, Dict[Any, Any]]] = {}

    def __call__(self, choices: Sequence[Any]) -> Status:
        node: Any = self.tree
        try:
            for c in choices:
                node = node[c]
                # mark_status was called, thus future choices
                # will be ignored.
                if isinstance(node, Status):
                    assert node != Status.OVERRUN
                    return node
            # If we never entered an unknown region of the tree
            # or hit a Status value, then we know that another
            # choice will be made next and the result will overrun.
            return Status.OVERRUN
        except KeyError:
            pass

        # We now have to actually call the test function to find out
        # what happens.
        test_case = TestCase.for_choices(choices)
        self.test_function(test_case)
        assert test_case.status is not None

        # We enter the choices made in a tree.
        node = self.tree
        for i, choice_node in enumerate(test_case.nodes):
            c = choice_node.value
            if i + 1 < len(test_case.nodes) or test_case.status == Status.OVERRUN:
                try:
                    node = node[c]
                except KeyError:
                    node = node.setdefault(c, {})
            else:
                node[c] = test_case.status
        return test_case.status


class TestingState:
    def __init__(
        self,
        random: Random,
        test_function: Callable[[TestCase], None],
        max_examples: int,
    ):
        self.random = random
        self.max_examples = max_examples
        self.__test_function = test_function
        self.valid_test_cases = 0
        self.calls = 0
        self.result: Optional[List[ChoiceNode]] = None
        self.best_scoring: Optional[Tuple[int, List[ChoiceNode]]] = None
        self.test_is_trivial = False

    def test_function(self, test_case: TestCase) -> None:
        try:
            self.__test_function(test_case)
        except StopTest:
            pass
        if test_case.status is None:
            test_case.status = Status.VALID
        self.calls += 1
        if test_case.status >= Status.INVALID and len(test_case.nodes) == 0:
            self.test_is_trivial = True
        if test_case.status >= Status.VALID:
            self.valid_test_cases += 1

            if test_case.targeting_score is not None:
                relevant_info = (test_case.targeting_score, test_case.nodes)
                if self.best_scoring is None:
                    self.best_scoring = relevant_info
                else:
                    best, _ = self.best_scoring
                    if test_case.targeting_score > best:
                        self.best_scoring = relevant_info

        if test_case.status == Status.INTERESTING and (
            self.result is None or sort_key(test_case.nodes) < sort_key(self.result)
        ):
            self.result = test_case.nodes

    def target(self) -> None:
        """If any test cases have had ``target()`` called on them, do a simple
        hill climbing algorithm to attempt to optimise that target score."""
        if self.result is not None or self.best_scoring is None:
            return

        def adjust(i: int, step: int) -> bool:
            """Can we improve the score by changing nodes[i] by ``step``?"""
            assert self.best_scoring is not None
            score, nodes = self.best_scoring
            if not isinstance(nodes[i].kind, IntegerChoice):
                return False
            if nodes[i].value + step < 0 or nodes[i].value.bit_length() >= 64:
                return False
            values = [n.value for n in nodes]
            values[i] += step
            test_case = TestCase(
                prefix=values, random=self.random, max_size=BUFFER_SIZE
            )
            self.test_function(test_case)
            assert test_case.status is not None
            return (
                test_case.status >= Status.VALID
                and test_case.targeting_score is not None
                and test_case.targeting_score > score
            )

        while self.should_keep_generating():
            i = self.random.randrange(0, len(self.best_scoring[1]))
            sign = 0
            for k in [1, -1]:
                if not self.should_keep_generating():
                    return
                if adjust(i, k):
                    sign = k
                    break
            if sign == 0:
                continue

            k = 1
            while self.should_keep_generating() and adjust(i, sign * k):
                k *= 2

            while k > 0:
                while self.should_keep_generating() and adjust(i, sign * k):
                    pass
                k //= 2

    def run(self) -> None:
        self.generate()
        self.target()
        self.shrink()

    def should_keep_generating(self) -> bool:
        return (
            not self.test_is_trivial
            and self.result is None
            and self.valid_test_cases < self.max_examples
            and
            # We impose a limit on the maximum number of calls as
            # well as the maximum number of valid examples. This is
            # to avoid taking a prohibitively long time on tests which
            # have hard or impossible to satisfy preconditions.
            self.calls < self.max_examples * 10
        )

    def generate(self) -> None:
        """Run random generation until either we have found an interesting
        test case or hit the limit of how many test cases we should
        evaluate."""
        while self.should_keep_generating() and (
            self.best_scoring is None or self.valid_test_cases <= self.max_examples // 2
        ):
            self.test_function(
                TestCase(prefix=(), random=self.random, max_size=BUFFER_SIZE)
            )

    def shrink(self) -> None:
        """If we have found an interesting example, try shrinking it
        so that the choice sequence leading to our best example is
        shortlex smaller than the one we originally found. This improves
        the quality of the generated test case, as per our paper.

        https://drmaciver.github.io/papers/reduction-via-generation-preview.pdf
        """
        if not self.result:
            return

        # Shrinking will typically try the same choice sequences over
        # and over again, so we cache the test function in order to
        # not end up reevaluating it in those cases. This also allows
        # us to catch cases where we try something that is e.g. a prefix
        # of something we've previously tried, which is guaranteed
        # not to work.
        cached = CachedTestFunction(self.test_function)

        def consider(nodes: List[ChoiceNode]) -> bool:
            assert self.result is not None
            if sort_key(nodes) == sort_key(self.result):
                return True
            return cached([n.value for n in nodes]) == Status.INTERESTING

        assert consider(self.result)

        # We are going to perform a number of transformations to
        # the current result, iterating until none of them make any
        # progress - i.e. until we make it through an entire iteration
        # of the loop without changing the result.
        prev = None
        while prev != self.result:
            prev = self.result

            # A note on weird loop order: We iterate backwards
            # through the choice sequence rather than forwards,
            # because later bits tend to depend on earlier bits
            # so it's easier to make changes near the end and
            # deleting bits at the end may allow us to make
            # changes earlier on that we we'd have missed.
            #
            # Note that we do not restart the loop at the end
            # when we find a successful shrink. This is because
            # things we've already tried are less likely to work.
            #
            # If this guess is wrong, that's OK, this isn't a
            # correctness problem, because if we made a successful
            # reduction then we are not at a fixed point and
            # will restart the loop at the end the next time
            # round. In some cases this can result in performance
            # issues, but the end result should still be fine.

            # First try deleting each choice we made in chunks.
            # We try longer chunks because this allows us to
            # delete whole composite elements: e.g. deleting an
            # element from a generated list requires us to delete
            # both the choice of whether to include it and also
            # the element itself, which may involve more than one
            # choice. Some things will take more than 8 choices
            # in the sequence. That's too bad, we may not be
            # able to delete those. In Hypothesis proper we
            # record the boundaries corresponding to ``any``
            # calls so that we can try deleting those, but
            # that's pretty high overhead and also a bunch of
            # slightly annoying code that it's not worth porting.
            #
            # We could instead do a quadratic amount of work
            # to try all boundaries, but in general we don't
            # want to do that because even a shrunk test case
            # can involve a relatively large number of choices.
            k = 8
            while k > 0:
                i = len(self.result) - k - 1
                while i >= 0:
                    if i >= len(self.result):
                        # Can happen if we successfully lowered
                        # the value at i - 1
                        i -= 1
                        continue
                    attempt = self.result[:i] + self.result[i + k :]
                    assert len(attempt) < len(self.result)
                    if not consider(attempt):
                        # This fixes a common problem that occurs
                        # when you have dependencies on some
                        # length parameter. e.g. draw a number
                        # between 0 and 10 and then draw that
                        # many elements. This can't delete
                        # everything that occurs that way, but
                        # it can delete some things and often
                        # will get us unstuck when nothing else
                        # does.
                        if (
                            i > 0
                            and attempt[i - 1].value != attempt[i - 1].kind.simplest
                        ):
                            attempt = list(attempt)
                            attempt[i - 1] = attempt[i - 1].with_value(
                                attempt[i - 1].value - 1
                            )
                            if consider(attempt):
                                i += 1
                        i -= 1
                k -= 1

            def replace(values: Mapping[int, Any]) -> bool:
                """Attempts to replace some node values in the current
                result. Useful for some purely lexicographic
                reductions that we are about to perform."""
                assert self.result is not None
                attempt = list(self.result)
                for i, v in values.items():
                    # The size of self.result can change during shrinking.
                    # If that happens, stop attempting to make use of these
                    # replacements because some other shrink pass is better
                    # to run now.
                    if i >= len(attempt):
                        return False
                    attempt[i] = attempt[i].with_value(v)
                return consider(attempt)

            # Now we try replacing blocks of choices with their
            # simplest values. Note that unlike the above we skip
            # k = 1 because we handle that in the next step. Often
            # (but not always) the simplest values are the shortlex
            # smallest that a region can be.
            k = 8

            while k > 1:
                i = len(self.result) - k
                while i >= 0:
                    if replace(
                        {j: self.result[j].kind.simplest for j in range(i, i + k)}
                    ):
                        # If we've succeeded then all of [i, i + k]
                        # is at simplest so we adjust i so that the
                        # next region does not overlap with this at all.
                        i -= k
                    else:
                        # Otherwise we might still be able to simplify
                        # some of these values but not the last one,
                        # so we just go back one.
                        i -= 1
                k -= 1

            # Now try replacing each choice with a smaller value,
            # using type-appropriate strategies.
            i = len(self.result) - 1
            while i >= 0:
                node = self.result[i]
                if isinstance(node.kind, BooleanChoice):
                    # Booleans: just try False.
                    replace({i: False})
                elif isinstance(node.kind, IntegerChoice):
                    # Integers: binary search toward min_value.
                    bin_search_down(
                        node.kind.min_value,
                        node.value,
                        lambda v: replace({i: v}),
                    )
                elif isinstance(node.kind, FloatChoice):
                    self.__shrink_float(i, node.kind, replace)
                elif isinstance(node.kind, StringChoice):
                    self.__shrink_sequence(
                        i,
                        node.kind.min_size,
                        node.kind.simplest,
                        lambda v, j: ord(v[j]),
                        lambda v, j, e: v[:j] + chr(e) + v[j + 1 :],
                        node.kind.min_codepoint,
                        replace,
                    )
                else:
                    assert isinstance(node.kind, BytesChoice)
                    self.__shrink_sequence(
                        i,
                        node.kind.min_size,
                        node.kind.simplest,
                        lambda v, j: v[j],
                        lambda v, j, e: v[:j] + bytes([e]) + v[j + 1 :],
                        0,
                        replace,
                    )
                i -= 1

            # NB from here on this is just showing off cool shrinker tricks and
            # you probably don't need to worry about it and can skip these bits
            # unless they're easy and you want bragging rights for how much
            # better you are at shrinking than the local QuickCheck equivalent.

            # Try sorting out of order ranges of integer choices,
            # as ``sort(x) <= x`` so this is always a lexicographic
            # reduction. We only sort ranges that are all integers
            # since sorting mixed types is meaningless.
            k = 8
            while k > 1:
                for i in range(len(self.result) - k - 1, -1, -1):
                    region = self.result[i : i + k]
                    if not all(isinstance(n.kind, IntegerChoice) for n in region):
                        continue
                    consider(
                        self.result[:i]
                        + sorted(region, key=lambda n: n.value)
                        + self.result[i + k :]
                    )
                k -= 1

            # Try adjusting nearby pairs of integer choices by
            # redistributing value between them. This is useful for
            # tests that depend on the sum of some generated values.
            for k in [2, 1]:
                for i in range(len(self.result) - 1 - k, -1, -1):
                    j = i + k
                    # In theory a swap could shorten self.result,
                    # putting j out of bounds. In practice the
                    # zeroing pass always finds such shortenings
                    # first, so this is just a safety check.
                    assert j < len(self.result)
                    kind_i = self.result[i].kind
                    kind_j = self.result[j].kind
                    if not isinstance(kind_i, IntegerChoice) or not isinstance(
                        kind_j, IntegerChoice
                    ):
                        continue
                    # Try swapping out of order pairs
                    if self.result[i].value > self.result[j].value:
                        replace(
                            {
                                j: self.result[i].value,
                                i: self.result[j].value,
                            }
                        )
                    if j < len(self.result) and self.result[i].value > kind_i.min_value:
                        previous_i = self.result[i].value
                        previous_j = self.result[j].value
                        bin_search_down(
                            kind_i.min_value,
                            previous_i,
                            lambda v: replace({i: v, j: previous_j + (previous_i - v)}),
                        )

    def __shrink_float(
        self,
        i: int,
        kind: FloatChoice,
        replace: Callable[[Mapping[int, Any]], bool],
    ) -> None:
        """Shrink a float choice at index i toward human-readable
        simplicity.

        1. Replace special values (NaN → inf → finite)
        2. Try range edges
        3. If negative, try flipping sign
        4. Shrink string representation parts (exponent, fractional,
           integer) as integers
        """
        assert self.result is not None
        value = self.result[i].value

        def try_float(f: float) -> bool:
            if kind.validate(f):
                return replace({i: f})
            return False

        # Step 1: Replace special values with simpler ones.
        if math.isnan(value):
            for v in [math.inf, -math.inf, 0.0]:
                if try_float(v):
                    return
            return
        if math.isinf(value):
            if value < 0:
                try_float(math.inf)
                value = self.result[i].value
            assert math.isinf(value)
            try_float(sys.float_info.max if value > 0 else -sys.float_info.max)
            value = self.result[i].value

        # Step 2: Try range edges.
        if math.isfinite(kind.min_value):
            try_float(kind.min_value)
        if math.isfinite(kind.max_value):
            try_float(kind.max_value)
        value = self.result[i].value

        # Step 3: If negative, try flipping sign.
        if value < 0:
            try_float(-value)
            value = self.result[i].value

        if not math.isfinite(value):
            return

        # Step 4: Shrink string parts as integers. For negative
        # values (in forced-negative ranges), negate before parsing
        # and negate back when trying replacements.
        negate = value < 0
        if negate:
            value = -value

        def try_positive(f: float) -> bool:
            return try_float(-f if negate else f)

        exp_part, frac_part, int_part, _ = _parse_float_string(value)
        if exp_part:
            exp_abs = exp_part.lstrip("+-")
            exp_sign = exp_part[0] if exp_part[0] in "+-" else ""
            if exp_sign == "-":
                try_positive(float(f"{int_part}.{frac_part}e{exp_abs}"))
            assert exp_abs  # Python's str() always has digits after 'e'
            bin_search_down(
                0,
                int(exp_abs),
                lambda e: try_positive(
                    float(f"{int_part}.{frac_part}e{exp_sign}{e}")
                    if e > 0
                    else float(f"{int_part}.{frac_part}")
                ),
            )
            value = abs(self.result[i].value)
            exp_part, frac_part, int_part, _ = _parse_float_string(value)

        if frac_part and frac_part != "0":
            reversed_frac = int(frac_part[::-1])
            bin_search_down(
                0,
                reversed_frac,
                lambda rf: try_positive(float(f"{int_part}.{str(rf)[::-1]}")),
            )
            value = abs(self.result[i].value)
            exp_part, frac_part, int_part, _ = _parse_float_string(value)

        if int_part and int(int_part) > 0:
            bin_search_down(
                0,
                int(int_part),
                lambda i_val: try_positive(
                    float(f"{i_val}.{frac_part}") if frac_part else float(i_val)
                ),
            )

    def __shrink_sequence(
        self,
        i: int,
        min_size: int,
        simplest: Any,
        element_value: Callable[[Any, int], int],
        replace_element: Callable[[Any, int, int], Any],
        min_element: int,
        replace: Callable[[Mapping[int, Any]], bool],
    ) -> None:
        """Shrink a sequence choice (string or bytes) at index i.

        Tries: simplest value, then shorten, then remove individual
        elements, then shrink each element value toward min_element.

        element_value: extract an integer from element j of sequence v
        replace_element: build a new sequence with element j replaced
        """
        if replace({i: simplest}):
            return
        assert self.result is not None
        cur = self.result[i].value
        bin_search_down(
            min_size,
            len(cur),
            lambda sz: replace({i: cur[:sz]}),
        )
        assert self.result is not None
        for j in range(len(self.result[i].value) - 1, -1, -1):
            assert self.result is not None
            v = self.result[i].value
            if j < len(v) and len(v) > min_size:
                replace({i: v[:j] + v[j + 1 :]})
        assert self.result is not None
        for j in range(len(self.result[i].value) - 1, -1, -1):
            assert self.result is not None
            v = self.result[i].value
            if j < len(v) and element_value(v, j) > min_element:
                result = self.result
                assert result is not None
                bin_search_down(
                    min_element,
                    element_value(v, j),
                    lambda e: replace({i: replace_element(result[i].value, j, e)}),
                )


def bin_search_down(lo: int, hi: int, f: Callable[[int], bool]) -> int:
    """Returns n in [lo, hi] such that f(n) is True,
    where it is assumed and will not be checked that
    f(hi) is True.

    Will return ``lo`` if ``f(lo)`` is True, otherwise
    the only guarantee that is made is that ``f(n - 1)``
    is False and ``f(n)`` is True. In particular this
    does *not* guarantee to find the smallest value,
    only a locally minimal one.
    """
    if f(lo):
        return lo
    # Try subtracting powers of 10 and dividing by 10. This helps
    # when the value has structure at decimal boundaries (e.g.
    # reversed fractional parts of floats).
    k = 1
    while k < hi:
        k *= 10
    while k >= 10:
        k //= 10
        if hi - k >= lo and f(hi - k):
            hi -= k
    if hi // 10 >= lo and f(hi // 10):
        hi = hi // 10
    while lo + 1 < hi:
        mid = lo + (hi - lo) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class DirectoryDB:
    """A very basic key/value store that just uses a file system
    directory to store values. You absolutely don't have to copy this
    and should feel free to use a more reasonable key/value store
    if you have easy access to one."""

    def __init__(self, directory: str):
        self.directory = directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

    def __to_file(self, key: str) -> str:
        return os.path.join(
            self.directory, hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        )

    def __setitem__(self, key: str, value: bytes) -> None:
        with open(self.__to_file(key), "wb") as o:
            o.write(value)

    def get(self, key: str) -> Optional[bytes]:
        f = self.__to_file(key)
        if not os.path.exists(f):
            return None
        with open(f, "rb") as i:
            return i.read()

    def __delitem__(self, key: str) -> None:
        try:
            os.unlink(self.__to_file(key))
        except FileNotFoundError:
            raise KeyError()


class Frozen(Exception):
    """Attempted to make choices on a test case that has been
    completed."""


class StopTest(Exception):
    """Raised when a test should stop executing early."""


class Unsatisfiable(Exception):
    """Raised when a test has no valid examples."""


class Status(IntEnum):
    # Test case didn't have enough data to complete
    OVERRUN = 0

    # Test case contained something that prevented completion
    INVALID = 1

    # Test case completed just fine but was boring
    VALID = 2

    # Test case completed and was interesting
    INTERESTING = 3
