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

This module is the standalone core that knows about integers and
booleans only. Float, bytes, and string support is added by the
package's __init__.py module.


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
import os
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


class SerializationTag(IntEnum):
    INTEGER = 0
    BOOLEAN = 1
    BYTES = 2
    FLOAT = 3
    STRING = 4


# Serialization registry. Type modules register their serializers
# so the database can persist and replay choice sequences.
_SERIALIZERS: Dict[type, Tuple[int, Callable[[Any], bytes]]] = {}
_DESERIALIZERS: Dict[int, Callable[[bytes, int], Tuple[Any, int]]] = {}


def register_serializer(
    choice_type: type,
    tag: int,
    serialize: Callable[[Any], bytes],
    deserialize: Callable[[bytes, int], Tuple[Any, int]],
) -> None:
    """Register serialization for a ChoiceType subclass.

    serialize(value) -> bytes
    deserialize(data, offset) -> (value, new_offset)
        Should raise IndexError or ValueError on truncated data."""
    _SERIALIZERS[choice_type] = (tag, serialize)
    _DESERIALIZERS[tag] = deserialize


def _serialize_choices(nodes: Sequence[ChoiceNode]) -> bytes:
    """Serialize a choice sequence to bytes for database storage."""
    parts: List[bytes] = []
    for n in nodes:
        tag, serialize = _SERIALIZERS[type(n.kind)]
        parts.append(bytes([tag]) + serialize(n.value))
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
            if tag not in _DESERIALIZERS:
                return None
            value, i = _DESERIALIZERS[tag](data, i)
            values.append(value)
    except (IndexError, ValueError):
        return None
    return values


def _deserialize_fixed(size: int, convert: Callable[[bytes], Any]) -> Callable:
    """Helper to create a deserializer for fixed-size values."""

    def deserialize(data: bytes, offset: int) -> Tuple[Any, int]:
        if offset + size > len(data):
            raise ValueError("truncated")
        return convert(data[offset : offset + size]), offset + size

    return deserialize


def _deserialize_length_prefixed(
    convert: Callable[[bytes], Any],
) -> Callable:
    """Helper to create a deserializer for length-prefixed values."""

    def deserialize(data: bytes, offset: int) -> Tuple[Any, int]:
        if offset + 4 > len(data):
            raise ValueError("truncated")
        length = int.from_bytes(data[offset : offset + 4], "big")
        offset += 4
        if offset + length > len(data):
            raise ValueError("truncated")
        return convert(data[offset : offset + length]), offset + length

    return deserialize


# Register core types.
register_serializer(
    IntegerChoice,
    SerializationTag.INTEGER,
    lambda v: v.to_bytes(8, "big"),
    _deserialize_fixed(8, lambda b: int.from_bytes(b, "big")),
)

register_serializer(
    BooleanChoice,
    SerializationTag.BOOLEAN,
    lambda v: bytes([int(v)]),
    _deserialize_fixed(1, lambda b: bool(b[0])),
)


# We cap the maximum amount of entropy a test case can use.
# This prevents cases where the generated test case size explodes
# by effectively rejecting test cases that use too many choices.
BUFFER_SIZE = 8 * 1024

_DEFAULT_DATABASE_PATH = ".minithesis-cache"


def run_test(
    max_examples: int = 100,
    random: Optional[Random] = None,
    database: Optional[Database] = None,
    quiet: bool = False,
) -> Callable[[Callable[["TestCase"], None]], None]:
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
        return self._make_choice(
            IntegerChoice(min_value, max_value),
            lambda: self.random.randint(min_value, max_value),
        )

    # draw_float, draw_bytes, draw_string are added by minithesis.__init__
    # when it is imported. These stubs exist only for type-checking.
    def draw_float(
        self,
        min_value: float = -float("inf"),
        max_value: float = float("inf"),
        *,
        allow_nan: bool = True,
        allow_infinity: bool = True,
    ) -> float:
        """Returns a random float. Implemented in minithesis.__init__."""
        raise NotImplementedError("import minithesis to use draw_float")

    def draw_string(
        self,
        min_codepoint: int = 0,
        max_codepoint: int = 0x10FFFF,
        min_size: int = 0,
        max_size: int = 10,
    ) -> str:
        """Returns a random string. Implemented in minithesis.__init__."""
        raise NotImplementedError("import minithesis to use draw_string")

    def draw_bytes(self, min_size: int, max_size: int) -> bytes:
        """Returns a random byte string. Implemented in minithesis.__init__."""
        raise NotImplementedError("import minithesis to use draw_bytes")

    def choice(self, n: int) -> int:
        """Returns a number in the range [0, n]"""
        result = self.draw_integer(0, n)
        if self._should_print():
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
            self._make_choice(
                BooleanChoice(p),
                lambda: self.random.random() <= p,
                forced=forced,
            )
        )
        if self._should_print():
            print(f"weighted({p}): {result}")
        return result

    def forced_choice(self, n: int) -> int:
        """Inserts a forced integer choice into the choice sequence,
        as if some call to choice() had returned ``n``. You almost
        never need this, but sometimes it can be a useful hint to
        the shrinker."""
        if n.bit_length() > 64 or n < 0:
            raise ValueError(f"Invalid choice {n}")
        return self._make_choice(
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

        if self._should_print():
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

    def _should_print(self) -> bool:
        return self.print_results and self.depth == 0

    def _make_choice(
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


# Shrink pass registry. Each pass is a function taking a
# TestingState and attempting to simplify state.result.
# Passes are run in order, repeating until a fixed point.
SHRINK_PASSES: List[Callable[["TestingState"], None]] = []


def shrink_pass(
    fn: Callable[["TestingState"], None],
) -> Callable[["TestingState"], None]:
    """Decorator that registers a function as a shrink pass."""
    SHRINK_PASSES.append(fn)
    return fn


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
        self._cached = CachedTestFunction(self.test_function)
        assert self.consider(self.result)

        # Run registered shrink passes repeatedly until none of
        # them make any progress.
        prev = None
        while prev != self.result:
            prev = self.result
            for pass_fn in SHRINK_PASSES:
                pass_fn(self)

    def consider(self, nodes: List[ChoiceNode]) -> bool:
        """Test whether a choice sequence is interesting."""
        assert self.result is not None
        if sort_key(nodes) == sort_key(self.result):
            return True
        return self._cached([n.value for n in nodes]) == Status.INTERESTING

    def replace(self, values: Mapping[int, Any]) -> bool:
        """Attempt to replace node values at given indices."""
        assert self.result is not None
        attempt = list(self.result)
        for i, v in values.items():
            if i >= len(attempt):
                return False
            attempt[i] = attempt[i].with_value(v)
        return self.consider(attempt)


@shrink_pass
def delete_chunks(state: TestingState) -> None:
    """Try deleting chunks of choices from the sequence.

    We try longer chunks because this allows us to delete whole
    composite elements: e.g. deleting an element from a generated
    list requires us to delete both the choice of whether to include
    it and also the element itself, which may involve more than one
    choice.

    We iterate backwards because later bits tend to depend on earlier
    bits, so it's easier to make changes near the end."""
    assert state.result is not None
    k = 8
    while k > 0:
        i = len(state.result) - k - 1
        while i >= 0:
            if i >= len(state.result):
                i -= 1
                continue
            attempt = state.result[:i] + state.result[i + k :]
            assert len(attempt) < len(state.result)
            if not state.consider(attempt):
                if i > 0 and attempt[i - 1].value != attempt[i - 1].kind.simplest:
                    attempt = list(attempt)
                    attempt[i - 1] = attempt[i - 1].with_value(attempt[i - 1].value - 1)
                    if state.consider(attempt):
                        i += 1
                i -= 1
        k -= 1


@shrink_pass
def zero_choices(state: TestingState) -> None:
    """Replace blocks of choices with their simplest values.
    Skip k=1 because we handle that in the per-choice pass."""
    assert state.result is not None
    k = 8
    while k > 1:
        i = len(state.result) - k
        while i >= 0:
            if state.replace(
                {j: state.result[j].kind.simplest for j in range(i, i + k)}
            ):
                i -= k
            else:
                i -= 1
        k -= 1


@shrink_pass
def shrink_individual_integers(state: TestingState) -> None:
    """Binary search each integer choice toward its min_value."""
    assert state.result is not None
    i = len(state.result) - 1
    while i >= 0:
        node = state.result[i]
        if isinstance(node.kind, IntegerChoice):
            bin_search_down(
                node.kind.min_value,
                node.value,
                lambda v: state.replace({i: v}),
            )
        i -= 1


@shrink_pass
def shrink_individual_booleans(state: TestingState) -> None:
    """Try replacing each boolean choice with False."""
    assert state.result is not None
    i = len(state.result) - 1
    while i >= 0:
        node = state.result[i]
        if isinstance(node.kind, BooleanChoice):
            state.replace({i: False})
        i -= 1


@shrink_pass
def sort_integer_ranges(state: TestingState) -> None:
    """Try sorting out of order ranges of integer choices.
    sort(x) <= x, so this is always a lexicographic reduction."""
    assert state.result is not None
    k = 8
    while k > 1:
        for i in range(len(state.result) - k - 1, -1, -1):
            region = state.result[i : i + k]
            if not all(isinstance(n.kind, IntegerChoice) for n in region):
                continue
            state.consider(
                state.result[:i]
                + sorted(region, key=lambda n: n.value)
                + state.result[i + k :]
            )
        k -= 1


@shrink_pass
def redistribute_integers(state: TestingState) -> None:
    """Try adjusting nearby pairs of integer choices by
    redistributing value between them. Useful for tests that
    depend on the sum of some generated values."""
    assert state.result is not None
    for k in [2, 1]:
        for i in range(len(state.result) - 1 - k, -1, -1):
            j = i + k
            assert j < len(state.result)
            kind_i = state.result[i].kind
            kind_j = state.result[j].kind
            if not isinstance(kind_i, IntegerChoice) or not isinstance(
                kind_j, IntegerChoice
            ):
                continue
            if state.result[i].value > state.result[j].value:
                state.replace(
                    {
                        j: state.result[i].value,
                        i: state.result[j].value,
                    }
                )
            if j < len(state.result) and state.result[i].value > kind_i.min_value:
                previous_i = state.result[i].value
                previous_j = state.result[j].value
                bin_search_down(
                    kind_i.min_value,
                    previous_i,
                    lambda v: state.replace({i: v, j: previous_j + (previous_i - v)}),
                )


def _shrink_sequence(
    value: Any,
    min_size: int,
    simplest: Any,
    element_value: Callable[[Any, int], int],
    replace_element: Callable[[Any, int, int], Any],
    min_element: int,
    try_replace: Callable[[Any], bool],
) -> None:
    """Shrink a sequence choice (string or bytes).

    Tries: simplest value, then shorten, then remove individual
    elements, then shrink each element value toward min_element.

    element_value: extract an integer from element j of sequence v
    replace_element: build a new sequence with element j replaced
    """
    # We track the current value locally. When try_replace succeeds
    # with a new value v, we update our local tracking to v.
    current = [value]

    def tracking_replace(v: Any) -> bool:
        if try_replace(v):
            current[0] = v
            return True
        return False

    if tracking_replace(simplest):
        return
    cur = current[0]
    bin_search_down(
        min_size,
        len(cur),
        lambda sz: tracking_replace(cur[:sz]),
    )
    for j in range(len(current[0]) - 1, -1, -1):
        v = current[0]
        if j < len(v) and len(v) > min_size:
            tracking_replace(v[:j] + v[j + 1 :])
    for j in range(len(current[0]) - 1, -1, -1):
        v = current[0]
        if j < len(v) and element_value(v, j) > min_element:
            bin_search_down(
                min_element,
                element_value(v, j),
                lambda e: tracking_replace(replace_element(current[0], j, e)),
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
