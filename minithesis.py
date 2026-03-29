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
3. A small library of primitive possibilities (generators) and combinators.
4. A Test case database for replay between runs.
5. Targeted property-based testing
6. A caching layer for mapping choice sequences to outcomes


Anything that supports 1 and 2 is a reasonable good first porting
goal. You'll probably want to port most of the possibilities library
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


def _serialize_choices(nodes: Sequence[ChoiceNode]) -> bytes:
    """Serialize a choice sequence to bytes for database storage."""
    parts: List[bytes] = []
    for n in nodes:
        if isinstance(n.kind, IntegerChoice):
            parts.append(bytes([_TAG_INTEGER]) + n.value.to_bytes(8, "big"))
        elif isinstance(n.kind, BooleanChoice):
            parts.append(bytes([_TAG_BOOLEAN, int(n.value)]))
        else:
            assert isinstance(n.kind, BytesChoice)
            parts.append(
                bytes([_TAG_BYTES]) + len(n.value).to_bytes(4, "big") + n.value
            )
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
                values.append(int.from_bytes(data[i : i + 8], "big"))
                i += 8
            elif tag == _TAG_BOOLEAN:
                values.append(bool(data[i]))
                i += 1
            elif tag == _TAG_BYTES:
                length = int.from_bytes(data[i : i + 4], "big")
                i += 4
                values.append(data[i : i + length])
                i += length
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
    @given wraps a function to expose it to the the test runner.
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
            db: Database = DirectoryDB(".minithesis-cache")
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


class TestCase(object):
    """Represents a single generated test case, which consists
    of an underlying sequence of typed choices that produce
    possibilities."""

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

    def any(self, possibility: Possibility[U]) -> U:
        """Return a possible value from ``possibility``."""
        try:
            self.depth += 1
            result = possibility.produce(self)
        finally:
            self.depth -= 1

        if self.__should_print():
            print(f"any({possibility}): {result}")
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


class Possibility(Generic[T]):
    """Represents some range of values that might be used in
    a test, that can be requested from a ``TestCase``.

    Pass one of these to TestCase.any to get a concrete value.
    """

    def __init__(self, produce: Callable[[TestCase], T], name: Optional[str] = None):
        self.produce = produce
        self.name = produce.__name__ if name is None else name

    def __repr__(self) -> str:
        return self.name

    def map(self, f: Callable[[T], S]) -> Possibility[S]:
        """Returns a ``Possibility`` where values come from
        applying ``f`` to some possible value for ``self``."""
        return Possibility(
            lambda test_case: f(test_case.any(self)),
            name=f"{self.name}.map({f.__name__})",
        )

    def bind(self, f: Callable[[T], Possibility[S]]) -> Possibility[S]:
        """Returns a ``Possibility`` where values come from
        applying ``f`` (which should return a new ``Possibility``
        to some possible value for ``self`` then returning a possible
        value from that."""

        def produce(test_case: TestCase) -> S:
            return test_case.any(f(test_case.any(self)))

        return Possibility[S](
            produce,
            name=f"{self.name}.bind({f.__name__})",
        )

    def satisfying(self, f: Callable[[T], bool]) -> Possibility[T]:
        """Returns a ``Possibility`` whose values are any possible
        value of ``self`` for which ``f`` returns True."""

        def produce(test_case: TestCase) -> T:
            for _ in range(3):
                candidate = test_case.any(self)
                if f(candidate):
                    return candidate
            test_case.reject()

        return Possibility[T](produce, name=f"{self.name}.select({f.__name__})")


def integers(m: int, n: int) -> Possibility[int]:
    """Any integer in the range [m, n] is possible"""
    return Possibility(lambda tc: tc.draw_integer(m, n), name=f"integers({m}, {n})")


def binary(min_size: int = 0, max_size: int = 8) -> Possibility[bytes]:
    """Any byte string with length in [min_size, max_size] is possible."""
    return Possibility(
        lambda tc: tc.draw_bytes(min_size, max_size),
        name=f"binary({min_size}, {max_size})",
    )


def lists(
    elements: Possibility[U],
    min_size: int = 0,
    max_size: float = float("inf"),
) -> Possibility[List[U]]:
    """Any lists whose elements are possible values from ``elements`` are possible."""

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

    return Possibility[List[U]](produce, name=f"lists({elements.name})")


def just(value: U) -> Possibility[U]:
    """Only ``value`` is possible."""
    return Possibility[U](lambda tc: value, name=f"just({value})")


def nothing() -> Possibility[NoReturn]:
    """No possible values. i.e. Any call to ``any`` will reject
    the test case."""

    def produce(tc: TestCase) -> NoReturn:
        tc.reject()

    return Possibility(produce)


def mix_of(*possibilities: Possibility[T]) -> Possibility[T]:
    """Possible values can be any value possible for one of ``possibilities``."""
    if not possibilities:
        return nothing()
    return Possibility(
        lambda tc: tc.any(possibilities[tc.choice(len(possibilities) - 1)]),
        name="mix_of({', '.join(p.name for p in possibilities)})",
    )


def tuples(*possibilities: Possibility[Any]) -> Possibility[Any]:
    """Any tuple t of of length len(possibilities) such that t[i] is possible
    for possibilities[i] is possible."""
    return Possibility(
        lambda tc: tuple(tc.any(p) for p in possibilities),
        name="tuples({', '.join(p.name for p in possibilities)})",
    )


# We cap the maximum amount of entropy a test case can use.
# This prevents cases where the generated test case size explodes
# by effectively rejection
BUFFER_SIZE = 8 * 1024


def sort_key(nodes: Sequence[ChoiceNode]) -> Tuple:
    """Returns a key that can be used for the shrinking order
    of test cases. Shorter choice sequences are simpler, and
    among equal lengths we prefer smaller values.

    This comparison is safe because in a non-flaky test, the choice
    type at each position is determined by previous values, so two
    sequences always have the same type at the first index they differ."""
    return (len(nodes), [n.sort_key for n in nodes])


class CachedTestFunction(object):
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


class TestingState(object):
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
                        # is zero so we adjust i so that the next region
                        # does not overlap with this at all.
                        i -= k
                    else:
                        # Otherwise we might still be able to zero some
                        # of these values but not the last one, so we
                        # just go back one.
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
                else:
                    assert isinstance(node.kind, BytesChoice)
                    # Bytes: try simplest, then shorten, then
                    # remove individual bytes, then shrink each
                    # byte value toward 0.
                    kind = node.kind
                    if not replace({i: kind.simplest}):
                        result = self.result
                        assert result is not None
                        cur = result[i].value
                        bin_search_down(
                            kind.min_size,
                            len(cur),
                            lambda sz: replace({i: cur[:sz]}),
                        )
                        result = self.result
                        assert result is not None
                        for j in range(len(result[i].value) - 1, -1, -1):
                            v = result[i].value
                            if j < len(v) and len(v) > kind.min_size:
                                replace({i: v[:j] + v[j + 1 :]})
                                result = self.result
                                assert result is not None
                        result = self.result
                        assert result is not None
                        for j in range(len(result[i].value) - 1, -1, -1):
                            v = result[i].value
                            if j < len(v) and v[j] != 0:
                                bin_search_down(
                                    0,
                                    v[j],
                                    lambda b: replace(
                                        {
                                            i: result[i].value[:j]
                                            + bytes([b])
                                            + result[i].value[j + 1 :]
                                        }
                                    ),
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
