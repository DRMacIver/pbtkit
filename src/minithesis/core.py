# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import types
from collections import defaultdict
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
    Sequence,
    Tuple,
    TypeVar,
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
        if self.min_value <= 0 <= self.max_value:
            return 0
        elif abs(self.min_value) <= abs(self.max_value):
            return self.min_value
        else:
            return self.max_value

    def validate(self, value: int) -> bool:
        return isinstance(value, int) and self.min_value <= value <= self.max_value

    def sort_key(self, value: int) -> Any:
        return (abs(value), value < 0)


@dataclass(frozen=True)
class BooleanChoice(ChoiceType[bool]):
    p: float

    @property
    def simplest(self) -> bool:
        return False

    def validate(self, value: bool) -> bool:
        return isinstance(value, int) and value in (0, 1)


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


# We cap the maximum amount of entropy a test case can use.
# This prevents cases where the generated test case size explodes
# by effectively rejecting test cases that use too many choices.
BUFFER_SIZE = 8 * 1024


def run_test(
    max_examples: int = 100,
    random: Optional[Random] = None,
    quiet: bool = False,
    **kwargs: Any,
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
    * quiet: Will not print anything on failure if True.

    Additional keyword arguments are stored on the state object's
    ``extras`` namespace, where they can be used by extension hooks.
    """

    def accept(test: Callable[[TestCase], None]) -> None:
        def mark_failures_interesting(test_case: TestCase) -> None:
            try:
                test(test_case)
            except Exception:
                if test_case.status is not None:
                    raise
                test_case.mark_status(Status.INTERESTING)

        state = MinithesisState(
            random or Random(),
            mark_failures_interesting,
            max_examples,
            test_name=test.__name__,
            **kwargs,
        )

        for hook in SETUP_HOOKS:
            hook(state)

        if state.result is None:
            state.run()

        if state.valid_test_cases == 0:
            raise Unsatisfiable()

        for hook in TEARDOWN_HOOKS:
            hook(state)

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
        """Returns a random float. Stub — import minithesis.floats to enable."""
        raise NotImplementedError("import minithesis.floats to use draw_float")

    def draw_string(
        self,
        min_codepoint: int = 0,
        max_codepoint: int = 0x10FFFF,
        min_size: int = 0,
        max_size: int = 10,
    ) -> str:
        """Returns a random string. Stub — import minithesis.text to enable."""
        raise NotImplementedError("import minithesis.text to use draw_string")

    def draw_bytes(self, min_size: int, max_size: int) -> bytes:
        """Returns a random byte string. Stub — import minithesis.bytes to enable."""
        raise NotImplementedError("import minithesis.bytes to use draw_bytes")

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
        """Set a score to maximize. Stub — import minithesis to enable."""
        raise NotImplementedError("import minithesis.targeting to use target")

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
            self.mark_status(Status.EARLY_STOP)
        if forced is not None:
            value = forced
        elif len(self.nodes) < len(self.prefix):
            value = self.prefix[len(self.nodes)]
        else:
            value = rnd_method()
        self.nodes.append(ChoiceNode(kind, value, forced is not None))
        if forced is None and not kind.validate(value):
            self.mark_status(Status.EARLY_STOP)
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
# MinithesisState and attempting to simplify state.result.
# Passes are run in order, repeating until a fixed point.
SHRINK_PASSES: List[Callable[["MinithesisState"], None]] = []
# Value shrinker registry. Maps choice type to list of value
# shrinker functions (kind, value, try_replace) -> None.
VALUE_SHRINKERS: Dict[type, List[Callable]] = defaultdict(list)
TEST_FUNCTION_HOOKS: List[Callable[["MinithesisState", "TestCase"], None]] = []
RUN_PHASES: List[Callable[["MinithesisState"], None]] = []
SETUP_HOOKS: List[Callable[["MinithesisState"], None]] = []
TEARDOWN_HOOKS: List[Callable[["MinithesisState"], None]] = []


def shrink_pass(
    fn: Callable[["MinithesisState"], None],
) -> Callable[["MinithesisState"], None]:
    """Decorator that registers a function as a shrink pass."""
    SHRINK_PASSES.append(fn)
    return fn


def test_function_hook(
    fn: Callable[["MinithesisState", "TestCase"], None],
) -> Callable[["MinithesisState", "TestCase"], None]:
    """Decorator that registers a hook called after each test function run."""
    TEST_FUNCTION_HOOKS.append(fn)
    return fn


def run_phase(
    fn: Callable[["MinithesisState"], None],
) -> Callable[["MinithesisState"], None]:
    """Decorator that registers a phase in the test run (between generate and shrink)."""
    RUN_PHASES.append(fn)
    return fn


def setup_hook(
    fn: Callable[["MinithesisState"], None],
) -> Callable[["MinithesisState"], None]:
    """Decorator that registers a hook called before the test run."""
    SETUP_HOOKS.append(fn)
    return fn


def teardown_hook(
    fn: Callable[["MinithesisState"], None],
) -> Callable[["MinithesisState"], None]:
    """Decorator that registers a hook called after the test run."""
    TEARDOWN_HOOKS.append(fn)
    return fn


def value_shrinker(
    choice_type: type,
) -> Callable:
    """Decorator that registers a value shrinker as a shrink pass.

    The decorated function takes (kind, value, try_replace) and is
    called for each node of the matching choice type during shrinking.
    try_replace(v) -> bool attempts to replace the current value with v."""

    def accept(fn: Callable) -> Callable:
        def shrink_pass_fn(state: "MinithesisState") -> None:
            assert state.result is not None
            i = 0
            while i < len(state.result):
                node = state.result[i]
                if isinstance(node.kind, choice_type):
                    fn(node.kind, node.value, lambda v: state.replace({i: v}))
                i += 1

        shrink_pass_fn.__name__ = fn.__name__
        shrink_pass_fn.__qualname__ = fn.__qualname__
        SHRINK_PASSES.append(shrink_pass_fn)
        VALUE_SHRINKERS[choice_type].append(fn)
        return fn

    return accept


def sort_key(nodes: Sequence[ChoiceNode]) -> Tuple:
    """Returns a key that can be used for the shrinking order
    of test cases. Shorter choice sequences are simpler, and
    among equal lengths we prefer smaller values.

    This comparison is safe because in a non-flaky test, the choice
    type at each position is determined by previous values, so two
    sequences always have the same type at the first index they differ."""
    return (len(nodes), [n.sort_key for n in nodes])


class MinithesisState:
    def __init__(
        self,
        random: Random,
        test_function: Callable[[TestCase], None],
        max_examples: int,
        **kwargs: Any,
    ):
        self.random = random
        self.max_examples = max_examples
        self.__test_function = test_function
        self.valid_test_cases = 0
        self.calls = 0
        self.result: Optional[List[ChoiceNode]] = None
        self.best_scoring: Optional[Tuple[int, List[ChoiceNode]]] = None
        self.test_is_trivial = False
        self.extras = types.SimpleNamespace(**kwargs)

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

        for hook in TEST_FUNCTION_HOOKS:
            hook(self, test_case)

        if test_case.status == Status.INTERESTING and (
            self.result is None or sort_key(test_case.nodes) < sort_key(self.result)
        ):
            self.result = test_case.nodes

    def run(self) -> None:
        self.generate()
        for phase in RUN_PHASES:
            phase(self)
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
        test_case = TestCase.for_choices([n.value for n in nodes])
        self.test_function(test_case)
        assert test_case.status is not None
        return test_case.status == Status.INTERESTING

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
def delete_chunks(state: MinithesisState) -> None:
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
                if (
                    i > 0
                    and isinstance(attempt[i - 1].kind, (IntegerChoice, BooleanChoice))
                    and attempt[i - 1].value != attempt[i - 1].kind.simplest
                ):
                    attempt = list(attempt)
                    attempt[i - 1] = attempt[i - 1].with_value(attempt[i - 1].value - 1)
                    if state.consider(attempt):
                        i += 1
                i -= 1
        k -= 1


@shrink_pass
def zero_choices(state: MinithesisState) -> None:
    """Replace blocks of choices with their simplest values.
    Skip k=1 because we handle that in the per-choice pass."""
    assert state.result is not None
    k = 8
    while k > 0:
        i = len(state.result) - k
        while i >= 0:
            if state.replace(
                {j: state.result[j].kind.simplest for j in range(i, i + k)}
            ):
                i -= k
            else:
                i -= 1
        k -= 1


@value_shrinker(IntegerChoice)
def shrink_integer_toward_zero(
    kind: IntegerChoice, value: int, try_replace: Callable[[int], bool]
) -> None:
    """Try simplest, then flip negative to positive."""
    if value != kind.simplest:
        try_replace(kind.simplest)
    if value < 0 and kind.validate(-value):
        try_replace(-value)


@value_shrinker(IntegerChoice)
def binary_search_integer(
    kind: IntegerChoice, value: int, try_replace: Callable[[int], bool]
) -> None:
    """Binary search the absolute value toward 0."""
    if value > 0:
        bin_search_down(
            max(kind.simplest, 0),
            value,
            try_replace,
        )
    elif value < 0:
        bin_search_down(
            abs(min(kind.simplest, 0)),
            -value,
            lambda a: try_replace(-a),
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
    while lo + 1 < hi:
        mid = lo + (hi - lo) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Frozen(Exception):
    """Attempted to make choices on a test case that has been
    completed."""


class StopTest(Exception):
    """Raised when a test should stop executing early."""


class Unsatisfiable(Exception):
    """Raised when a test has no valid examples."""


class Status(IntEnum):
    # Test case stopped before completing: either ran out of
    # data, or a replayed value failed type validation.
    EARLY_STOP = 0

    # Test case contained something that prevented completion
    INVALID = 1

    # Test case completed just fine but was boring
    VALID = 2

    # Test case completed and was interesting
    INTERESTING = 3
