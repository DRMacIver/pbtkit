# This file is part of Pbtkit, which may be found at
# https://github.com/DRMacIver/pbtkit
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import types
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from random import Random
from typing import (
    Any,
    Generic,
    NoReturn,
    TypeVar,
)

from pbtkit.features import feature_enabled, needed_for

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

    @property
    def unit(self) -> U:
        """The second simplest value for this choice type.

        Used when punning a value from one choice type to another:
        simplest maps to simplest, everything else maps to unit."""
        raise NotImplementedError

    def validate(self, value: U) -> bool:
        """Return True if value is valid for this choice type."""
        raise NotImplementedError

    def sort_key(self, value: U) -> Any:
        """Returns a comparable key for ordering values during
        shrinking. By default just returns the value itself."""
        return value

    @needed_for("indexing")
    @property
    def supports_index(self) -> bool:
        """Whether this choice type implements to_index/from_index."""
        return type(self).to_index is not ChoiceType.to_index

    @needed_for("shrinking.index_passes")
    @property
    def max_index(self) -> int:
        """The largest valid index for from_index. Returns 0 for
        types that don't support indexing."""
        raise NotImplementedError

    @needed_for("indexing")
    def to_index(self, value: U) -> int:
        """Convert a valid value to a non-negative integer index.

        Index 0 corresponds to simplest. Larger indices represent
        less simple values. The mapping must satisfy:
        from_index(to_index(v)) == v for all valid v."""
        raise NotImplementedError

    @needed_for("indexing")
    def from_index(self, index: int) -> U | None:
        """Convert a non-negative integer index back to a value.

        Returns None if the index doesn't correspond to a valid value.
        Must satisfy: from_index(0) == simplest, and
        to_index(from_index(i)) <= i when from_index(i) is not None."""
        raise NotImplementedError


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

    @property
    def unit(self) -> int:
        s = self.simplest
        # Try s+1 first (next positive), then s-1 (next negative).
        # Falls back to simplest for single-value ranges.
        if self.validate(s + 1):
            return s + 1
        if self.validate(s - 1):
            return s - 1
        return s

    def validate(self, value: int) -> bool:
        return isinstance(value, int) and self.min_value <= value <= self.max_value

    def sort_key(self, value: int) -> Any:
        return (abs(value), value < 0)

    @needed_for("shrinking.index_passes")
    @property
    def max_index(self) -> int:
        return self.max_value - self.min_value

    @needed_for("indexing")
    def to_index(self, value: int) -> int:
        """Dense index matching sort_key order (O(1)).

        Counts valid values with strictly smaller sort_key. Since
        sort_key is (abs(d), d < 0) where d = value - simplest,
        values are ordered: s, s+1, s-1, s+2, s-2, ..."""
        s = self.simplest
        d = value - s
        if d == 0:
            return 0
        above = self.max_value - s  # how many steps above s are valid
        below = s - self.min_value  # how many steps below s are valid
        ad = abs(d)
        # Count valid values at distances 1..ad-1 (both sides)
        count = min(ad - 1, above) + min(ad - 1, below)
        if d > 0:
            return count + 1
        # d < 0: also count positive side at this distance
        if ad <= above:
            count += 1
        return count + 1

    @needed_for("indexing")
    def from_index(self, index: int) -> int | None:
        """Return the index-th valid value in sort_key order (O(1))."""
        s = self.simplest
        if index == 0:
            return s if self.validate(s) else None
        above = self.max_value - s
        below = s - self.min_value
        # Binary search-style: at each distance d, there are up to 2
        # values (s+d and s-d). Find which distance and side.
        remaining = index
        # At distance d (1-indexed):
        #   positive side exists if d <= above
        #   negative side exists if d <= below
        # Use closed-form: at distances 1..d, total valid values =
        #   min(d, above) + min(d, below)
        lo, hi = 1, above + below
        while lo < hi:
            mid = (lo + hi) // 2
            total = min(mid, above) + min(mid, below)
            if total >= remaining:
                hi = mid
            else:
                lo = mid + 1
        d = lo
        total_at_d = min(d, above) + min(d, below)
        if total_at_d < remaining:
            return None  # index exceeds total valid values
        # How many values at distances < d?
        before = min(d - 1, above) + min(d - 1, below)
        pos_in_d = remaining - before  # 1 or 2
        if pos_in_d == 1 and d <= above:
            return s + d
        assert d <= below
        return s - d


@dataclass(frozen=True)
class BooleanChoice(ChoiceType[bool]):
    p: float

    @property
    def simplest(self) -> bool:
        return False

    @property
    def unit(self) -> bool:
        return True

    def validate(self, value: bool) -> bool:
        return isinstance(value, int) and value in (0, 1)

    @needed_for("shrinking.index_passes")
    @property
    def max_index(self) -> int:
        return 1

    @needed_for("indexing")
    def to_index(self, value: bool) -> int:
        return int(value)

    @needed_for("indexing")
    def from_index(self, index: int) -> bool | None:
        if index == 0:
            return False
        if index == 1:
            return True
        return None


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

# Maximum number of outer shrink loop iterations. Prevents pathological
# cases where each pass makes tiny progress (e.g. 1 ULP on a float)
# that triggers a full restart of all passes. With ~20 passes, each
# iteration is relatively expensive, so 500 iterations gives ample room
# for productive shrinking while bounding worst-case time.
MAX_SHRINK_ITERATIONS = 500


def run_test(
    max_examples: int = 100,
    random: Random | None = None,
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
    fail with an exception if pbtkit finds some test case
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

        state = PbtkitState(
            random or Random(),
            mark_failures_interesting,
            max_examples,
            test_name=test.__name__,
            **kwargs,
        )
        # Store the original (unwrapped) test function so setup hooks can
        # inspect and replace it (e.g. pbtkit.draw_names does this).
        state._original_test = test
        # The function used for the final failing replay. Hooks may update
        # this to a rewritten version.
        state._print_function = test

        for hook in SETUP_HOOKS:
            hook(state)

        if state.result is None:
            state.run()

        if state.valid_test_cases == 0:
            raise Unsatisfiable()

        for hook in TEARDOWN_HOOKS:
            hook(state)

        if state.result is not None:
            state._print_function(
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
        prefix_nodes: Sequence[ChoiceNode] | None = None,
    ) -> TestCase:
        """Returns a test case that makes this series of choices."""
        return TestCase(
            prefix=choices,
            random=None,
            max_size=len(choices),
            print_results=print_results,
            prefix_nodes=prefix_nodes,
        )

    def __init__(
        self,
        prefix: Sequence[Any],
        random: Random | None,
        max_size: float = float("inf"),
        print_results: bool = False,
        prefix_nodes: Sequence[ChoiceNode] | None = None,
    ):
        self.prefix = prefix
        self._random = random
        self.max_size = max_size
        self.nodes: list[ChoiceNode] = []
        self.status: Status | None = None
        self.print_results = print_results
        self.depth = 0
        self._draw_counter = 0
        self.targeting_score: int | None = None
        self.prefix_nodes = prefix_nodes
        if feature_enabled("spans"):  # needed_for("spans")
            # Span tracking — (label, start, stop) regions over nodes.
            self.spans: list[tuple[str, int, int]] = []
            self._span_stack: list[tuple[str, int]] = []
        # Named-draw tracking, used by pbtkit.draw_names when imported.
        self._named_draw_used: set[str] = set()
        self._named_draw_flags: dict[str, bool] = {}

    def draw_integer(self, min_value: int, max_value: int) -> int:
        """Returns a number in the range [min_value, max_value]."""
        n = max_value - min_value
        if n.bit_length() > 64 or n < 0:
            raise ValueError(f"Invalid range [{min_value}, {max_value}]")

        def _draw_uniform() -> int:
            return self.random.randint(min_value, max_value)

        generate = _draw_uniform
        if feature_enabled("edge_case_boosting"):  # needed_for("edge_case_boosting")
            from pbtkit.edge_case_boosting import BOUNDARY_PROBABILITY

            nasty = [min_value, max_value]
            if min_value <= 0 <= max_value and min_value != 0 and max_value != 0:
                nasty.append(0)
            threshold = len(nasty) * BOUNDARY_PROBABILITY

            def _draw_boosted() -> int:
                if self.random.random() < threshold:
                    return self.random.choice(nasty)
                return _draw_uniform()

            generate = _draw_boosted

        return self._make_choice(
            IntegerChoice(min_value, max_value),
            generate,
        )

    # draw_float, draw_bytes, draw_string are added by pbtkit.__init__
    # when it is imported. These stubs exist only for type-checking.
    def draw_float(
        self,
        min_value: float = -float("inf"),
        max_value: float = float("inf"),
        *,
        allow_nan: bool = True,
        allow_infinity: bool = True,
    ) -> float:
        """Returns a random float. Stub — import pbtkit.floats to enable."""
        raise NotImplementedError("import pbtkit.floats to use draw_float")

    def draw_string(
        self,
        min_codepoint: int = 0,
        max_codepoint: int = 0x10FFFF,
        min_size: int = 0,
        max_size: int = 10,
    ) -> str:
        """Returns a random string. Stub — import pbtkit.text to enable."""
        raise NotImplementedError("import pbtkit.text to use draw_string")

    def draw_bytes(self, min_size: int, max_size: int) -> bytes:
        """Returns a random byte string. Stub — import pbtkit.bytes to enable."""
        raise NotImplementedError("import pbtkit.bytes to use draw_bytes")

    def choice(self, n: int) -> int:
        """Returns a number in the range [0, n]"""
        result = self.draw_integer(0, n)
        if self._should_print():
            print(f"choice({n}): {result}")
        return result

    def weighted(self, p: float, *, forced: bool | None = None) -> bool:
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
        """Set a score to maximize. Stub — import pbtkit to enable."""
        raise NotImplementedError("import pbtkit.targeting to use target")

    def start_span(self, label: str) -> None:
        """Begin a labelled region. Stub — import pbtkit.spans to enable."""
        raise NotImplementedError("import pbtkit.spans to use start_span")

    def stop_span(self) -> None:
        """End the most recent span. Stub — import pbtkit.spans to enable."""
        raise NotImplementedError("import pbtkit.spans to use stop_span")

    def draw_named(self, generator: "Generator[U]", name: str, repeatable: bool) -> U:
        """Draw with an explicit name for output. Stub — import pbtkit.draw_names to enable."""
        raise NotImplementedError("import pbtkit.draw_names to use draw_named")

    def draw(self, generator: Generator[U]) -> U:
        """Return a value from ``generator``, printing it if this is a failing example."""
        if feature_enabled("spans"):  # needed_for("spans")
            self.start_span(repr(generator))
        try:
            self.depth += 1
            result = generator.produce(self)
        finally:
            self.depth -= 1
            if feature_enabled("spans"):  # needed_for("spans")
                self.stop_span()

        if self._should_print():
            self._draw_counter += 1
            print(f"draw_{self._draw_counter} = {result!r}")
        return result

    def draw_silent(self, generator: Generator[U]) -> U:
        """Return a value from ``generator`` without printing it."""
        try:
            self.depth += 1
            result = generator.produce(self)
        finally:
            self.depth -= 1
        return result

    def note(self, message: str) -> None:
        """Print ``message`` when this is the final failing example."""
        if self.print_results:
            print(message)

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
        forced: U | None = None,
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
            # When replaying from a prefix and the value doesn't validate
            # (e.g., a one_of branch changed the downstream type), pun
            # the value: simplest→simplest, anything else→unit.
            if not kind.validate(value):
                idx = len(self.nodes)
                if (
                    self.prefix_nodes is not None
                    and idx < len(self.prefix_nodes)
                    and value == self.prefix_nodes[idx].kind.simplest
                ):
                    value = kind.simplest
                else:
                    value = kind.unit
        else:
            value = rnd_method()
        self.nodes.append(ChoiceNode(kind, value, forced is not None))
        assert forced is not None or kind.validate(value)
        return value


class Generator(Generic[T]):
    """Represents some range of values that might be used in
    a test, that can be requested from a ``TestCase``.

    Pass one of these to TestCase.any to get a concrete value.
    """

    def __init__(self, produce: Callable[[TestCase], T], name: str | None = None):
        self.produce = produce
        self.name = produce.__name__ if name is None else name

    def __repr__(self) -> str:
        return self.name

    def map(self, f: Callable[[T], S]) -> Generator[S]:
        """Returns a ``Generator`` where values come from
        applying ``f`` to some possible value for ``self``."""
        return Generator(
            lambda test_case: f(test_case.draw(self)),
            name=f"{self.name}.map({f.__name__})",
        )

    def flat_map(self, f: Callable[[T], Generator[S]]) -> Generator[S]:
        """Returns a ``Generator`` where values come from
        drawing a value from ``self``, passing it to ``f`` to
        get a new ``Generator``, then drawing from that."""

        def produce(test_case: TestCase) -> S:
            return test_case.draw(f(test_case.draw(self)))

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
                candidate = test_case.draw(self)
                if f(candidate):
                    return candidate
            test_case.reject()

        return Generator[T](produce, name=f"{self.name}.filter({f.__name__})")


# Shrink pass registry. Each pass is a function taking a
# Shrinker and attempting to simplify shrinker.current.
# Passes are run in order, repeating until a fixed point.
SHRINK_PASSES: list[Callable[[Shrinker], None]] = []
# Value shrinker registry. Maps choice type to list of value
# shrinker functions (kind, value, try_replace) -> None.
VALUE_SHRINKERS: dict[type, list[Callable]] = defaultdict(list)
GENERATION_TYPES: list[Callable[["PbtkitState"], None]] = []
GENERATION_HOOKS: list[Callable[["PbtkitState", "TestCase"], None]] = []
TEST_FUNCTION_HOOKS: list[Callable[["PbtkitState", "TestCase"], None]] = []
RUN_PHASES: list[Callable[["PbtkitState"], None]] = []
SETUP_HOOKS: list[Callable[["PbtkitState"], None]] = []
TEARDOWN_HOOKS: list[Callable[["PbtkitState"], None]] = []


def shrink_pass(
    fn: Callable[[Shrinker], None],
) -> Callable[[Shrinker], None]:
    """Decorator that registers a function as a shrink pass."""
    SHRINK_PASSES.append(fn)
    return fn


def generation_type(
    fn: Callable[["PbtkitState"], None],
) -> Callable[["PbtkitState"], None]:
    """Decorator that registers a function as a generation type."""
    GENERATION_TYPES.append(fn)
    return fn


def generation_hook(
    fn: Callable[["PbtkitState", "TestCase"], None],
) -> Callable[["PbtkitState", "TestCase"], None]:
    """Decorator that registers a hook run after each test case during generation.

    Unlike test_function_hooks, generation hooks only run during the
    generation phase and receive the just-evaluated test case.  They
    may call state.test_function() to evaluate mutations."""
    GENERATION_HOOKS.append(fn)
    return fn


RANDOM_GENERATION_BATCH = 10


@generation_type
def random_generation(state: "PbtkitState") -> None:
    """Standard random generation: run a batch of random test cases."""
    for _ in range(RANDOM_GENERATION_BATCH):
        if not state.should_keep_generating():
            return
        tc = TestCase(prefix=(), random=state.random, max_size=BUFFER_SIZE)
        state.test_function(tc)
        # Run generation hooks (e.g. span mutation) on each generated test case.
        for hook in GENERATION_HOOKS:
            if not state.should_keep_generating():
                return
            hook(state, tc)


def test_function_hook(
    fn: Callable[["PbtkitState", "TestCase"], None],
) -> Callable[["PbtkitState", "TestCase"], None]:
    """Decorator that registers a hook called after each test function run."""
    TEST_FUNCTION_HOOKS.append(fn)
    return fn


def run_phase(
    fn: Callable[["PbtkitState"], None],
) -> Callable[["PbtkitState"], None]:
    """Decorator that registers a phase in the test run (between generate and shrink)."""
    RUN_PHASES.append(fn)
    return fn


def setup_hook(
    fn: Callable[["PbtkitState"], None],
) -> Callable[["PbtkitState"], None]:
    """Decorator that registers a hook called before the test run."""
    SETUP_HOOKS.append(fn)
    return fn


def teardown_hook(
    fn: Callable[["PbtkitState"], None],
) -> Callable[["PbtkitState"], None]:
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
        def shrink_pass_fn(shrinker: Shrinker) -> None:
            i = 0
            while i < len(shrinker.current.nodes):
                node = shrinker.current.nodes[i]
                if isinstance(node.kind, choice_type):
                    fn(node.kind, node.value, lambda v: shrinker.replace({i: v}))
                i += 1

        shrink_pass_fn.__name__ = fn.__name__
        shrink_pass_fn.__qualname__ = fn.__qualname__
        SHRINK_PASSES.append(shrink_pass_fn)
        VALUE_SHRINKERS[choice_type].append(fn)
        return fn

    return accept


def sort_key(nodes: Sequence[ChoiceNode]) -> tuple:
    """Returns a key that can be used for the shrinking order
    of test cases. Shorter choice sequences are simpler, and
    among equal lengths we prefer smaller values.

    This comparison is safe because in a non-flaky test, the choice
    type at each position is determined by previous values, so two
    sequences always have the same type at the first index they differ."""
    return (len(nodes), [n.sort_key for n in nodes])


class PbtkitState:
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
        self.result: list[ChoiceNode] | None = None
        self.best_scoring: tuple[int, list[ChoiceNode]] | None = None
        self.test_is_trivial = False
        self.extras = types.SimpleNamespace(**kwargs)
        # Set by run_test() after construction; hooks may update _print_function.
        self._original_test: Callable[[TestCase], None] | None = None
        self._print_function: Callable[[TestCase], None] | None = None

    @needed_for("draw_names")
    def replace_test_function(self, new_test: Callable[[TestCase], None]) -> None:
        """Replace the test function. The new function is wrapped in the same
        error-marking shim as the original so that exceptions are counted as
        interesting test cases."""

        def mark_failures_interesting(test_case: TestCase) -> None:
            try:
                new_test(test_case)
            except Exception:
                if test_case.status is not None:
                    raise
                test_case.mark_status(Status.INTERESTING)

        self.__test_function = mark_failures_interesting

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
        """Run generation types until either we have found an interesting
        test case or hit the limit of how many test cases we should
        evaluate.  Each generation type runs a small batch then returns,
        and we pick one at random each iteration."""
        while self.should_keep_generating():
            gen = self.random.choice(GENERATION_TYPES)
            gen(self)

    def shrink(self) -> None:
        """If we have found an interesting example, try shrinking it
        so that the choice sequence leading to our best example is
        shortlex smaller than the one we originally found. This improves
        the quality of the generated test case, as per our paper.

        https://drmaciver.github.io/papers/reduction-via-generation-preview.pdf
        """
        if not self.result:
            return

        # Seed a Shrinker from the current best result. Re-running the
        # test both validates the choice sequence is still interesting
        # and gives us a concrete TestCase to hand to the Shrinker.
        nodes = self.result
        initial = TestCase.for_choices([n.value for n in nodes], prefix_nodes=nodes)
        self.test_function(initial)
        assert initial.status == Status.INTERESTING
        Shrinker(
            state=self,
            initial=initial,
            is_interesting=lambda tc: tc.status == Status.INTERESTING,
        ).shrink()


class Shrinker:
    """Drives shrinking of a single interesting test case against an
    is_interesting predicate.

    The shrinker holds a `current` TestCase (its shrink target) and a
    predicate that tells it which completed test cases count as
    interesting. Each registered pass receives a Shrinker, reads
    `shrinker.current.nodes`, and calls `shrinker.consider` /
    `shrinker.replace` / `shrinker.test_function` to try replacements.
    """

    def __init__(
        self,
        state: "PbtkitState",
        initial: TestCase,
        is_interesting: Callable[[TestCase], bool],
    ):
        assert initial.status is not None
        assert is_interesting(initial)
        self.state = state
        self.is_interesting = is_interesting
        self.current: TestCase = initial

    def test_function(self, test_case: TestCase) -> None:
        """Run a test case through the underlying state, and update
        `current` if the resulting test case is interesting and
        shortlex-smaller than the current target."""
        self.state.test_function(test_case)
        assert test_case.status is not None
        if self.is_interesting(test_case) and sort_key(test_case.nodes) < sort_key(
            self.current.nodes
        ):
            self.current = test_case

    def consider(self, nodes: list[ChoiceNode]) -> bool:
        """Test whether a choice sequence is interesting under the
        shrinker's predicate. Returns False for anything the predicate
        rejects, even if the state would otherwise record it."""
        if sort_key(nodes) == sort_key(self.current.nodes):
            return True
        test_case = TestCase.for_choices([n.value for n in nodes], prefix_nodes=nodes)
        self.test_function(test_case)
        assert test_case.status is not None
        return self.is_interesting(test_case)

    def replace(self, values: Mapping[int, Any]) -> bool:
        """Attempt to replace node values at given indices on the
        current target."""
        attempt = list(self.current.nodes)
        for i, v in values.items():
            assert i < len(attempt)
            attempt[i] = attempt[i].with_value(v)
        return self.consider(attempt)

    def shrink(self) -> None:
        """Run registered shrink passes repeatedly until none of them
        make any progress or we hit the iteration cap. The cap
        prevents pathological cases where each pass makes tiny
        progress (e.g. 1 ULP on a float) that triggers a full restart
        of all passes."""
        prev = None
        iterations = 0
        while prev != self.current.nodes and iterations < MAX_SHRINK_ITERATIONS:
            prev = self.current.nodes
            iterations += 1
            for pass_fn in SHRINK_PASSES:
                pass_fn(self)


@shrink_pass
def delete_chunks(shrinker: Shrinker) -> None:
    """Try deleting chunks of choices from the sequence.

    We try longer chunks because this allows us to delete whole
    composite elements: e.g. deleting an element from a generated
    list requires us to delete both the choice of whether to include
    it and also the element itself, which may involve more than one
    choice.

    We iterate backwards because later bits tend to depend on earlier
    bits, so it's easier to make changes near the end."""
    k = 8
    while k > 0:
        i = len(shrinker.current.nodes) - k - 1
        while i >= 0:
            if i >= len(shrinker.current.nodes):
                i -= 1
                continue
            attempt = shrinker.current.nodes[:i] + shrinker.current.nodes[i + k :]
            assert len(attempt) < len(shrinker.current.nodes)
            if not shrinker.consider(list(attempt)):
                if (
                    i > 0
                    and isinstance(attempt[i - 1].kind, (IntegerChoice, BooleanChoice))
                    and attempt[i - 1].value != attempt[i - 1].kind.simplest
                ):
                    modified = list(attempt)
                    modified[i - 1] = modified[i - 1].with_value(
                        modified[i - 1].value - 1
                    )
                    if shrinker.consider(modified):
                        i += 1
                i -= 1
        k -= 1


@shrink_pass
def zero_choices(shrinker: Shrinker) -> None:
    """Replace blocks of choices with their simplest values.
    Skip k=1 because we handle that in the per-choice pass."""
    k = len(shrinker.current.nodes)
    while k > 0:
        i = 0
        while i + k <= len(shrinker.current.nodes):
            nodes = shrinker.current.nodes
            if nodes[i].value == nodes[i].kind.simplest:
                i += 1
            else:
                shrinker.replace({j: nodes[j].kind.simplest for j in range(i, i + k)})
                i += k
        k //= 2


@value_shrinker(IntegerChoice)
def swap_integer_sign(
    kind: IntegerChoice, value: int, try_replace: Callable[[int], bool]
) -> None:
    """Try simplest, then flip negative to positive."""
    if value != kind.simplest:
        try_replace(kind.simplest)
    if value < 0 and kind.validate(-value):
        try_replace(-value)


@value_shrinker(IntegerChoice)
def binary_search_integer_towards_zero(
    kind: IntegerChoice, value: int, try_replace: Callable[[int], bool]
) -> None:
    """Binary search the absolute value toward 0, then linear scan
    small values for non-monotonic functions (e.g. sampled_from)."""
    if value > 0:
        bin_search_down(
            max(kind.simplest, 0),
            value,
            try_replace,
        )
        # Linear scan values that binary search may skip
        # when the function is non-monotonic. Scan more when
        # the range is small (e.g. sampled_from with few options).
        range_size = kind.max_value - kind.min_value + 1
        lo = max(kind.simplest, 0)
        scan_count = min(range_size, 32) if range_size <= 128 else 8
        for v in range(lo, min(value, lo + scan_count)):
            try_replace(v)
        # Also try negative values with smaller absolute value,
        # which are simpler under sort_key (e.g. -1 < 2).
        if kind.min_value < 0:
            upper = min(value - 1, -kind.min_value)
            if upper >= 1:
                # Explicitly try the upper bound since bin_search_down
                # assumes f(hi) is True without calling it.
                try_replace(-upper)
                bin_search_down(1, upper, lambda a: try_replace(-a))
    elif value < 0:
        bin_search_down(
            abs(min(kind.simplest, 0)),
            -value,
            lambda a: try_replace(-a),
        )
        # Linear scan negative values for non-monotonic functions.
        range_size = kind.max_value - kind.min_value + 1
        neg_scan = min(-value, 32) if range_size <= 128 else 8
        for v in range(1, neg_scan):
            try_replace(-v)
        # Also try positive values with smaller absolute value,
        # which are simpler under sort_key (e.g. 2 < -3).
        if kind.max_value > 0:
            upper = min(-value - 1, kind.max_value)
            if upper >= 1:
                try_replace(upper)
                lo_pos = max(kind.simplest, 0)
                bin_search_down(lo_pos, upper, try_replace)
                # Linear scan positive values for non-monotonic functions.
                range_size = kind.max_value - kind.min_value + 1
                scan_count = min(range_size, 32) if range_size <= 128 else 8
                for v in range(lo_pos, min(upper + 1, lo_pos + scan_count)):
                    try_replace(v)


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
