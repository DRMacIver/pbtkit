"""Test function caching for pbtkit.

This module provides cached_test_function, a decorator/wrapper that
adds choice-tree caching to PbtkitState.test_function. It avoids
redundant test evaluations during shrinking by predicting outcomes
from previously observed choice sequences.

It is imported by the package's __init__.py to apply the wrapper
and register the run phase that activates caching.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from typing import Any

from pbtkit.core import (
    ChoiceNode,
    PbtkitState,
    Status,
    TestCase,
    run_phase,
)

# Sentinel key for storing the ChoiceType at each trie node.
# Cannot collide with choice values (which are ints, bools, etc.).
_KIND = object()


def _cache_key(value: Any) -> Any:
    """Return a dict key that distinguishes True, 1, 1.0, -0.0, and NaN variants.

    Python considers True == 1 == 1.0 and uses the same hash,
    so a plain dict lookup would collide. We prefix with the
    type name to prevent this.

    Floats use their exact bit pattern because Python's equality
    conflates 0.0 and -0.0 (and can't distinguish NaN bit patterns)."""
    if isinstance(value, float):
        return ("float", struct.pack("!d", value))
    return (type(value).__name__, value)


class CachedTestFunction:
    """A choice-tree cache for test function results.

    Maintains a tree of previously observed choices and their
    outcomes, allowing prediction of results without re-running
    the test function. Can also be called directly with a choices
    list for standalone use.

    You can safely omit implementing this at the cost of
    somewhat increased shrinking time.
    """

    def __init__(self, test_function: Callable[[TestCase], None] | None = None):
        self._test_function = test_function
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
        self.tree: dict[Any, Status | dict[Any, Any]] = {}

    def lookup(self, choices: Sequence[Any]) -> tuple[Status, list[ChoiceNode]] | None:
        """Check if the outcome can be predicted from the cache.
        Returns (Status, nodes) if known, or None on cache miss.
        The nodes list reconstructs the ChoiceNode sequence from
        stored kind information."""
        node: Any = self.tree
        nodes: list[ChoiceNode] = []
        try:
            for c in choices:
                # Read the kind stored at this trie level.
                kind = node.get(_KIND)
                if kind is not None:
                    nodes.append(ChoiceNode(kind, c, False))
                node = node[_cache_key(c)]
                # mark_status was called at this point, so future
                # choices are irrelevant.
                if isinstance(node, Status):
                    assert node != Status.EARLY_STOP
                    return (node, nodes)
            # All choices consumed but more would be needed — overrun.
            return (Status.EARLY_STOP, nodes)
        except KeyError:
            return None

    def record(self, test_case: TestCase) -> None:
        """Record the outcome of a test case in the cache tree.

        Stores the ChoiceType (kind) at each trie level so that
        lookup can reconstruct ChoiceNode objects on cache hit."""
        assert test_case.status is not None
        node: Any = self.tree
        for i, choice_node in enumerate(test_case.nodes):
            key = _cache_key(choice_node.value)
            # Store the kind for this position on the current node.
            node[_KIND] = choice_node.kind
            if i + 1 < len(test_case.nodes) or test_case.status == Status.EARLY_STOP:
                try:
                    existing = node[key]
                except KeyError:
                    node = node.setdefault(key, {})
                    continue
                # A previous recording at this position was a terminal
                # Status, which should have been caught by lookup.
                assert not isinstance(existing, Status)
                node = existing
            else:
                node[key] = test_case.status

    def __call__(self, choices: Sequence[Any]) -> Status:
        """Look up choices in the cache, calling the test function
        on cache miss. Requires test_function to have been passed
        to __init__."""
        result = self.lookup(choices)
        if result is not None:
            return result[0]
        assert self._test_function is not None
        test_case = TestCase.for_choices(choices)
        self._test_function(test_case)
        assert test_case.status is not None
        self.record(test_case)
        return test_case.status


def cached_test_function(fn: Callable) -> Callable:
    """Wrap a PbtkitState.test_function method to add
    choice-tree caching during shrinking.

    When the cache is not active (i.e. during generation),
    the original method is called directly. A run_phase
    activates the cache before shrinking begins."""

    def wrapper(self: PbtkitState, test_case: TestCase) -> None:
        cache: CachedTestFunction | None = getattr(self.extras, "cache", None)
        # Only cache deterministic test cases (from for_choices).
        # Test cases with a random source (generation, targeting)
        # must always run the real test function.
        if cache is None or test_case._random is not None:
            return fn(self, test_case)
        result = cache.lookup(list(test_case.prefix))
        if result is not None:
            test_case.status = result[0]
            test_case.nodes = result[1]
            return
        fn(self, test_case)
        assert test_case.status is not None
        cache.record(test_case)

    return wrapper


# Apply to PbtkitState.
PbtkitState.test_function = cached_test_function(PbtkitState.test_function)


@run_phase
def _activate_cache(state: PbtkitState) -> None:
    """Activate the choice-tree cache before shrinking."""
    state.extras.cache = CachedTestFunction()
