"""Test function caching for minithesis.

This module provides cached_test_function, a decorator/wrapper that
adds choice-tree caching to MinithesisState.test_function. It avoids
redundant test evaluations during shrinking by predicting outcomes
from previously observed choice sequences.

It is imported by the package's __init__.py to apply the wrapper
and register the run phase that activates caching.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

from minithesis.core import (
    MinithesisState,
    Status,
    TestCase,
    run_phase,
)


class CachedTestFunction:
    """A choice-tree cache for test function results.

    Maintains a tree of previously observed choices and their
    outcomes, allowing prediction of results without re-running
    the test function. Can also be called directly with a choices
    list for standalone use.

    You can safely omit implementing this at the cost of
    somewhat increased shrinking time.
    """

    def __init__(self, test_function: Optional[Callable[[TestCase], None]] = None):
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
        self.tree: Dict[Any, Union[Status, Dict[Any, Any]]] = {}

    def lookup(self, choices: Sequence[Any]) -> Optional[Status]:
        """Check if the outcome can be predicted from the cache.
        Returns the Status if known, or None on cache miss."""
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
            return None

    def record(self, test_case: TestCase) -> None:
        """Record the outcome of a test case in the cache tree."""
        assert test_case.status is not None
        node: Any = self.tree
        for i, choice_node in enumerate(test_case.nodes):
            c = choice_node.value
            if i + 1 < len(test_case.nodes) or test_case.status == Status.OVERRUN:
                try:
                    existing = node[c]
                except KeyError:
                    node = node.setdefault(c, {})
                    continue
                # A previous recording at this position recorded a
                # terminal Status. That shorter result takes precedence,
                # so stop recording here.
                if isinstance(existing, Status):
                    return
                node = existing
            else:
                node[c] = test_case.status

    def __call__(self, choices: Sequence[Any]) -> Status:
        """Look up choices in the cache, calling the test function
        on cache miss. Requires test_function to have been passed
        to __init__."""
        status = self.lookup(choices)
        if status is not None:
            return status
        assert self._test_function is not None
        test_case = TestCase.for_choices(choices)
        self._test_function(test_case)
        assert test_case.status is not None
        self.record(test_case)
        return test_case.status


def cached_test_function(fn: Callable) -> Callable:
    """Wrap a MinithesisState.test_function method to add
    choice-tree caching during shrinking.

    When the cache is not active (i.e. during generation),
    the original method is called directly. A run_phase
    activates the cache before shrinking begins."""

    def wrapper(self: MinithesisState, test_case: TestCase) -> None:
        cache: Optional[CachedTestFunction] = getattr(self.extras, "cache", None)
        # Only cache deterministic test cases (from for_choices).
        # Test cases with a random source (generation, targeting)
        # must always run the real test function.
        if cache is None or test_case._random is not None:
            return fn(self, test_case)
        status = cache.lookup(list(test_case.prefix))
        if status is not None:
            test_case.status = status
            return
        fn(self, test_case)
        assert test_case.status is not None
        cache.record(test_case)

    return wrapper


# Apply to MinithesisState.
MinithesisState.test_function = cached_test_function(MinithesisState.test_function)


@run_phase
def _activate_cache(state: MinithesisState) -> None:
    """Activate the choice-tree cache before shrinking."""
    state.extras.cache = CachedTestFunction()
