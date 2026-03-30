"""Test function caching for minithesis.

This module provides CachedTestFunction, a caching layer that avoids
redundant test function evaluations during shrinking. It is imported
by the package's __init__.py to register the run phase.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Union

from minithesis.core import (
    MinithesisState,
    Status,
    TestCase,
    run_phase,
)


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


@run_phase
def _install_cache(state: MinithesisState) -> None:
    """Install a CachedTestFunction on the state before shrinking."""
    state._cached = CachedTestFunction(state.test_function)
