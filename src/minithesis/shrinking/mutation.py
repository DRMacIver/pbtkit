"""Mutation-based shrink pass for minithesis.

Tries random mutations of the current best result to escape local optima
that deterministic passes can't find. Particularly useful when switching
a branch index requires multiple downstream values to change
simultaneously (e.g. one_of branch switching).

Kept as a last resort — mutations increase entropy, creating more work
for subsequent deterministic passes.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from minithesis.bytes import BytesChoice
from minithesis.core import (
    BUFFER_SIZE,
    BooleanChoice,
    IntegerChoice,
    MinithesisState,
    TestCase,
    shrink_pass,
)
from minithesis.text import StringChoice

if TYPE_CHECKING:
    from minithesis.core import ChoiceNode

# Number of random continuations to try per mutation.
RANDOM_ATTEMPTS = 5


def _max_value_for(node: ChoiceNode) -> object:
    """Return the maximum valid value for a choice node's kind."""
    kind = node.kind
    if isinstance(kind, IntegerChoice):
        return kind.max_value
    if isinstance(kind, BooleanChoice):
        return True
    if isinstance(kind, BytesChoice):
        return b"\xff" * kind.max_size
    if isinstance(kind, StringChoice):
        return chr(kind.max_codepoint) * kind.max_size
    return node.value


@shrink_pass
def mutate_and_shrink(state: MinithesisState) -> None:
    """Try random mutations of a few positions to escape local optima.

    For each position, try changing its value (increment or decrement)
    and filling the continuation with random values. Also probes with
    maximized continuations to handle cases like list→bytes
    redistribution where the bytes need to be at max length."""
    assert state.result is not None
    if len(state.result) > 32:
        return
    rng = Random(0)
    i = 0
    while i < len(state.result):
        node = state.result[i]
        if not isinstance(node.value, int):
            i += 1
            continue
        # Try both incrementing and decrementing.
        candidates = []
        if node.kind.validate(node.value + 1):
            candidates.append(node.value + 1)
        if node.kind.validate(node.value - 1):
            candidates.append(node.value - 1)
        for new_val in candidates:
            prefix = [n.value for n in state.result[:i]] + [new_val]
            # Probe with simplest continuation to discover downstream kinds.
            probe = TestCase(
                prefix=tuple(prefix),
                random=Random(0),
                max_size=BUFFER_SIZE,
            )
            state.test_function(probe)
            # Try maximizing all downstream values to find boundary cases
            # (e.g. making bytes as long as possible).
            if len(probe.nodes) > len(prefix):
                max_values = prefix + [
                    _max_value_for(n) for n in probe.nodes[len(prefix) :]
                ]
                tc_max = TestCase.for_choices(max_values)
                state.test_function(tc_max)
            # Try random continuations for general exploration.
            for _ in range(RANDOM_ATTEMPTS):
                tc = TestCase(
                    prefix=tuple(prefix),
                    random=rng,
                    max_size=BUFFER_SIZE,
                )
                state.test_function(tc)
        i += 1
