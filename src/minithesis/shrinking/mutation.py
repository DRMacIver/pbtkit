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

from minithesis.core import (
    BUFFER_SIZE,
    MinithesisState,
    TestCase,
    shrink_pass,
)


@shrink_pass
def mutate_and_shrink(state: MinithesisState) -> None:
    """Try random mutations of a few positions to escape local optima.

    For each position, try incrementing it and filling the continuation
    with random values (rather than simplest). This discovers branch
    switches where the new branch needs non-trivial downstream values.

    Limited to short sequences to avoid excessive entropy."""
    assert state.result is not None
    if len(state.result) > 16:
        return
    rng = Random(0)
    i = 0
    while i < len(state.result):
        node = state.result[i]
        if not isinstance(node.value, int):
            i += 1
            continue
        new_val = node.value + 1
        if not node.kind.validate(new_val):
            i += 1
            continue
        # Try a handful of random continuations after the increment.
        for _ in range(3):
            prefix = [n.value for n in state.result[:i]] + [new_val]
            tc = TestCase(
                prefix=tuple(prefix),
                random=rng,
                max_size=BUFFER_SIZE,
            )
            state.test_function(tc)
        i += 1
