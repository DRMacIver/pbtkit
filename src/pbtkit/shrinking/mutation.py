"""Mutation-based shrink pass for pbtkit.

Tries random mutations of the current best result to escape local optima
that deterministic passes can't find. Particularly useful when switching
a branch index requires multiple downstream values to change
simultaneously (e.g. one_of branch switching).

Kept as a last resort — mutations increase entropy, creating more work
for subsequent deterministic passes.
"""

from __future__ import annotations

from random import Random

from pbtkit.core import (
    PbtkitState,
    TestCase,
    shrink_pass,
)

# Number of random continuations to try per mutation.
RANDOM_ATTEMPTS = 3


@shrink_pass
def mutate_and_shrink(state: PbtkitState) -> None:
    """Try random mutations of a few positions to escape local optima.

    For each indexed position, try changing its value by small index
    offsets and filling the continuation with random values. Also probes
    with extreme continuations to handle boundary cases."""
    assert state.result is not None
    if len(state.result) > 32:
        return
    i = 0
    while i < len(state.result):
        node = state.result[i]
        kind = node.kind
        assert kind.supports_index
        current_idx = kind.to_index(node.value)
        # Try small index offsets (±1 through ±5).
        candidates = []
        for delta in range(1, 6):
            for sign in [1, -1]:
                new_idx = current_idx + delta * sign
                if new_idx < 0:
                    continue
                v = kind.from_index(new_idx)
                if v is not None and v != node.value and v not in candidates:
                    candidates.append(v)
        for new_val in candidates:
            prefix = [n.value for n in state.result[:i]] + [new_val]
            # Probe with simplest continuation to discover downstream kinds.
            probe = TestCase(
                prefix=tuple(prefix),
                random=Random(0),
                max_size=len(state.result),
            )
            state.test_function(probe)
            # Try random continuations for general exploration.
            for attempt in range(RANDOM_ATTEMPTS):
                tc = TestCase(
                    prefix=tuple(prefix),
                    random=Random(i * 1000 + attempt),
                    max_size=len(state.result),
                )
                state.test_function(tc)
            # Also try setting each of the next few positions to
            # unit (from_index(1)), with random continuation.
            for j_offset in range(1, min(3, len(state.result) - i)):
                j = i + j_offset
                assert j < len(state.result)
                kind_j = state.result[j].kind
                assert kind_j.supports_index
                unit_val = kind_j.from_index(1)
                if unit_val is None:
                    continue
                two_prefix = prefix + [
                    unit_val if k == j else state.result[k].kind.simplest
                    for k in range(i + 1, j + 1)
                ]
                for attempt in range(RANDOM_ATTEMPTS):
                    tc2 = TestCase(
                        prefix=tuple(two_prefix),
                        random=Random(i * 1000 + j_offset * 100 + attempt),
                        max_size=len(state.result),
                    )
                    state.test_function(tc2)
        i += 1
