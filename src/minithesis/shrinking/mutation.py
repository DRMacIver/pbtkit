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


def _extreme_value_for(node: ChoiceNode, maximize: bool) -> object:
    """Return the maximum or minimum valid value for a choice node's kind."""
    kind = node.kind
    if isinstance(kind, IntegerChoice):
        return kind.max_value if maximize else kind.min_value
    if isinstance(kind, BooleanChoice):
        return True if maximize else False
    if isinstance(kind, BytesChoice):
        if maximize:
            return b"\xff" * kind.max_size
        return b"\x00" * kind.min_size
    if isinstance(kind, StringChoice):
        if maximize:
            return chr(kind.max_codepoint) * kind.max_size
        return chr(kind.min_codepoint) * kind.min_size
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
    i = 0
    while i < len(state.result):
        # Cap all attempts at current best length — anything longer
        # can't be a shrink.
        max_size = len(state.result)
        node = state.result[i]
        if not isinstance(node.value, int):
            i += 1
            continue
        # Try small offsets (±1 through ±5) to cover different
        # modular residues and nearby branch switches.
        candidates = []
        for delta in range(1, 6):
            for sign in [1, -1]:
                v = node.value + delta * sign
                if node.kind.validate(v) and v not in candidates:
                    candidates.append(v)
        for new_val in candidates:
            prefix = [n.value for n in state.result[:i]] + [new_val]
            # Probe with simplest continuation to discover downstream kinds.
            probe = TestCase(
                prefix=tuple(prefix),
                random=Random(0),
                max_size=max_size,
            )
            state.test_function(probe)
            # Try extreme downstream values (max and min) to find
            # boundary cases like max-length bytes or min-value integers.
            if len(probe.nodes) > len(prefix):
                for maximize in [True, False]:
                    extreme = prefix + [
                        _extreme_value_for(n, maximize)
                        for n in probe.nodes[len(prefix) :]
                    ]
                    tc_ext = TestCase(
                        prefix=tuple(extreme),
                        random=Random(i),
                        max_size=max_size,
                    )
                    state.test_function(tc_ext)
            # Also try setting each of the next few positions to
            # 0 or 1, with random continuation. This handles 2-position
            # compound changes (e.g. branch index + boolean value).
            for j in range(1, min(4, len(state.result) - i)):
                for fill in [0, 1]:
                    two_prefix = prefix + [
                        fill if k == i + j else state.result[k].kind.simplest
                        for k in range(i + 1, min(i + j + 1, len(state.result)))
                    ]
                    for attempt in range(RANDOM_ATTEMPTS):
                        tc2 = TestCase(
                            prefix=tuple(two_prefix),
                            random=Random(i * 1000 + j * 100 + fill * 10 + attempt),
                            max_size=max_size,
                        )
                        state.test_function(tc2)
            # Try random continuations for general exploration.
            for attempt in range(RANDOM_ATTEMPTS):
                tc = TestCase(
                    prefix=tuple(prefix),
                    random=Random(i * 1000 + attempt),
                    max_size=max_size,
                )
                state.test_function(tc)
        i += 1
