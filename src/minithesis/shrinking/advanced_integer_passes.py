"""Advanced integer shrink passes for minithesis.

This module provides redistribute_integers, which improves shrinking
quality for tests involving multiple integer choices by redistributing
value between pairs. It is imported by the package's __init__.py to
register as a shrink pass.
"""

from __future__ import annotations

from typing import Any

from minithesis.bytes import BytesChoice
from minithesis.core import (
    BooleanChoice,
    IntegerChoice,
    MinithesisState,
    Status,
    TestCase,
    bin_search_down,
    shrink_pass,
)
from minithesis.floats import FloatChoice


def _integer_indices(state: MinithesisState) -> list[int]:
    """Return indices of all IntegerChoice nodes in the result."""
    assert state.result is not None
    return [
        i for i, node in enumerate(state.result) if isinstance(node.kind, IntegerChoice)
    ]


def _bumpable_indices(state: MinithesisState) -> list[int]:
    """Return indices of IntegerChoice and BooleanChoice nodes."""
    assert state.result is not None
    return [
        i
        for i, node in enumerate(state.result)
        if isinstance(node.kind, (IntegerChoice, BooleanChoice))
    ]


@shrink_pass
def lower_and_bump(state: MinithesisState) -> None:
    """For IntegerChoice nodes with value > 0, try decrementing and
    bumping a later integer or boolean. First runs the decrement alone
    to discover the kind of the later node in the new context, then
    tries the boundary and powers of 2.

    Value punning in _make_choice handles the case where decrementing
    changes the type at position j (e.g., one_of branch switch)."""
    assert state.result is not None
    for gap in range(1, min(len(_bumpable_indices(state)), 8)):
        idx = 0
        while idx < len(_integer_indices(state)):
            int_indices = _integer_indices(state)
            bump_indices = _bumpable_indices(state)
            i = int_indices[idx]
            if state.result[i].value <= 0:
                idx += 1
                continue
            # Find the bump target: the gap'th bumpable index after i.
            targets_after_i = [k for k in bump_indices if k > i]
            if gap - 1 >= len(targets_after_i):
                idx += 1
                continue
            j = targets_after_i[gap - 1]
            new_i = state.result[i].value - 1
            # Run the decrement to observe the kind at position j.
            attempt = list(state.result)
            attempt[i] = attempt[i].with_value(new_i)
            tc = TestCase.for_choices([n.value for n in attempt], prefix_nodes=attempt)
            state.test_function(tc)
            assert tc.status is not None
            if j < len(tc.nodes):
                kind_j = tc.nodes[j].kind
                if isinstance(kind_j, IntegerChoice):
                    state.replace({i: new_i, j: kind_j.max_value})
            # Try bumping from current value by powers of 2.
            bump = 1
            found = False
            while bump <= 256 and j < len(state.result):
                new_j = state.result[j].value + bump
                if state.replace({i: new_i, j: new_j}):
                    found = True
                    break
                bump *= 2
            # If bumps from current value didn't work (e.g. range changed
            # and current+bump is always out of range), try absolute powers
            # of 2 to explore the new range, both positive and negative.
            if not found:
                bump = 1
                while bump <= 256 and j < len(state.result):
                    if state.replace({i: new_i, j: bump}):
                        break
                    if state.replace({i: new_i, j: -bump}):
                        break
                    bump *= 2
            idx += 1


@shrink_pass
def redistribute_integers(state: MinithesisState) -> None:
    """Try adjusting pairs of integer choices by redistributing
    value between them. Operates on pairs of IntegerChoice nodes
    at various distances, skipping non-integer choices in between.
    Useful for tests that depend on the sum of some generated values."""
    assert state.result is not None
    indices = _integer_indices(state)
    for gap in range(1, min(len(indices), 8)):
        for pair_idx in range(len(indices) - gap, 0, -1):
            # Recompute indices since previous iterations may have
            # changed the result structure (e.g. via value punning).
            indices = _integer_indices(state)
            if pair_idx - 1 + gap >= len(indices):
                continue
            i = indices[pair_idx - 1]
            j = indices[pair_idx - 1 + gap]
            assert j < len(state.result) and i < len(state.result)
            assert isinstance(state.result[i].kind, IntegerChoice)
            assert isinstance(state.result[j].kind, IntegerChoice)
            # Try to redistribute value from i toward j, reducing |i|.
            # Keep the sum constant: when i changes by delta, j changes
            # by -delta.
            if state.result[i].value != state.result[i].kind.simplest:
                previous_i = state.result[i].value
                previous_j = state.result[j].value
                if previous_i > 0:
                    bin_search_down(
                        0,
                        previous_i,
                        lambda v: state.replace(
                            {i: v, j: previous_j + (previous_i - v)}
                        ),
                    )
                else:
                    assert previous_i < 0
                    bin_search_down(
                        0,
                        -previous_i,
                        lambda a: state.replace(
                            {i: -a, j: previous_j + (previous_i + a)}
                        ),
                    )


@shrink_pass
def try_shortening_via_increment(state: MinithesisState) -> None:
    """Try incrementing each boolean/integer/bytes value to see if the test
    takes a shorter path (e.g., triggering an earlier assertion).

    A value shrinker can only make values simpler, but sometimes making
    a value LESS simple (e.g., False→True) causes an earlier exit,
    producing a shorter and thus overall simpler choice sequence."""
    assert state.result is not None
    i = 0
    while i < len(state.result):
        node = state.result[i]
        if isinstance(node.kind, BytesChoice):
            candidates: list[Any] = [b"\x00" * node.kind.max_size]
        elif isinstance(node.kind, IntegerChoice):
            # Try both +1 and max_value. +1 catches gradual thresholds,
            # max_value catches sampled_from where only the last index
            # maps to a value that triggers an early exit.
            candidates = []
            if node.kind.validate(node.value + 1):
                candidates.append(node.value + 1)
            if (
                node.kind.max_value != node.value
                and node.kind.max_value != node.value + 1
            ):
                candidates.append(node.kind.max_value)
        elif isinstance(node.kind, FloatChoice):
            # Try unit, negative unit, and range boundaries.
            candidates = []
            for v in [node.kind.unit, -node.kind.unit, node.kind.min_value]:
                if v != node.value and node.kind.validate(v):
                    candidates.append(v)
        elif isinstance(node.kind, BooleanChoice):
            if node.value is False:
                candidates = [True]
            else:
                i += 1
                continue
        else:
            i += 1
            continue

        for incremented in candidates:
            # Run the test with the incremented value. If the test takes a
            # shorter interesting path, state.test_function updates state.result.
            attempt = list(state.result)
            attempt[i] = attempt[i].with_value(incremented)

            # First try bumping the value and zeroing the rest of the
            # test case (likely to make it much smaller)
            tc_zeroed = TestCase.for_choices(
                [n.value if j <= i else n.kind.simplest for j, n in enumerate(attempt)]
            )
            state.test_function(tc_zeroed)
            if (
                len(tc_zeroed.nodes) < len(state.result)
                and tc_zeroed.status is not None
                and tc_zeroed.status < Status.INTERESTING
            ):
                # Bump-and-zero reduced the length but didn't produce an
                # interesting test case. Try with just the bump.
                tc = TestCase.for_choices([n.value for n in attempt])
                state.test_function(tc)
        i += 1
