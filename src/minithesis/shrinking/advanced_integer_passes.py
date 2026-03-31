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
from minithesis.text import StringChoice


def _integer_indices(state: MinithesisState) -> list[int]:
    """Return indices of all IntegerChoice nodes in the result."""
    assert state.result is not None
    return [
        i for i, node in enumerate(state.result) if isinstance(node.kind, IntegerChoice)
    ]


def _bumpable_indices(state: MinithesisState) -> list[int]:
    """Return indices of IntegerChoice, BooleanChoice, FloatChoice,
    and BytesChoice nodes."""
    assert state.result is not None
    return [
        i
        for i, node in enumerate(state.result)
        if isinstance(
            node.kind,
            (IntegerChoice, BooleanChoice, FloatChoice, BytesChoice, StringChoice),
        )
    ]


@shrink_pass
def lower_and_bump(state: MinithesisState) -> None:
    """For IntegerChoice/BooleanChoice nodes with value > 0, try
    decrementing and bumping a later integer or boolean. First runs
    the decrement alone to discover the kind of the later node in
    the new context, then tries the boundary and powers of 2.

    Value punning in _make_choice handles the case where decrementing
    changes the type at position j (e.g., one_of branch switch)."""
    assert state.result is not None
    for gap in range(1, min(len(_bumpable_indices(state)), 8)):
        idx = 0
        while idx < len(_bumpable_indices(state)):
            bump_indices = _bumpable_indices(state)
            i = bump_indices[idx]
            node_i = state.result[i]
            if isinstance(node_i.kind, (FloatChoice, BytesChoice, StringChoice)):
                if node_i.value == node_i.kind.simplest:
                    idx += 1
                    continue
                new_i = node_i.kind.simplest
            elif node_i.value <= 0:
                idx += 1
                continue
            else:
                new_i = node_i.value - 1
            # Find the bump target: the gap'th bumpable index after i.
            bump_indices = _bumpable_indices(state)
            targets_after_i = [k for k in bump_indices if k > i]
            if gap - 1 >= len(targets_after_i):
                idx += 1
                continue
            j = targets_after_i[gap - 1]
            # Run the decrement to observe the kind at position j.
            attempt = list(state.result)
            attempt[i] = attempt[i].with_value(new_i)
            tc = TestCase.for_choices([n.value for n in attempt], prefix_nodes=attempt)
            state.test_function(tc)
            assert tc.status is not None
            # Also try the decrement with everything after i zeroed.
            zeroed = list(attempt)
            for k in range(i + 1, len(zeroed)):
                zeroed[k] = zeroed[k].with_value(zeroed[k].kind.simplest)
            tc_z = TestCase.for_choices([n.value for n in zeroed], prefix_nodes=zeroed)
            state.test_function(tc_z)
            if j < len(tc.nodes) and j < len(state.result):
                kind_j = tc.nodes[j].kind
                # Only try boundary values when the kind at j matches
                # between the trial and the current result (avoids
                # putting e.g. int values into BytesChoice nodes).
                if isinstance(kind_j, IntegerChoice) and isinstance(
                    state.result[j].kind, IntegerChoice
                ):
                    state.replace({i: new_i, j: kind_j.max_value})
                    state.replace({i: new_i, j: kind_j.simplest})
            # For FloatChoice targets, try small whole numbers.
            if j < len(state.result) and isinstance(state.result[j].kind, FloatChoice):
                for fv in [1.0, -1.0, 2.0, -2.0]:
                    if state.result[j].kind.validate(fv):
                        state.replace({i: new_i, j: fv})
            # For StringChoice/BytesChoice targets, try unit values.
            if j < len(state.result) and isinstance(
                state.result[j].kind, (StringChoice, BytesChoice)
            ):
                state.replace({i: new_i, j: state.result[j].kind.unit})
            # Try bumping from current value by powers of 2.
            # Skip bumping for BytesChoice/StringChoice targets since
            # integer arithmetic doesn't apply.
            if j < len(state.result) and isinstance(
                state.result[j].kind, (IntegerChoice, BooleanChoice)
            ):
                bump = 1
                found = False
                while bump <= 256 and j < len(state.result):
                    new_j = state.result[j].value + bump
                    if state.replace({i: new_i, j: new_j}):
                        found = True
                        break
                    bump *= 2
                if not found and j < len(state.result):
                    state.replace({i: new_i, j: 0})
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
            assert pair_idx - 1 + gap < len(indices)
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
        candidates: list[Any] = []
        if isinstance(node.kind, BytesChoice):
            candidates = [b"\x00" * node.kind.max_size]
        elif isinstance(node.kind, StringChoice):
            # Try max-length string of simplest character.
            simplest_char = chr(node.kind.min_codepoint)
            candidates = [simplest_char * node.kind.max_size]
        elif isinstance(node.kind, IntegerChoice):
            # Try +1, max_value, and min_value. +1 catches gradual
            # thresholds, max_value catches sampled_from, min_value
            # catches cases where going negative shortens the path.
            candidates = []
            for v in [node.value + 1, node.kind.max_value, node.kind.min_value]:
                if v != node.value and node.kind.validate(v) and v not in candidates:
                    candidates.append(v)
        elif isinstance(node.kind, FloatChoice):
            # Try small numbers, powers of 2, and range boundaries.
            float_vals: list[float] = []
            for n in range(-8, 9):
                float_vals.append(float(n))
            p = 16.0
            while p <= 256.0:
                float_vals.extend([p, -p])
                p *= 2.0
            candidates = []
            for v in float_vals + [node.kind.min_value, node.kind.max_value]:
                if v != node.value and node.kind.validate(v) and v not in candidates:
                    candidates.append(v)
        elif isinstance(node.kind, BooleanChoice) and not node.value:
            candidates = [True]
        if not candidates:
            i += 1
            continue
        for incremented in candidates:
            # Run the test with the incremented value. If the test takes a
            # shorter interesting path, state.test_function updates state.result.
            attempt = list(state.result)
            attempt[i] = attempt[i].with_value(incremented)

            # Try bumping the value and filling the rest with simplest.
            # Pass prefix_nodes so value punning works correctly.
            zeroed = list(attempt)
            for j in range(i + 1, len(zeroed)):
                zeroed[j] = zeroed[j].with_value(zeroed[j].kind.simplest)
            tc_zeroed = TestCase.for_choices(
                [n.value for n in zeroed], prefix_nodes=zeroed
            )
            state.test_function(tc_zeroed)
            # Try the increment with original values (not zeroed) to
            # handle cases where filters reject simplest values.
            tc_orig = TestCase.for_choices(
                [n.value for n in attempt], prefix_nodes=attempt
            )
            state.test_function(tc_orig)
            # Try setting each position after i to 1 or -1 individually
            # (rest at simplest). Handles one_of branch switches (1/True)
            # and negative threshold conditions (-1).
            for j in range(i + 1, min(i + 9, len(attempt))):
                for fill_val in [1, -1]:
                    filled = list(zeroed)
                    filled[j] = filled[j].with_value(fill_val)
                    tc_filled = TestCase.for_choices(
                        [n.value for n in filled], prefix_nodes=filled
                    )
                    state.test_function(tc_filled)
        i += 1
