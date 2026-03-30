"""Advanced integer shrink passes for minithesis.

This module provides redistribute_integers, which improves shrinking
quality for tests involving multiple integer choices by redistributing
value between pairs. It is imported by the package's __init__.py to
register as a shrink pass.
"""

from __future__ import annotations

from minithesis.core import (
    BooleanChoice,
    IntegerChoice,
    MinithesisState,
    Status,
    TestCase,
    bin_search_down,
    shrink_pass,
    sort_key,
)


def _integer_indices(state: MinithesisState) -> list[int]:
    """Return indices of all IntegerChoice nodes in the result."""
    assert state.result is not None
    return [
        i for i, node in enumerate(state.result) if isinstance(node.kind, IntegerChoice)
    ]


@shrink_pass
def lower_and_bump_adjacent(state: MinithesisState) -> None:
    """For each IntegerChoice with value > 0, try subtracting 1 and
    bumping the next choice. First runs the decrement alone to discover
    the kind at the next position in the new context, then tries
    boundary values for that kind."""
    assert state.result is not None
    # Recompute indices each iteration so we never use stale
    # positions after a successful replace changes the result.
    idx = 0
    while idx < len(_integer_indices(state)):
        indices = _integer_indices(state)
        i = indices[idx]
        if state.result[i].value <= 0:
            idx += 1
            continue
        new_i = state.result[i].value - 1
        # Run the decrement to observe what happens at position i+1.
        attempt = list(state.result)
        attempt[i] = attempt[i].with_value(new_i)
        tc = TestCase.for_choices([n.value for n in attempt])
        state.test_function(tc)
        assert tc.status is not None
        j = i + 1
        if j < len(tc.nodes):
            kind_j = tc.nodes[j].kind
            if isinstance(kind_j, IntegerChoice):
                # Try the boundary value. Use consider with the probe's
                # nodes to avoid type mismatch (the current result at j
                # may have a different kind).
                probe_attempt = list(tc.nodes[: j + 1])
                probe_attempt[j] = probe_attempt[j].with_value(kind_j.max_value)
                state.consider(probe_attempt)
            elif isinstance(kind_j, BooleanChoice):
                probe_attempt = list(tc.nodes[: j + 1])
                probe_attempt[j] = probe_attempt[j].with_value(True)
                state.consider(probe_attempt)
        # Also try powers of 2 for integer choices at position j.
        if j < len(state.result) and isinstance(state.result[j].kind, IntegerChoice):
            bump = 1
            while bump <= 256 and j < len(state.result):
                new_j = state.result[j].value + bump
                if state.replace({i: new_i, j: new_j}):
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
    """Try incrementing each boolean/integer value to see if the test
    takes a shorter path (e.g., triggering an earlier assertion).

    A value shrinker can only make values simpler, but sometimes making
    a value LESS simple (e.g., False→True) causes an earlier exit,
    producing a shorter and thus overall simpler choice sequence."""
    assert state.result is not None
    i = 0
    while i < len(state.result):
        node = state.result[i]
        if not isinstance(node.kind, (BooleanChoice, IntegerChoice)):
            i += 1
            continue
        incremented = node.value + 1
        if not node.kind.validate(incremented):
            i += 1
            continue
        # Run the test with the incremented value. If the test takes a
        # shorter interesting path, state.test_function updates state.result.
        attempt = list(state.result)
        attempt[i] = attempt[i].with_value(incremented)
        tc = TestCase.for_choices([n.value for n in attempt])
        state.test_function(tc)
        i += 1
