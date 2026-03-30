"""Advanced integer shrink passes for minithesis.

This module provides redistribute_integers, which improves shrinking
quality for tests involving multiple integer choices by redistributing
value between pairs. It is imported by the package's __init__.py to
register as a shrink pass.
"""

from __future__ import annotations

from minithesis.core import (
    IntegerChoice,
    MinithesisState,
    TestCase,
    bin_search_down,
    shrink_pass,
)


def _integer_indices(state: MinithesisState) -> list[int]:
    """Return indices of all IntegerChoice nodes in the result."""
    assert state.result is not None
    return [
        i for i, node in enumerate(state.result) if isinstance(node.kind, IntegerChoice)
    ]


@shrink_pass
def lower_and_bump_adjacent(state: MinithesisState) -> None:
    """For adjacent IntegerChoice pairs where the first is > 0,
    try subtracting 1 from it and bumping the next. First runs
    the decrement alone to discover the max_value of the next
    node's kind in the new context, then tries the boundary."""
    assert state.result is not None
    indices = _integer_indices(state)
    for idx in range(len(indices) - 1):
        i = indices[idx]
        j = indices[idx + 1]
        assert i < len(state.result) and j < len(state.result)
        if state.result[i].value <= 0:
            continue
        new_i = state.result[i].value - 1
        # Run the decrement to observe the kind at position j.
        attempt = list(state.result)
        attempt[i] = attempt[i].with_value(new_i)
        tc = TestCase.for_choices([n.value for n in attempt])
        state.test_function(tc)
        assert tc.status is not None
        if j < len(tc.nodes):
            kind_j = tc.nodes[j].kind
            if isinstance(kind_j, IntegerChoice):
                # Try the boundary value directly.
                state.replace({i: new_i, j: kind_j.max_value})
        # Also try powers of 2 for cases where max_value is large.
        bump = 1
        while bump <= 256:
            new_j = state.result[j].value + bump
            if state.replace({i: new_i, j: new_j}):
                break
            bump *= 2


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
