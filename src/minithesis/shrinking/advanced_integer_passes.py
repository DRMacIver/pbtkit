"""Advanced integer shrink passes for minithesis.

This module provides sort_integer_ranges and redistribute_integers,
which improve shrinking quality for tests involving multiple integer
choices. They are imported by the package's __init__.py to register
as shrink passes.
"""

from __future__ import annotations

from typing import List

from minithesis.core import (
    IntegerChoice,
    MinithesisState,
    bin_search_down,
    shrink_pass,
)


@shrink_pass
def sort_integer_ranges(state: MinithesisState) -> None:
    """Try sorting ranges of integer choices by their sort key,
    skipping non-integer choices between them. Sorted order is
    always <= the original in shortlex, so this is a valid reduction."""
    assert state.result is not None
    result = state.result
    indices = _integer_indices(state)
    k = min(len(indices), 8)
    while k > 1:
        for start in range(len(indices) - k + 1):
            idx = indices[start : start + k]
            values = [result[i].value for i in idx]
            sorted_values = sorted(
                values, key=lambda v: result[idx[0]].kind.sort_key(v)
            )
            if sorted_values != values:
                state.replace(dict(zip(idx, sorted_values)))
        k -= 1


def _integer_indices(state: MinithesisState) -> List[int]:
    """Return indices of all IntegerChoice nodes in the result."""
    assert state.result is not None
    return [
        i for i, node in enumerate(state.result) if isinstance(node.kind, IntegerChoice)
    ]


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
            # Sort the pair by sort key (smaller absolute value first).
            if state.result[i].sort_key > state.result[j].sort_key:
                state.replace(
                    {
                        j: state.result[i].value,
                        i: state.result[j].value,
                    }
                )
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
