"""Advanced integer shrink passes for minithesis.

This module provides redistribute_integers, which improves shrinking
quality for tests involving multiple integer choices by redistributing
value between pairs. It is imported by the package's __init__.py to
register as a shrink pass.
"""

from __future__ import annotations

from typing import List

from minithesis.core import (
    IntegerChoice,
    MinithesisState,
    bin_search_down,
    shrink_pass,
)


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
