"""Sorting shrink pass for minithesis.

Groups values by ChoiceType and tries sorting each group by sort key.
First attempts a full sort, then falls back to insertion sort where
each swap is validated.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from minithesis.core import (
    MinithesisState,
    shrink_pass,
)


@shrink_pass
def sort_values(state: MinithesisState) -> None:
    """Group values by choice type and try sorting each group."""
    assert state.result is not None

    # Group indices by choice type.
    groups: Dict[type, List[int]] = defaultdict(list)
    for i, node in enumerate(state.result):
        groups[type(node.kind)].append(i)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        _try_sort_group(state, indices)


def _try_sort_group(state: MinithesisState, indices: List[int]) -> None:
    """Try sorting the values at the given indices by the sort key
    of the first node's kind. First try a full sort, then fall back
    to insertion sort."""
    assert state.result is not None

    # Try a full sort.
    kind = state.result[indices[0]].kind
    values = [state.result[i].value for i in indices]
    sorted_values = sorted(values, key=kind.sort_key)
    if sorted_values != values:
        if state.replace(dict(zip(indices, sorted_values))):
            return

    # Fall back to insertion sort: for each element, swap it
    # backward until it's in the right place or a swap fails.
    # If a swap fails (e.g. due to structural dependency between
    # adjacent positions), skip one position and try swapping
    # with the element one further back.
    for pos in range(1, len(indices)):
        j = pos
        while j > 0:
            idx_j = indices[j]
            idx_prev = indices[j - 1]
            if state.result[idx_prev].sort_key <= state.result[idx_j].sort_key:
                break
            if state.replace(
                {
                    idx_prev: state.result[idx_j].value,
                    idx_j: state.result[idx_prev].value,
                }
            ):
                j -= 1
                continue
            # Adjacent swap failed; try skipping one position.
            if j >= 2:
                idx_skip = indices[j - 2]
                if state.result[idx_skip].sort_key > state.result[idx_j].sort_key:
                    if state.replace(
                        {
                            idx_skip: state.result[idx_j].value,
                            idx_j: state.result[idx_skip].value,
                        }
                    ):
                        j -= 2
                        continue
            break
