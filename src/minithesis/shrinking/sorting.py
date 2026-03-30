"""Sorting shrink pass for minithesis.

Groups values by ChoiceType and tries sorting each group by sort key.
First attempts a full sort, then falls back to insertion sort where
each swap is validated.
"""

from __future__ import annotations

from collections import defaultdict

from minithesis.core import (
    MinithesisState,
    shrink_pass,
)


@shrink_pass
def sort_values(state: MinithesisState) -> None:
    """Group values by choice type and try sorting each group."""
    assert state.result is not None

    # Process each choice type. Recompute groups each iteration
    # since sorting one group can change the result structure.
    processed: set[type] = set()
    while True:
        groups: dict[type, list[int]] = defaultdict(list)
        for i, node in enumerate(state.result):
            groups[type(node.kind)].append(i)
        found = False
        for choice_type, indices in groups.items():
            if choice_type in processed or len(indices) < 2:
                continue
            found = True
            processed.add(choice_type)
            _try_sort_group(state, choice_type, indices)
            break
        if not found:
            break


def _try_sort_group(
    state: MinithesisState, choice_type: type, indices: list[int]
) -> None:
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
    for pos in range(1, len(indices)):
        j = pos
        while j > 0:
            idx_j = indices[j]
            idx_prev = indices[j - 1]
            # Recompute valid indices after each swap, since a
            # successful replace can change the result structure.
            indices = [
                i
                for i in indices
                if i < len(state.result) and type(state.result[i].kind) == choice_type
            ]
            assert j < len(indices)
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
            break
