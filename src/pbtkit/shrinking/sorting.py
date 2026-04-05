"""Sorting shrink pass for pbtkit.

Groups values by ChoiceType and tries sorting each group by sort key.
First attempts a full sort, then falls back to insertion sort where
each swap is validated.
"""

from __future__ import annotations

from collections import defaultdict

from pbtkit.core import (
    PbtkitState,
    shrink_pass,
)
from pbtkit.features import feature_enabled


@shrink_pass
def sort_values(state: PbtkitState) -> None:
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


def _try_sort_group(state: PbtkitState, choice_type: type, indices: list[int]) -> None:
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
    # The full sort above always succeeds for simple types (integers,
    # booleans); insertion sort is only needed when collections cause
    # structural changes during replacement that make the full sort fail.
    if feature_enabled("collections"):  # needed_for("collections")
        for pos in range(1, len(indices)):
            j = pos
            while j > 0:
                # Recompute valid indices since a prior swap may have
                # changed the result structure (e.g. via value punning).
                indices = [
                    i
                    for i in indices
                    if i < len(state.result)
                    and type(state.result[i].kind) == choice_type
                ]
                if j >= len(indices):  # needed_for("collections")
                    break
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


@shrink_pass
def swap_adjacent_blocks(state: PbtkitState) -> None:
    """Try swapping adjacent blocks of choices of the same size.

    This handles cases like dictionary entries where each entry spans
    multiple choices (e.g. [continue, key, value]) and the sorting
    pass can't swap individual values without breaking structure."""
    assert state.result is not None
    for block_size in range(2, 9):
        i = 0
        while i + 2 * block_size <= len(state.result):
            j = i + block_size
            # Only swap blocks with matching type structure.
            types_a = [type(state.result[i + k].kind) for k in range(block_size)]
            types_b = [type(state.result[j + k].kind) for k in range(block_size)]
            if types_a != types_b:
                i += 1
                continue
            block_a = [state.result[i + k].value for k in range(block_size)]
            block_b = [state.result[j + k].value for k in range(block_size)]
            if block_a == block_b:
                i += 1
                continue
            swap = {}
            for k in range(block_size):
                swap[i + k] = block_b[k]
                swap[j + k] = block_a[k]
            state.replace(swap)
            i += 1
