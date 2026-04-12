"""Sorting shrink pass for pbtkit.

Groups values by ChoiceType and tries sorting each group by sort key.
First attempts a full sort, then falls back to insertion sort where
each swap is validated.
"""

from __future__ import annotations

from collections import defaultdict

from pbtkit.core import (
    Shrinker,
    shrink_pass,
)
from pbtkit.features import feature_enabled


@shrink_pass
def sort_values(shrinker: Shrinker) -> None:
    """Group values by choice type and try sorting each group."""
    # Process each choice type. Recompute groups each iteration
    # since sorting one group can change the current structure.
    processed: set[type] = set()
    while True:
        groups: dict[type, list[int]] = defaultdict(list)
        for i, node in enumerate(shrinker.current.nodes):
            groups[type(node.kind)].append(i)
        found = False
        for choice_type, indices in groups.items():
            if choice_type in processed or len(indices) < 2:
                continue
            found = True
            processed.add(choice_type)
            _try_sort_group(shrinker, choice_type, indices)
            break
        if not found:
            break


def _try_sort_group(shrinker: Shrinker, choice_type: type, indices: list[int]) -> None:
    """Try sorting the values at the given indices by the sort key
    of the first node's kind. First try a full sort, then fall back
    to insertion sort."""
    # Try a full sort.
    nodes = shrinker.current.nodes
    kind = nodes[indices[0]].kind
    values = [nodes[i].value for i in indices]
    sorted_values = sorted(values, key=kind.sort_key)
    if sorted_values != values:
        if shrinker.replace(dict(zip(indices, sorted_values))):
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
                # changed the current structure (e.g. via value punning).
                nodes = shrinker.current.nodes
                indices = [
                    i
                    for i in indices
                    if i < len(nodes) and type(nodes[i].kind) == choice_type
                ]
                if j >= len(indices):  # needed_for("collections")
                    break
                idx_j = indices[j]
                idx_prev = indices[j - 1]
                if nodes[idx_prev].sort_key <= nodes[idx_j].sort_key:
                    break
                if shrinker.replace(
                    {
                        idx_prev: nodes[idx_j].value,
                        idx_j: nodes[idx_prev].value,
                    }
                ):
                    j -= 1
                    continue
                break


@shrink_pass
def swap_adjacent_blocks(shrinker: Shrinker) -> None:
    """Try swapping adjacent blocks of choices of the same size.

    This handles cases like dictionary entries where each entry spans
    multiple choices (e.g. [continue, key, value]) and the sorting
    pass can't swap individual values without breaking structure."""
    for block_size in range(2, 9):
        i = 0
        while i + 2 * block_size <= len(shrinker.current.nodes):
            nodes = shrinker.current.nodes
            j = i + block_size
            # Only swap blocks with matching type structure.
            types_a = [type(nodes[i + k].kind) for k in range(block_size)]
            types_b = [type(nodes[j + k].kind) for k in range(block_size)]
            if types_a != types_b:
                i += 1
                continue
            block_a = [nodes[i + k].value for k in range(block_size)]
            block_b = [nodes[j + k].value for k in range(block_size)]
            if block_a == block_b:
                i += 1
                continue
            swap = {}
            for k in range(block_size):
                swap[i + k] = block_b[k]
                swap[j + k] = block_a[k]
            shrinker.replace(swap)
            i += 1
