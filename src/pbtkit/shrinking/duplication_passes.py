"""Duplication shrink passes for pbtkit.

This module adds passes that reduce duplicate values simultaneously.
For each (ChoiceType, value) pair that appears more than once, it
applies all registered value shrinkers to the duplicates at the same
time, so that values which must stay equal can be shrunk together.
"""

from __future__ import annotations

from typing import Any

from pbtkit.core import (
    VALUE_SHRINKERS,
    Shrinker,
    shrink_pass,
)


def _find_duplicate_groups(
    shrinker: Shrinker,
) -> list[tuple[type, list[int]]]:
    """Find groups of indices sharing the same (type, value),
    with at least 2 members."""
    groups: dict[tuple[type, Any], list[int]] = {}
    for i, node in enumerate(shrinker.current.nodes):
        key = (type(node.kind), node.value)
        groups.setdefault(key, []).append(i)
    return [
        (choice_type, indices)
        for (choice_type, _), indices in groups.items()
        if len(indices) >= 2
    ]


def _valid_indices(
    shrinker: Shrinker, indices: list[int], choice_type: type
) -> list[int]:
    """Filter indices to those still valid and matching the choice type."""
    nodes = shrinker.current.nodes
    return [
        i for i in indices if i < len(nodes) and isinstance(nodes[i].kind, choice_type)
    ]


@shrink_pass
def shrink_duplicates(shrinker: Shrinker) -> None:
    """Find duplicate (type, value) pairs and try shrinking them
    simultaneously using all registered value shrinkers."""
    for choice_type, indices in _find_duplicate_groups(shrinker):
        value_shrinkers = VALUE_SHRINKERS.get(choice_type, [])
        for vs in value_shrinkers:
            # Re-validate: current may have changed since groups were computed.
            valid = _valid_indices(shrinker, indices, choice_type)
            if len(valid) < 2:
                break
            node = shrinker.current.nodes[valid[0]]
            # Try all at once.
            vs(
                node.kind,
                node.value,
                lambda v: shrinker.replace({i: v for i in valid}),
            )
            # Try adjacent pairs (wrapping) if there are more than two.
            valid = _valid_indices(shrinker, indices, choice_type)
            if len(valid) > 2:
                n = len(valid)
                for j in range(n):
                    a = valid[j]
                    b = valid[(j + 1) % n]
                    pair = _valid_indices(shrinker, [a, b], choice_type)
                    if len(pair) < 2:
                        assert False, "pair invalidated mid-loop in duplication pass"
                    node = shrinker.current.nodes[a]
                    vs(
                        node.kind,
                        node.value,
                        lambda v: shrinker.replace({a: v, b: v}),
                    )
