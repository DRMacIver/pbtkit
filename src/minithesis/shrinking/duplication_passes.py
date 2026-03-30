"""Duplication shrink passes for minithesis.

This module adds passes that reduce duplicate values simultaneously.
For each (ChoiceType, value) pair that appears more than once, it
applies all registered value shrinkers to the duplicates at the same
time, so that values which must stay equal can be shrunk together.
"""

from __future__ import annotations

from typing import Any

from minithesis.core import (
    VALUE_SHRINKERS,
    MinithesisState,
    shrink_pass,
)


def _find_duplicate_groups(
    state: MinithesisState,
) -> list[tuple[type, list[int]]]:
    """Find groups of indices sharing the same (type, value),
    with at least 2 members."""
    assert state.result is not None
    groups: dict[tuple[type, Any], list[int]] = {}
    for i, node in enumerate(state.result):
        key = (type(node.kind), node.value)
        groups.setdefault(key, []).append(i)
    return [
        (choice_type, indices)
        for (choice_type, _), indices in groups.items()
        if len(indices) >= 2
    ]


@shrink_pass
def shrink_duplicates(state: MinithesisState) -> None:
    """Find duplicate (type, value) pairs and try shrinking them
    simultaneously using all registered value shrinkers."""
    assert state.result is not None

    for choice_type, indices in _find_duplicate_groups(state):
        shrinkers = VALUE_SHRINKERS.get(choice_type, [])
        for shrinker in shrinkers:
            node = state.result[indices[0]]
            # Try all at once.
            shrinker(
                node.kind,
                node.value,
                lambda v: state.replace({i: v for i in indices}),
            )
            # Try adjacent pairs (wrapping) if there are more than two.
            if len(indices) > 2:
                n = len(indices)
                for j in range(n):
                    a = indices[j]
                    b = indices[(j + 1) % n]
                    node = state.result[a]
                    shrinker(
                        node.kind,
                        node.value,
                        lambda v: state.replace({a: v, b: v}),
                    )
