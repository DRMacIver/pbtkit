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
    PbtkitState,
    shrink_pass,
)


def _find_duplicate_groups(
    state: PbtkitState,
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


def _valid_indices(
    state: PbtkitState, indices: list[int], choice_type: type
) -> list[int]:
    """Filter indices to those still valid and matching the choice type."""
    assert state.result is not None
    return [
        i
        for i in indices
        if i < len(state.result) and isinstance(state.result[i].kind, choice_type)
    ]


@shrink_pass
def shrink_duplicates(state: PbtkitState) -> None:
    """Find duplicate (type, value) pairs and try shrinking them
    simultaneously using all registered value shrinkers."""
    assert state.result is not None

    for choice_type, indices in _find_duplicate_groups(state):
        shrinkers = VALUE_SHRINKERS.get(choice_type, [])
        for shrinker in shrinkers:
            # Re-validate: result may have changed since groups were computed.
            valid = _valid_indices(state, indices, choice_type)
            if len(valid) < 2:
                break
            node = state.result[valid[0]]
            # Try all at once.
            shrinker(
                node.kind,
                node.value,
                lambda v: state.replace({i: v for i in valid}),
            )
            # Try adjacent pairs (wrapping) if there are more than two.
            valid = _valid_indices(state, indices, choice_type)
            if len(valid) > 2:
                n = len(valid)
                for j in range(n):
                    a = valid[j]
                    b = valid[(j + 1) % n]
                    pair = _valid_indices(state, [a, b], choice_type)
                    assert len(pair) >= 2, "BUG: pair invalidated mid-loop"
                    node = state.result[a]
                    shrinker(
                        node.kind,
                        node.value,
                        lambda v: state.replace({a: v, b: v}),
                    )
