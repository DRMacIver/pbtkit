"""Bind deletion shrink pass for pbtkit.

When a value controls the length of a downstream sequence (e.g.
via flat_map), reducing that value may shorten the test case without
making it interesting. This pass detects that situation and tries
deleting the now-excess choices to recover an interesting result.
"""

from __future__ import annotations

from typing import Any

from pbtkit.core import (
    VALUE_SHRINKERS,
    ChoiceNode,
    PbtkitState,
    Status,
    TestCase,
    shrink_pass,
)


@shrink_pass
def bind_deletion(state: PbtkitState) -> None:
    """For each non-minimal value, try shrinking it and then deleting
    the excess choices that result from the shorter test case."""
    assert state.result is not None

    i = 0
    while i < len(state.result):
        node = state.result[i]
        if node.value == node.kind.simplest:
            i += 1
            continue

        shrinkers = VALUE_SHRINKERS.get(type(node.kind), [])
        for shrinker in shrinkers:
            node = state.result[i]
            shrinker(
                node.kind,
                node.value,
                lambda v: _try_replace_with_deletion(state, i, v),
            )
        i += 1


def _try_replace_with_deletion(state: PbtkitState, idx: int, value: Any) -> bool:
    """Try replacing the value at idx. If the result is valid but
    not interesting, and the test case used fewer choices, try
    deleting regions after idx to recover an interesting result."""
    assert state.result is not None
    expected_len = len(state.result)

    # First try a straight replace — if it's interesting, done.
    if state.replace({idx: value}):
        return True

    # Build the attempt and run the test to see what happens.
    attempt = list(state.result)
    assert idx < len(attempt)
    attempt[idx] = attempt[idx].with_value(value)
    choices = [n.value for n in attempt]
    test_case = TestCase.for_choices(choices)
    state.test_function(test_case)
    assert test_case.status is not None

    if test_case.status != Status.VALID:
        return False

    actual_len = len(test_case.nodes)
    if actual_len >= expected_len:
        return False

    # The test case was shorter by k choices. Try deleting regions
    # of size <= k after idx, starting from near the end.
    k = expected_len - actual_len
    return _try_deletions(state, attempt, idx, k)


def _try_deletions(
    state: PbtkitState,
    attempt: list[ChoiceNode],
    idx: int,
    k: int,
) -> bool:
    """Try deleting regions of size 1..k after idx in the attempt,
    starting from the end."""
    for size in range(k, 0, -1):
        j = len(attempt) - size
        while j > idx:
            candidate = attempt[:j] + attempt[j + size :]
            if state.consider(candidate):
                return True
            j -= 1
    return False
