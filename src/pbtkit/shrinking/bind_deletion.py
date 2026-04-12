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
    Shrinker,
    Status,
    TestCase,
    shrink_pass,
)


@shrink_pass
def bind_deletion(shrinker: Shrinker) -> None:
    """For each non-minimal value, try shrinking it and then deleting
    the excess choices that result from the shorter test case."""
    i = 0
    while i < len(shrinker.current.nodes):
        node = shrinker.current.nodes[i]
        if node.value == node.kind.simplest:
            i += 1
            continue

        value_shrinkers = VALUE_SHRINKERS.get(type(node.kind), [])
        for vs in value_shrinkers:
            if i >= len(shrinker.current.nodes):
                assert False, "index past result after shrinker in bind deletion"
            node = shrinker.current.nodes[i]
            vs(
                node.kind,
                node.value,
                lambda v: _try_replace_with_deletion(shrinker, i, v),
            )
        i += 1


def _try_replace_with_deletion(shrinker: Shrinker, idx: int, value: Any) -> bool:
    """Try replacing the value at idx. If the result is valid but
    not interesting, and the test case used fewer choices, try
    deleting regions after idx to recover an interesting result."""
    expected_len = len(shrinker.current.nodes)

    # First try a straight replace — if it's interesting, done.
    if shrinker.replace({idx: value}):
        return True

    # Build the attempt and run the test to see what happens.
    attempt = list(shrinker.current.nodes)
    assert idx < len(attempt)
    attempt[idx] = attempt[idx].with_value(value)
    choices = [n.value for n in attempt]
    test_case = TestCase.for_choices(choices)
    shrinker.test_function(test_case)
    assert test_case.status is not None

    if test_case.status != Status.VALID:
        return False

    actual_len = len(test_case.nodes)
    if actual_len >= expected_len:
        return False

    # The test case was shorter by k choices. Try deleting regions
    # of size <= k after idx, starting from near the end.
    k = expected_len - actual_len
    return _try_deletions(shrinker, attempt, idx, k)


def _try_deletions(
    shrinker: Shrinker,
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
            if shrinker.consider(candidate):
                return True
            j -= 1
    return False
