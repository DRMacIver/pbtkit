"""Sequence shrinking utilities for minithesis.

Provides shrink_sequence, used by the bytes and text value shrinkers
to shrink variable-length sequence choices (shorten, remove elements,
reduce element values).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from minithesis.core import bin_search_down


def shrink_sequence(
    value: Any,
    min_size: int,
    simplest: Any,
    element_value: Callable[[Any, int], int],
    replace_element: Callable[[Any, int, int], Any],
    min_element: int,
    try_replace: Callable[[Any], bool],
) -> None:
    """Shrink a sequence choice (string or bytes).

    Tries: simplest value, then shorten, then remove individual
    elements, then shrink each element value toward min_element.

    element_value: extract an integer from element j of sequence v
    replace_element: build a new sequence with element j replaced
    """
    # We track the current value locally. When try_replace succeeds
    # with a new value v, we update our local tracking to v.
    current = [value]

    def tracking_replace(v: Any) -> bool:
        if try_replace(v):
            current[0] = v
            return True
        return False

    if tracking_replace(simplest):
        return
    cur = current[0]
    bin_search_down(
        min_size,
        len(cur),
        lambda sz: tracking_replace(cur[:sz]),
    )
    for j in range(len(current[0]) - 1, -1, -1):
        v = current[0]
        if j < len(v) and len(v) > min_size:
            tracking_replace(v[:j] + v[j + 1 :])
    for j in range(len(current[0]) - 1, -1, -1):
        v = current[0]
        if j < len(v) and element_value(v, j) > min_element:
            bin_search_down(
                min_element,
                element_value(v, j),
                lambda e: tracking_replace(replace_element(current[0], j, e)),
            )
