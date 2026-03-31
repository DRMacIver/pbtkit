"""Common sequence redistribution logic for bytes and strings.

Provides redistribute_sequence_pair, used by both the advanced bytes
and advanced string passes to transfer length from one sequence value
to another while maintaining a test condition.
"""

from __future__ import annotations

from typing import AnyStr, Callable

from minithesis.core import bin_search_down


def redistribute_sequence_pair(
    s: AnyStr,
    t: AnyStr,
    condition: Callable[[AnyStr, AnyStr], bool],
) -> None:
    """Try to redistribute length from s to t to produce a sort-order
    smaller pair.

    Under shortlex ordering, shorter sequences are simpler. So we try
    to make s as short as possible by moving its suffix to the start
    of t.

    The condition function takes (new_s, new_t) and returns True if
    the replacement was accepted (i.e. the test was still interesting
    with the new values).
    """
    if len(s) == 0:
        return

    empty = s[:0]  # type-preserving empty: b'' for bytes, '' for str

    # Try moving everything from s to t.
    if condition(empty, s + t):
        return

    # Try moving the last character of s to the start of t.
    if not condition(s[:-1], s[-1:] + t):
        return

    # Binary search for the longest suffix of s that can be moved.
    # We know moving 1 char works. Find the most we can move.
    bin_search_down(
        1,
        len(s),
        lambda n: condition(s[:-n], s[-n:] + t),
    )
