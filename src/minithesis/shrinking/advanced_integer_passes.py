"""Advanced integer shrink passes for minithesis.

This module provides sort_integer_ranges and redistribute_integers,
which improve shrinking quality for tests involving multiple integer
choices. They are imported by the package's __init__.py to register
as shrink passes.
"""

from __future__ import annotations

from minithesis.core import (
    IntegerChoice,
    MinithesisState,
    bin_search_down,
    shrink_pass,
)


@shrink_pass
def sort_integer_ranges(state: MinithesisState) -> None:
    """Try sorting out of order ranges of integer choices.
    sort(x) <= x, so this is always a lexicographic reduction."""
    assert state.result is not None
    k = 8
    while k > 1:
        for i in range(len(state.result) - k - 1, -1, -1):
            region = state.result[i : i + k]
            if not all(isinstance(n.kind, IntegerChoice) for n in region):
                continue
            state.consider(
                state.result[:i]
                + sorted(region, key=lambda n: n.value)
                + state.result[i + k :]
            )
        k -= 1


@shrink_pass
def redistribute_integers(state: MinithesisState) -> None:
    """Try adjusting nearby pairs of integer choices by
    redistributing value between them. Useful for tests that
    depend on the sum of some generated values."""
    assert state.result is not None
    for k in [2, 1]:
        for i in range(len(state.result) - 1 - k, -1, -1):
            j = i + k
            assert j < len(state.result)
            kind_i = state.result[i].kind
            kind_j = state.result[j].kind
            if not isinstance(kind_i, IntegerChoice) or not isinstance(
                kind_j, IntegerChoice
            ):
                continue
            if state.result[i].value > state.result[j].value:
                state.replace(
                    {
                        j: state.result[i].value,
                        i: state.result[j].value,
                    }
                )
            if j < len(state.result) and state.result[i].value > kind_i.min_value:
                previous_i = state.result[i].value
                previous_j = state.result[j].value
                bin_search_down(
                    kind_i.min_value,
                    previous_i,
                    lambda v: state.replace({i: v, j: previous_j + (previous_i - v)}),
                )
