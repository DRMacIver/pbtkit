"""Advanced string shrink passes for minithesis.

Provides redistribute_string_pairs, which improves shrinking quality
for tests involving multiple string choices by transferring length
from earlier values to later ones.
"""

from __future__ import annotations

from minithesis.core import (
    MinithesisState,
    shrink_pass,
)
from minithesis.shrinking.sequence_redistribution import redistribute_sequence_pair
from minithesis.text import StringChoice


def _string_indices(state: MinithesisState) -> list[int]:
    """Return indices of all StringChoice nodes in the result."""
    assert state.result is not None
    return [i for i, n in enumerate(state.result) if isinstance(n.kind, StringChoice)]


@shrink_pass
def redistribute_string_pairs(state: MinithesisState) -> None:
    """Try redistributing length between pairs of string values.

    For adjacent and skip-one-adjacent pairs of StringChoice nodes,
    try moving characters from the first to the second."""
    assert state.result is not None
    for gap in range(1, 3):
        idx = 0
        while idx < len(_string_indices(state)) - gap:
            indices = _string_indices(state)
            i = indices[idx]
            j = indices[idx + gap]
            result = state.result
            redistribute_sequence_pair(
                result[i].value,
                result[j].value,
                lambda s, t: result[j].kind.validate(t) and state.replace({i: s, j: t}),
            )
            idx += 1
