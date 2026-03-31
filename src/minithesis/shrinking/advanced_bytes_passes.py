"""Advanced bytes shrink passes for minithesis.

Provides redistribute_bytes_pairs, which improves shrinking quality
for tests involving multiple bytes choices by transferring length
from earlier values to later ones.
"""

from __future__ import annotations

from minithesis.bytes import BytesChoice
from minithesis.core import (
    MinithesisState,
    shrink_pass,
)
from minithesis.shrinking.sequence_redistribution import redistribute_sequence_pair


def _bytes_indices(state: MinithesisState) -> list[int]:
    """Return indices of all BytesChoice nodes in the result."""
    assert state.result is not None
    return [i for i, n in enumerate(state.result) if isinstance(n.kind, BytesChoice)]


@shrink_pass
def redistribute_bytes_pairs(state: MinithesisState) -> None:
    """Try redistributing length between pairs of bytes values.

    For adjacent and skip-one-adjacent pairs of BytesChoice nodes,
    try moving bytes from the first to the second."""
    assert state.result is not None
    for gap in range(1, 3):
        idx = 0
        while idx < len(_bytes_indices(state)) - gap:
            indices = _bytes_indices(state)
            i = indices[idx]
            j = indices[idx + gap]
            result = state.result
            redistribute_sequence_pair(
                result[i].value,
                result[j].value,
                lambda s, t: result[j].kind.validate(t) and state.replace({i: s, j: t}),
            )
            idx += 1
