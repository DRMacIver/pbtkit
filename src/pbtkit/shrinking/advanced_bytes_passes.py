"""Advanced bytes shrink passes for pbtkit.

Provides redistribute_bytes_pairs, which improves shrinking quality
for tests involving multiple bytes choices by transferring length
from earlier values to later ones.
"""

from __future__ import annotations

from pbtkit.bytes import BytesChoice
from pbtkit.core import (
    Shrinker,
    shrink_pass,
)
from pbtkit.shrinking.sequence_redistribution import redistribute_sequence_pair


def _bytes_indices(shrinker: Shrinker) -> list[int]:
    """Return indices of all BytesChoice nodes in the current target."""
    return [
        i
        for i, n in enumerate(shrinker.current.nodes)
        if isinstance(n.kind, BytesChoice)
    ]


@shrink_pass
def redistribute_bytes_pairs(shrinker: Shrinker) -> None:
    """Try redistributing length between pairs of bytes values.

    For adjacent and skip-one-adjacent pairs of BytesChoice nodes,
    try moving bytes from the first to the second."""
    for gap in range(1, 3):
        idx = 0
        while idx < len(_bytes_indices(shrinker)) - gap:
            indices = _bytes_indices(shrinker)
            i = indices[idx]
            j = indices[idx + gap]
            nodes = shrinker.current.nodes
            redistribute_sequence_pair(
                nodes[i].value,
                nodes[j].value,
                lambda s, t: (
                    nodes[j].kind.validate(t) and shrinker.replace({i: s, j: t})
                ),
            )
            idx += 1
