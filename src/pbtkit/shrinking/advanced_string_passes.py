"""Advanced string shrink passes for pbtkit.

Provides redistribute_string_pairs, which improves shrinking quality
for tests involving multiple string choices by transferring length
from earlier values to later ones.
"""

from __future__ import annotations

from pbtkit.core import (
    Shrinker,
    shrink_pass,
)
from pbtkit.shrinking.sequence_redistribution import redistribute_sequence_pair
from pbtkit.text import StringChoice


def _string_indices(shrinker: Shrinker) -> list[int]:
    """Return indices of all StringChoice nodes in the current target."""
    return [
        i
        for i, n in enumerate(shrinker.current.nodes)
        if isinstance(n.kind, StringChoice)
    ]


@shrink_pass
def redistribute_string_pairs(shrinker: Shrinker) -> None:
    """Try redistributing length between pairs of string values.

    For adjacent and skip-one-adjacent pairs of StringChoice nodes,
    try moving characters from the first to the second."""
    for gap in range(1, 3):
        idx = 0
        while idx < len(_string_indices(shrinker)) - gap:
            indices = _string_indices(shrinker)
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
