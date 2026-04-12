"""Advanced integer shrink passes for pbtkit.

Provides redistribute_integers, which improves shrinking quality
for tests involving multiple integer choices by redistributing
value between pairs.
"""

from __future__ import annotations

from pbtkit.core import (
    IntegerChoice,
    Shrinker,
    bin_search_down,
    shrink_pass,
)


def _integer_indices(shrinker: Shrinker) -> list[int]:
    """Return indices of all IntegerChoice nodes in the current target."""
    return [
        i
        for i, node in enumerate(shrinker.current.nodes)
        if isinstance(node.kind, IntegerChoice)
    ]


@shrink_pass
def redistribute_integers(shrinker: Shrinker) -> None:
    """Try adjusting pairs of integer choices by redistributing
    value between them. Operates on pairs of IntegerChoice nodes
    at various distances, skipping non-integer choices in between.
    Useful for tests that depend on the sum of some generated values."""
    indices = _integer_indices(shrinker)
    for gap in range(1, min(len(indices), 8)):
        for pair_idx in range(len(indices) - gap, 0, -1):
            indices = _integer_indices(shrinker)
            if pair_idx - 1 + gap >= len(indices):
                continue
            i = indices[pair_idx - 1]
            j = indices[pair_idx - 1 + gap]
            nodes = shrinker.current.nodes
            if nodes[i].value != nodes[i].kind.simplest:
                previous_i = nodes[i].value
                previous_j = nodes[j].value
                if previous_i > 0:
                    bin_search_down(
                        0,
                        previous_i,
                        lambda v: shrinker.replace(
                            {i: v, j: previous_j + (previous_i - v)}
                        ),
                    )
                else:
                    assert previous_i < 0
                    bin_search_down(
                        0,
                        -previous_i,
                        lambda a: shrinker.replace(
                            {i: -a, j: previous_j + (previous_i + a)}
                        ),
                    )
