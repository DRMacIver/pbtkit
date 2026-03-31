"""Bytes support for minithesis.

This module provides BytesChoice, the draw_bytes method, bytes
serialization, and the bytes shrink pass. It is imported by the
package's __init__.py to register everything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from minithesis.core import (
    ChoiceType,
    TestCase,
    shrink_pass,
    value_shrinker,
)

if TYPE_CHECKING:
    from minithesis.core import MinithesisState
from minithesis.shrinking.sequence import shrink_sequence

# ---------------------------------------------------------------------------
# BytesChoice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BytesChoice(ChoiceType[bytes]):
    min_size: int
    max_size: int

    @property
    def simplest(self) -> bytes:
        return b"\x00" * self.min_size

    @property
    def unit(self) -> bytes:
        # One byte longer than simplest.
        return self.simplest + b"\x00"

    def validate(self, value: bytes) -> bool:
        return isinstance(value, bytes) and self.min_size <= len(value) <= self.max_size

    def sort_key(self, value: bytes) -> Any:
        """Shortlex ordering: shorter is simpler, then lexicographic."""
        return (len(value), value)


@value_shrinker(BytesChoice)
def shrink_bytes(
    kind: BytesChoice,
    value: bytes,
    try_replace: Callable[[bytes], bool],
) -> None:
    """Shrink a bytes choice: shorten, remove bytes, reduce byte values."""
    shrink_sequence(
        value,
        kind.min_size,
        kind.simplest,
        lambda v, j: v[j],
        lambda v, j, e: v[:j] + bytes([e]) + v[j + 1 :],
        0,
        try_replace,
    )


# ---------------------------------------------------------------------------
# draw_bytes — monkey-patched onto TestCase
# ---------------------------------------------------------------------------


def _draw_bytes(self: TestCase, min_size: int, max_size: int) -> bytes:
    """Returns a random byte string with length in [min_size, max_size]."""
    return self._make_choice(
        BytesChoice(min_size, max_size),
        lambda: bytes(
            self.random.randint(0, 255)
            for _ in range(self.random.randint(min_size, max_size))
        ),
    )


# Attach draw_bytes to TestCase.
TestCase.draw_bytes = _draw_bytes


@shrink_pass
def redistribute_bytes(state: "MinithesisState") -> None:
    """Try shortening one bytes value and lengthening another.

    For pairs of BytesChoice nodes, transfer length from the first
    to the second while keeping the total constant."""
    assert state.result is not None
    byte_indices = [
        i for i, n in enumerate(state.result) if isinstance(n.kind, BytesChoice)
    ]
    for gap in range(1, min(len(byte_indices), 4)):
        for pair_idx in range(len(byte_indices) - gap):
            # Recompute indices since prior replacements may change structure.
            byte_indices = [
                i for i, n in enumerate(state.result) if isinstance(n.kind, BytesChoice)
            ]
            assert pair_idx + gap < len(byte_indices)
            i = byte_indices[pair_idx]
            j = byte_indices[pair_idx + gap]
            vi = state.result[i].value
            vj = state.result[j].value
            assert isinstance(vi, bytes) and isinstance(vj, bytes)
            if len(vi) == 0:
                continue
            # Try transferring all bytes from i to j.
            new_j = vj + b"\x00" * len(vi)
            if state.result[j].kind.validate(new_j):
                state.replace({i: state.result[i].kind.simplest, j: new_j})


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shrink pass
# ---------------------------------------------------------------------------
