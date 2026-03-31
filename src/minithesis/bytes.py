"""Bytes support for minithesis.

This module provides BytesChoice, the draw_bytes method, bytes
serialization, and the bytes shrink pass. It is imported by the
package's __init__.py to register everything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from minithesis.core import (
    ChoiceType,
    TestCase,
    value_shrinker,
)
from minithesis.features import needed_for
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
        # Second-simplest in sort_key order: if min_size > 0, increment
        # the last byte. Otherwise, b'\x01' (first non-zero single byte).
        if self.min_size > 0:
            return b"\x00" * (self.min_size - 1) + b"\x01"
        return b"\x01" if self.max_size > 0 else self.simplest

    def validate(self, value: bytes) -> bool:
        return isinstance(value, bytes) and self.min_size <= len(value) <= self.max_size

    def sort_key(self, value: bytes) -> Any:
        """Shortlex ordering: shorter is simpler, then lexicographic."""
        return (len(value), value)

    @needed_for("indexing")
    @property
    def max_index(self) -> int:
        return self.to_index(b"\xff" * self.max_size)

    @needed_for("indexing")
    def to_index(self, value: bytes) -> int:
        """Shortlex index: count all shorter byte strings from min_size,
        then the position within strings of this length."""
        # Count all strings of lengths min_size .. len(value)-1
        offset = sum(256**length for length in range(self.min_size, len(value)))
        # Position within strings of this length (big-endian number)
        position = int.from_bytes(value, "big") if value else 0
        return offset + position

    @needed_for("indexing")
    def from_index(self, index: int) -> bytes | None:
        """Inverse of shortlex index."""
        # Find the length bucket
        remaining = index
        for length in range(self.min_size, self.max_size + 1):
            bucket_size = 256**length
            if remaining < bucket_size:
                # Decode the big-endian number within this length
                result = []
                for _ in range(length):
                    result.append(remaining % 256)
                    remaining //= 256
                return bytes(reversed(result))
            remaining -= bucket_size
        return None


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


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shrink pass
# ---------------------------------------------------------------------------
