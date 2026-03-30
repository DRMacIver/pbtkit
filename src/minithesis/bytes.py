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


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shrink pass
# ---------------------------------------------------------------------------
