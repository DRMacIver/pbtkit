"""Bytes support for minithesis.

This module provides BytesChoice, the draw_bytes method, bytes
serialization, and the bytes shrink pass. It is imported by the
package's __init__.py to register everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minithesis.minithesis import (
    ChoiceType,
    SerializationTag,
    TestCase,
    TestingState,
    _deserialize_length_prefixed,
    _shrink_sequence,
    register_serializer,
    shrink_pass,
)


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

register_serializer(
    BytesChoice,
    SerializationTag.BYTES,
    lambda v: len(v).to_bytes(4, "big") + v,
    _deserialize_length_prefixed(lambda b: bytes(b)),
)


# ---------------------------------------------------------------------------
# Shrink pass
# ---------------------------------------------------------------------------


@shrink_pass
def shrink_individual_bytes(state: TestingState) -> None:
    """Shrink each bytes choice: shorten, remove bytes,
    reduce byte values."""
    assert state.result is not None
    i = len(state.result) - 1
    while i >= 0:
        node = state.result[i]
        if isinstance(node.kind, BytesChoice):
            _shrink_sequence(
                node.value,
                node.kind.min_size,
                node.kind.simplest,
                lambda v, j: v[j],
                lambda v, j, e: v[:j] + bytes([e]) + v[j + 1 :],
                0,
                lambda v: state.replace({i: v}),
            )
        i -= 1
