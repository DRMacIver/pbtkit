"""String/text support for minithesis.

This module provides StringChoice, the draw_string method, string
serialization, and the string shrink pass. It is imported by the
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


def _codepoint_key(c: int) -> int:
    """Map a codepoint to a sort key where ord('0') is simplest.

    Reorders the low 128 codepoints so that '0' maps to 0,
    '1' to 1, ..., '/' to 47, and anything above 127 keeps
    its natural position."""
    if c < 128:
        return (c - ord("0")) % 128
    return c


def _key_to_codepoint(k: int) -> int:
    """Inverse of _codepoint_key."""
    if k < 128:
        return (k + ord("0")) % 128
    return k


# ---------------------------------------------------------------------------
# StringChoice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StringChoice(ChoiceType[str]):
    min_codepoint: int
    max_codepoint: int
    min_size: int
    max_size: int

    @property
    def simplest(self) -> str:
        # The simplest codepoint in range under _codepoint_key ordering.
        best = min(
            range(self.min_codepoint, min(self.max_codepoint + 1, 128)),
            key=_codepoint_key,
            default=self.min_codepoint,
        )
        return chr(best) * self.min_size

    @property
    def unit(self) -> str:
        # Second-simplest in sort_key order: if min_size > 0, change
        # the last character to the second-simplest codepoint.
        # Otherwise, a single character of the second-simplest codepoint.
        # The second-simplest codepoint is '1' (key 1) for ASCII ranges.
        second_cp = _key_to_codepoint(1) if self.min_codepoint <= _key_to_codepoint(1) <= self.max_codepoint else self.min_codepoint
        if second_cp == _key_to_codepoint(0) and self.min_codepoint == self.max_codepoint:
            # Single-codepoint alphabet
            if self.min_size < self.max_size:
                return chr(self.min_codepoint) * (self.min_size + 1)
            return self.simplest
        if self.min_size > 0:
            return self.simplest[:-1] + chr(second_cp)
        return chr(second_cp) if self.max_size > 0 else self.simplest

    def validate(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        if not (self.min_size <= len(value) <= self.max_size):
            return False
        return all(
            self.min_codepoint <= ord(c) <= self.max_codepoint
            and not (0xD800 <= ord(c) <= 0xDFFF)
            for c in value
        )

    def sort_key(self, value: str) -> Any:
        """Shortlex ordering: shorter is simpler, then by mapped
        codepoint key (where '0' is simplest)."""
        return (len(value), tuple(_codepoint_key(ord(c)) for c in value))

    def _alphabet(self) -> list[int]:
        """Return valid codepoints sorted by _codepoint_key."""
        codepoints = [
            c
            for c in range(self.min_codepoint, self.max_codepoint + 1)
            if not (0xD800 <= c <= 0xDFFF)
        ]
        return sorted(codepoints, key=_codepoint_key)

    def to_index(self, value: str) -> int:
        """Shortlex index using mapped codepoint alphabet."""
        alphabet = self._alphabet()
        alpha_size = len(alphabet)
        # Build reverse lookup: codepoint → rank in alphabet
        rank = {c: i for i, c in enumerate(alphabet)}
        # Count all strings of lengths min_size .. len(value)-1
        offset = sum(
            alpha_size**length for length in range(self.min_size, len(value))
        )
        # Position within strings of this length (mixed-radix number)
        position = 0
        for ch in value:
            position = position * alpha_size + rank[ord(ch)]
        return offset + position

    def from_index(self, index: int) -> str | None:
        """Inverse of shortlex index."""
        alphabet = self._alphabet()
        alpha_size = len(alphabet)
        if alpha_size == 0:
            return "" if index == 0 and self.min_size == 0 else None
        remaining = index
        for length in range(self.min_size, self.max_size + 1):
            bucket_size = alpha_size**length
            if remaining < bucket_size:
                chars = []
                for _ in range(length):
                    chars.append(chr(alphabet[remaining % alpha_size]))
                    remaining //= alpha_size
                return "".join(reversed(chars))
            remaining -= bucket_size
        return None


# ---------------------------------------------------------------------------
# draw_string — monkey-patched onto TestCase
# ---------------------------------------------------------------------------


def _draw_string(
    self: TestCase,
    min_codepoint: int = 0,
    max_codepoint: int = 0x10FFFF,
    min_size: int = 0,
    max_size: int = 10,
) -> str:
    """Returns a random string with length in [min_size, max_size]
    and characters with codepoints in [min_codepoint, max_codepoint].
    Surrogates (0xD800-0xDFFF) are excluded."""
    # Compute the valid codepoint range excluding surrogates.
    if min_codepoint > max_codepoint:
        raise ValueError(f"Invalid codepoint range [{min_codepoint}, {max_codepoint}]")
    kind = StringChoice(min_codepoint, max_codepoint, min_size, max_size)

    def generate() -> str:
        length = self.random.randint(min_size, max_size)
        chars: list[str] = []
        for _ in range(length):
            # Rejection-sample to avoid surrogates (0xD800-0xDFFF).
            while True:
                cp = self.random.randint(min_codepoint, max_codepoint)
                if not (0xD800 <= cp <= 0xDFFF):
                    break
            chars.append(chr(cp))
        return "".join(chars)

    return self._make_choice(kind, generate)


# Attach draw_string to TestCase.
TestCase.draw_string = _draw_string


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shrink pass
# ---------------------------------------------------------------------------


@value_shrinker(StringChoice)
def shrink_string(
    kind: StringChoice,
    value: str,
    try_replace: Callable[[str], bool],
) -> None:
    """Shrink a string choice: shorten, remove chars, reduce
    codepoints toward '0' using the mapped codepoint ordering."""
    shrink_sequence(
        value,
        kind.min_size,
        kind.simplest,
        lambda v, j: _codepoint_key(ord(v[j])),
        lambda v, j, e: v[:j] + chr(_key_to_codepoint(e)) + v[j + 1 :],
        0,
        try_replace,
    )
