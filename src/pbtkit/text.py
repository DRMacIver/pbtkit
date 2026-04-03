"""String/text support for pbtkit.

This module provides StringChoice, the draw_string method, string
serialization, and the string shrink pass. It is imported by the
package's __init__.py to register everything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pbtkit.core import (
    ChoiceType,
    TestCase,
    value_shrinker,
)
from pbtkit.features import needed_for
from pbtkit.shrinking.sequence import shrink_sequence


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
        second_cp = (
            _key_to_codepoint(1)
            if self.min_codepoint <= _key_to_codepoint(1) <= self.max_codepoint
            else self.min_codepoint
        )
        if (
            second_cp == _key_to_codepoint(0)
            and self.min_codepoint == self.max_codepoint
        ):
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
        if not isinstance(value, str):
            # During shrinking, type changes can cause a non-string
            # value to land at a StringChoice position.
            return (0, ())
        return (len(value), tuple(_codepoint_key(ord(c)) for c in value))

    @needed_for("shrinking.index_passes")
    @property
    def max_index(self) -> int:
        alpha_size = self._alpha_size
        return (
            sum(
                alpha_size**length for length in range(self.min_size, self.max_size + 1)
            )
            - 1
        )

    @needed_for("indexing")
    @property
    def _alpha_size(self) -> int:
        """Count of valid codepoints in range, excluding surrogates."""
        total = self.max_codepoint - self.min_codepoint + 1
        # Subtract surrogates that fall within range.
        sur_lo = max(self.min_codepoint, 0xD800)
        sur_hi = min(self.max_codepoint, 0xDFFF)
        if sur_lo <= sur_hi:
            total -= sur_hi - sur_lo + 1
        return total

    @needed_for("indexing")
    def _codepoint_rank(self, codepoint: int) -> int:
        """Rank of a codepoint within valid codepoints sorted by key.

        O(1): counts how many valid codepoints have a strictly smaller
        _codepoint_key."""
        key = _codepoint_key(codepoint)
        # Count codepoints c in [min_codepoint, max_codepoint] (excluding
        # surrogates) with _codepoint_key(c) < key.
        count = 0
        # Low codepoints (< 128): _codepoint_key reorders them.
        lo = max(self.min_codepoint, 0)
        hi = min(self.max_codepoint, 127)
        if lo <= hi:
            # Keys for [lo, hi] are {_codepoint_key(c) for c in [lo, hi]}.
            # Count those < key.
            for c in range(lo, hi + 1):
                if _codepoint_key(c) < key:
                    count += 1
        # High codepoints (>= 128): _codepoint_key(c) == c, natural order.
        hi_lo = max(self.min_codepoint, 128)
        hi_hi = self.max_codepoint
        if hi_lo <= hi_hi:
            # These have key == c. Count those with c < key.
            if key > hi_lo:
                end = min(key - 1, hi_hi)
                n = end - hi_lo + 1
                # Subtract surrogates in [hi_lo, end].
                sur_lo = max(hi_lo, 0xD800)
                sur_hi = min(end, 0xDFFF)
                if sur_lo <= sur_hi:
                    n -= sur_hi - sur_lo + 1
                count += max(0, n)
        return count

    @needed_for("indexing")
    def _codepoint_at_rank(self, rank: int) -> int:
        """Codepoint at the given rank among valid codepoints sorted by key.

        O(n) where n = min(128, range width) for the low-codepoint portion,
        O(1) for high codepoints."""
        # Build sorted keys for the low portion (at most 128 entries).
        lo = max(self.min_codepoint, 0)
        hi = min(self.max_codepoint, 127)
        low_sorted = sorted(range(lo, hi + 1), key=_codepoint_key) if lo <= hi else []
        if rank < len(low_sorted):
            return low_sorted[rank]
        rank -= len(low_sorted)
        # High codepoints (>= 128) are in natural order, skip surrogates.
        hi_lo = max(self.min_codepoint, 128)
        c = hi_lo + rank
        # Skip past surrogates.
        if c >= 0xD800:
            c += 0xDFFF - 0xD800 + 1
        assert c <= self.max_codepoint
        return c

    @needed_for("indexing")
    def to_index(self, value: str) -> int:
        """Shortlex index using mapped codepoint alphabet."""
        alpha_size = self._alpha_size
        offset = sum(alpha_size**length for length in range(self.min_size, len(value)))
        position = 0
        for ch in value:
            position = position * alpha_size + self._codepoint_rank(ord(ch))
        return offset + position

    @needed_for("indexing")
    def from_index(self, index: int) -> str | None:
        """Inverse of shortlex index."""
        alpha_size = self._alpha_size
        assert alpha_size > 0
        remaining = index
        for length in range(self.min_size, self.max_size + 1):
            bucket_size = alpha_size**length
            if remaining < bucket_size:
                chars = []
                for _ in range(length):
                    chars.append(chr(self._codepoint_at_rank(remaining % alpha_size)))
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

    def _random_codepoint() -> int:
        """Draw a single valid codepoint, rejection-sampling surrogates."""
        while True:
            cp = self.random.randint(min_codepoint, max_codepoint)
            if not (0xD800 <= cp <= 0xDFFF):
                return cp

    def generate() -> str:
        # Build a small alphabet, then draw characters from it.
        # This massively boosts the chance of duplicate characters
        # (important for findability) while still covering the full
        # codepoint range.  Each alphabet entry has a 20% chance of
        # being ASCII (if ASCII is in range), otherwise uniform.
        ascii_lo = max(min_codepoint, 0)
        ascii_hi = min(max_codepoint, 127)
        has_ascii = ascii_lo <= ascii_hi
        alpha_size = self.random.randint(1, 10)
        alphabet: list[int] = []
        for _ in range(alpha_size):
            if has_ascii and self.random.random() < 0.2:
                alphabet.append(self.random.randint(ascii_lo, ascii_hi))
            else:
                alphabet.append(_random_codepoint())

        length = self.random.randint(min_size, max_size)
        return "".join(chr(self.random.choice(alphabet)) for _ in range(length))

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
