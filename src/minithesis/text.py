"""String/text support for minithesis.

This module provides StringChoice, the draw_string method, string
serialization, and the string shrink pass. It is imported by the
package's __init__.py to register everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from minithesis.core import (
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
        return chr(self.min_codepoint) * self.min_size

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
        """Shortlex ordering: shorter is simpler, then by codepoints."""
        return (len(value), value)


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
        chars: List[str] = []
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

register_serializer(
    StringChoice,
    SerializationTag.STRING,
    lambda v: (e := v.encode("utf-8"), len(e).to_bytes(4, "big") + e)[1],
    _deserialize_length_prefixed(lambda b: b.decode("utf-8")),
)


# ---------------------------------------------------------------------------
# Shrink pass
# ---------------------------------------------------------------------------


@shrink_pass
def shrink_individual_strings(state: TestingState) -> None:
    """Shrink each string choice: shorten, remove chars,
    reduce codepoints."""
    assert state.result is not None
    i = len(state.result) - 1
    while i >= 0:
        node = state.result[i]
        if isinstance(node.kind, StringChoice):
            _shrink_sequence(
                node.value,
                node.kind.min_size,
                node.kind.simplest,
                lambda v, j: ord(v[j]),
                lambda v, j, e: v[:j] + chr(e) + v[j + 1 :],
                node.kind.min_codepoint,
                lambda v: state.replace({i: v}),
            )
        i -= 1
