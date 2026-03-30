"""Float support for minithesis.

This module provides FloatChoice, the draw_float method, float
serialization, and the float shrink pass. It is imported by the
package's __init__.py to register everything.
"""

from __future__ import annotations

import math
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from minithesis.core import (
    ChoiceType,
    TestCase,
    bin_search_down,
    value_shrinker,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAN_DRAW_PROBABILITY = 0.01


# ---------------------------------------------------------------------------
# Float helpers
# ---------------------------------------------------------------------------


def shrink_by_tens(n: int, condition: Callable[[int], bool]) -> int:
    """Shrinks an integer `n` subject to condition `condition` in a
    manner that will be particularly useful if the condition is sensitive
    to its decimal representation."""
    while n >= 10 and condition(n // 10):
        n //= 10
    k = 1
    while k < n:
        k *= 10
    while k >= 10:
        k //= 10
        while n >= k and condition(n - k):
            n -= k
    return n


def _draw_unbounded_float(random: Any) -> float:
    """Generate a random float from the full float space,
    excluding NaN (via rejection sampling). NaN is ~0.05%
    of bit patterns so this rarely loops."""
    while True:
        result = _lex_to_float(random.getrandbits(64))
        if not math.isnan(result):
            return result


def _draw_nan(random: Any) -> float:
    """Generate a random NaN value."""
    # Set exponent to all 1s and a random non-zero mantissa.
    exponent = 0x7FF << 52
    sign = random.getrandbits(1) << 63
    mantissa = random.getrandbits(52) or 1  # ensure non-zero
    return struct.unpack("!d", struct.pack("!Q", sign | exponent | mantissa))[0]


def _lex_to_float(bits: int) -> float:
    """Convert a lexicographically ordered 64-bit integer to a float.
    Used by the unbounded float generator to produce floats from
    random bit patterns, covering the full float space."""
    if bits >> 63:
        bits = bits ^ (1 << 63)
    else:
        bits = bits ^ ((1 << 64) - 1)
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


def _shortlex(s: str) -> tuple[int, str]:
    """Shortlex key: shorter strings are simpler, then lexicographic."""
    return (len(s), s)


def _parse_float_string(value: float) -> tuple[str, str, str, str]:
    """Parse a finite float's string representation into components.

    Returns (exp_part, frac_part, int_part, sign) where:
    - exp_part is the exponent string (e.g. "+20", "-10", or "")
    - frac_part is the fractional digits (e.g. "5", "0", or "")
    - int_part is the integer digits (e.g. "1", "100")
    - sign is "" for positive, "-" for negative
    """
    s = str(value)
    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:]
    if "e" in s:
        mantissa, exp_part = s.split("e")
    else:
        mantissa = s
        exp_part = ""
    if "." in mantissa:
        int_part, frac_part = mantissa.split(".")
    else:
        int_part = mantissa
        frac_part = ""
    return exp_part, frac_part, int_part, sign


def _float_string_key(value: float) -> tuple:
    """Sort key for finite floats based on their string representation.

    Compares the (exponent, fractional, integer) triple in shortlex
    order, with positive preferred over negative. For the exponent,
    positive exponents are simpler than negative ones."""
    exp_part, frac_part, int_part, sign = _parse_float_string(value)
    # For the exponent, split sign from magnitude.
    if exp_part.startswith("-"):
        exp_sign = 1  # negative exponents are less simple
        exp_abs = exp_part[1:]
    elif exp_part.startswith("+"):
        exp_sign = 0
        exp_abs = exp_part[1:]
    else:
        exp_sign = 0
        exp_abs = exp_part
    exp_key = (_shortlex(exp_abs), exp_sign)
    frac_key = _shortlex(frac_part)
    int_key = _shortlex(int_part)
    # Prefer positive over negative (0 for positive, 1 for negative)
    sign_key = 0 if sign == "" else 1
    return (exp_key, frac_key, int_key, sign_key)


# ---------------------------------------------------------------------------
# FloatChoice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloatChoice(ChoiceType[float]):
    min_value: float
    max_value: float
    allow_nan: bool
    allow_infinity: bool

    @property
    def simplest(self) -> float:
        # Prefer 0.0 if in range, otherwise the bound closest to 0.
        if self.min_value <= 0.0 <= self.max_value:
            return 0.0
        elif abs(self.min_value) <= abs(self.max_value):
            return self.min_value
        else:
            return self.max_value

    @property
    def unit(self) -> float:
        s = self.simplest
        if self.validate(s + 1.0):
            return s + 1.0
        if self.validate(s - 1.0):
            return s - 1.0
        return s

    def validate(self, value: float) -> bool:
        if not isinstance(value, float):
            return False
        if math.isnan(value):
            return self.allow_nan
        if math.isinf(value):
            return self.allow_infinity
        return self.min_value <= value <= self.max_value

    def sort_key(self, value: float) -> Any:
        """Order floats by human-readable simplicity.

        Finite < inf < -inf < NaN. Among finite floats, we compare
        by their string representation split into (exponent, fractional,
        integer) parts in shortlex order. Positive is preferred to
        negative."""
        if math.isnan(value):
            return (3,)
        if math.isinf(value):
            return (1,) if value > 0 else (2,)
        return (0, _float_string_key(value))


# ---------------------------------------------------------------------------
# draw_float — monkey-patched onto TestCase
# ---------------------------------------------------------------------------


def _draw_float(
    self: TestCase,
    min_value: float = -math.inf,
    max_value: float = math.inf,
    *,
    allow_nan: bool = True,
    allow_infinity: bool = True,
) -> float:
    """Returns a random float in [min_value, max_value].

    NaN is disallowed whenever any bound is set (since NaN is
    not comparable). For bounded ranges, generates uniformly.
    For half-bounded ranges, generates uniformly in the finite
    portion and occasionally returns the infinite bound. For
    fully unbounded ranges, generates via random bit patterns."""
    # Disallow NaN when any bound is set, since NaN is not
    # comparable to numeric bounds.
    if min_value != -math.inf or max_value != math.inf:
        allow_nan = False
    kind = FloatChoice(min_value, max_value, allow_nan, allow_infinity)

    bounded = math.isfinite(min_value) and math.isfinite(max_value)
    half_bounded = not bounded and (
        math.isfinite(min_value) or math.isfinite(max_value)
    )

    if bounded:

        def generate() -> float:
            return self.random.uniform(min_value, max_value)

    elif half_bounded:
        # One bound is finite, the other is +-inf. Generate a
        # non-negative float and add/subtract from the bound.

        def generate() -> float:
            if allow_infinity and self.random.random() < 0.05:
                return math.inf if max_value == math.inf else -math.inf
            magnitude = abs(_draw_unbounded_float(self.random))
            if math.isfinite(min_value):
                return min_value + magnitude
            else:
                return max_value - magnitude

    elif allow_nan:
        # Fully unbounded with NaN allowed. NaN is only ~0.05%
        # of bit patterns, so boost it to ~NAN_DRAW_PROBABILITY.
        def generate() -> float:
            if self.random.random() < NAN_DRAW_PROBABILITY:
                return _draw_nan(self.random)
            return _draw_unbounded_float(self.random)

    else:
        # Fully unbounded, no NaN.
        def generate() -> float:
            return _draw_unbounded_float(self.random)

    return self._make_choice(kind, generate)


# Attach draw_float to TestCase.
TestCase.draw_float = _draw_float


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Float shrink helper
# ---------------------------------------------------------------------------


@value_shrinker(FloatChoice)
def _shrink_float(
    kind: FloatChoice,
    value: float,
    try_replace: Callable[[float], bool],
) -> None:
    """Shrink a float choice toward human-readable simplicity.

    1. Replace special values (NaN -> inf -> finite)
    2. Try range edges
    3. If negative, try flipping sign
    4. Shrink string representation parts (exponent, fractional,
       integer) as integers
    """
    # We track the current value locally. When try_replace succeeds
    # with a new value v, we update our local tracking to v.
    current = [value]

    def try_float(f: float) -> bool:
        if kind.validate(f):
            if try_replace(f):
                current[0] = f
                return True
        return False

    # Step 1: Replace special values with simpler ones.
    if math.isnan(current[0]):
        for v in [math.inf, -math.inf, 0.0]:
            if try_float(v):
                return
        return
    if math.isinf(current[0]):
        if current[0] < 0:
            try_float(math.inf)
        assert math.isinf(current[0])
        try_float(sys.float_info.max if current[0] > 0 else -sys.float_info.max)

    # Step 2: Try range edges.
    if math.isfinite(kind.min_value):
        try_float(kind.min_value)
    if math.isfinite(kind.max_value):
        try_float(kind.max_value)

    # Step 3: If negative, try flipping sign.
    if current[0] < 0:
        try_float(-current[0])

    if not math.isfinite(current[0]):
        return

    # Step 4: Shrink string parts as integers. For negative
    # values (in forced-negative ranges), negate before parsing
    # and negate back when trying replacements.
    negate = current[0] < 0
    if negate:
        current[0] = -current[0]

    def try_positive(f: float) -> bool:
        return try_float(-f if negate else f)

    exp_part, frac_part, int_part, _ = _parse_float_string(current[0])
    if exp_part:
        exp_abs = exp_part.lstrip("+-")
        exp_sign = exp_part[0] if exp_part[0] in "+-" else ""
        if exp_sign == "-":
            try_positive(float(f"{int_part}.{frac_part}e{exp_abs}"))
        assert exp_abs  # Python's str() always has digits after 'e'
        bin_search_down(
            0,
            int(exp_abs),
            lambda e: try_positive(
                float(f"{int_part}.{frac_part}e{exp_sign}{e}")
                if e > 0
                else float(f"{int_part}.{frac_part}")
            ),
        )
        current[0] = abs(current[0])
        exp_part, frac_part, int_part, _ = _parse_float_string(current[0])

    if frac_part and frac_part != "0":
        # Try rounding up to remove the fraction entirely.
        # e.g. 1.1 -> 2.0, which may be simpler by sort key.
        try_positive(float(int(int_part) + 1))
        current[0] = abs(current[0])
        exp_part, frac_part, int_part, _ = _parse_float_string(current[0])

    if frac_part and frac_part != "0":
        reversed_frac = int(frac_part[::-1])
        reversed_frac = shrink_by_tens(
            reversed_frac,
            lambda rf: try_positive(float(f"{int_part}.{str(rf)[::-1]}")),
        )
        bin_search_down(
            0,
            reversed_frac,
            lambda rf: try_positive(float(f"{int_part}.{str(rf)[::-1]}")),
        )
        current[0] = abs(current[0])
        exp_part, frac_part, int_part, _ = _parse_float_string(current[0])

    if int_part and int(int_part) > 0:
        bin_search_down(
            0,
            int(int_part),
            lambda i_val: try_positive(
                float(f"{i_val}.{frac_part}") if frac_part else float(i_val)
            ),
        )


# ---------------------------------------------------------------------------
# Shrink pass
# ---------------------------------------------------------------------------
