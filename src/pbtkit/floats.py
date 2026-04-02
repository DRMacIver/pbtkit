"""Float support for pbtkit.

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

from pbtkit.core import (
    ChoiceType,
    TestCase,
    bin_search_down,
    value_shrinker,
)
from pbtkit.features import feature_enabled, needed_for

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAN_DRAW_PROBABILITY = 0.01


# ---------------------------------------------------------------------------
# Float helpers
# ---------------------------------------------------------------------------


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


def _float_to_index(value: float) -> int:
    """Convert a finite float to a dense index based on IEEE 754 parts.

    Index 0 = +0.0, index 1 = -0.0. Then normal floats ordered by
    (exponent_rank, mantissa, sign) where exponent_rank zigzags from
    biased exponent 1023 (true exp 0). Subnormals come last (most complex).
    """
    bits = struct.unpack("!Q", struct.pack("!d", value))[0]
    sign = bits >> 63
    biased_exp = (bits >> 52) & 0x7FF
    mantissa = bits & ((1 << 52) - 1)

    # Zero is simplest.
    if biased_exp == 0 and mantissa == 0:
        return sign  # 0 for +0.0, 1 for -0.0

    # Subnormals (biased_exp=0, mantissa>0) come last.
    if biased_exp == 0:
        # After all normal floats. There are 2046 normal exponents.
        return 2 + 2046 * (2**53) + (mantissa - 1) * 2 + sign

    # Normal floats: zigzag exponent rank from 1023.
    if biased_exp == 1023:
        exp_rank = 0
    elif biased_exp > 1023:
        exp_rank = 2 * (biased_exp - 1023) - 1
    else:
        exp_rank = 2 * (1023 - biased_exp)

    return 2 + exp_rank * (2**53) + mantissa * 2 + sign


def _index_to_float(index: int) -> float:
    """Inverse of _float_to_index."""
    if index < 2:
        # Zero: index 0 = +0.0, index 1 = -0.0.
        bits = index << 63
        return struct.unpack("!d", struct.pack("!Q", bits))[0]

    index -= 2
    subnormal_start = 2046 * (2**53)
    if index >= subnormal_start:
        # Subnormal.
        index -= subnormal_start
        sign = index & 1
        mantissa = (index >> 1) + 1
        assert mantissa < (1 << 52)
        bits = (sign << 63) | mantissa
        return struct.unpack("!d", struct.pack("!Q", bits))[0]

    # Normal float.
    sign = index & 1
    index >>= 1
    mantissa = index & ((1 << 52) - 1)
    exp_rank = index >> 52

    # Inverse zigzag.
    if exp_rank == 0:
        biased_exp = 1023
    elif exp_rank % 2 == 1:
        biased_exp = 1023 + (exp_rank + 1) // 2
    else:
        biased_exp = 1023 - exp_rank // 2

    assert 1 <= biased_exp <= 2046

    bits = (sign << 63) | (biased_exp << 52) | mantissa
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


# 2 zeros + 2046 normal exponents × 2^53 slots + (2^52-1) subnormals × 2 signs
_MAX_FINITE_INDEX = 2 + 2046 * (2**53) + ((1 << 52) - 1) * 2 - 1


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
        # The simplest float is the valid one with the smallest raw index.
        # Try candidates in raw-index order: 0, -0, then for each
        # exponent rank try mantissa=0 positive and negative.
        # If mantissa=0 isn't in range but this exponent IS partially
        # in range, compute the boundary float.
        if self.validate(0.0):
            return 0.0

        best: float | None = None
        best_idx = _MAX_FINITE_INDEX + 1

        # Check boundaries — they're always valid and one of them
        # may have a very small index.
        for boundary in [self.min_value, self.max_value]:
            if math.isfinite(boundary):
                idx = _float_to_index(boundary)
                if idx < best_idx:
                    best = boundary
                    best_idx = idx

        # Check powers of 2 (mantissa=0) at each exponent rank.
        # These have the smallest index at each rank. Only need to
        # check ranks up to the current best.
        for exp_rank in range(2047):
            base_idx = 2 + exp_rank * (2**53)
            if base_idx >= best_idx:
                break  # All remaining ranks have larger indices.
            for sign in [0, 1]:
                v = _index_to_float(base_idx + sign)
                if not math.isnan(v) and self.validate(v):
                    if base_idx + sign < best_idx:
                        best = v
                        best_idx = base_idx + sign

        assert best is not None
        return best

    @property
    def unit(self) -> float:
        # The second-simplest value: find the next valid float after
        # simplest in the raw index order.
        s = self.simplest
        base = _float_to_index(s)
        for offset in range(1, 4):
            v = _index_to_float(base + offset)
            if not math.isnan(v) and self.validate(v):
                return v
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
        """Order floats by (exponent_rank, mantissa, sign).

        Finite < +inf < -inf < NaN. Among finite floats:
        - Exponents zigzag from 0: exp 0 < exp 1 < exp -1 < exp 2 < ...
        - Subnormals are most complex.
        - Within same exponent, smaller mantissa is simpler.
        - Within same exponent and mantissa, positive is simpler."""
        if math.isnan(value):
            return (1, 2)
        if math.isinf(value):
            return (1, 0) if value > 0 else (1, 1)
        return (0, _float_to_index(value))

    @needed_for("shrinking.index_passes")
    @property
    def max_index(self) -> int:
        return _MAX_FINITE_INDEX + 3  # +inf, -inf, nan

    @needed_for("indexing")
    def to_index(self, value: float) -> int:
        """Index in sort_key order (O(1)). Offset from simplest's raw index."""
        if math.isnan(value):
            bits = struct.unpack("!Q", struct.pack("!d", value))[0]
            nan_offset = bits & ((1 << 52) - 1)
            sign = bits >> 63
            return _MAX_FINITE_INDEX + 3 + nan_offset * 2 + sign
        if math.isinf(value):
            return _MAX_FINITE_INDEX + 1 if value > 0 else _MAX_FINITE_INDEX + 2
        base = _float_to_index(self.simplest)
        return _float_to_index(value) - base

    @needed_for("indexing")
    def from_index(self, index: int) -> float | None:
        """Inverse of to_index. Returns None for invalid indices.

        For bounded ranges, from_index(0) returns simplest, and
        higher indices are offset from the simplest's raw index."""
        if index > _MAX_FINITE_INDEX:
            offset = index - _MAX_FINITE_INDEX
            if offset == 1:
                return math.inf if self.allow_infinity else None
            if offset == 2:
                return -math.inf if self.allow_infinity else None
            if offset >= 3 and self.allow_nan:
                nan_rel = offset - 3
                sign = nan_rel & 1
                mantissa = (nan_rel >> 1) | (1 << 51)  # ensure non-zero mantissa
                bits = (sign << 63) | (0x7FF << 52) | mantissa
                value = struct.unpack("!d", struct.pack("!Q", bits))[0]
                return value if math.isnan(value) else None
            return None
        # For unbounded ranges, raw index == dense index.
        # For bounded ranges, offset from the simplest's raw index.
        base = _float_to_index(self.simplest)
        raw = base + index
        if raw > _MAX_FINITE_INDEX:
            return None
        value = _index_to_float(raw)
        if self.validate(value):
            return value
        return None


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
    # Disallow infinity when both bounds are finite — infinity is outside
    # any finite range.
    if math.isfinite(min_value) and math.isfinite(max_value):
        allow_infinity = False
    kind = FloatChoice(min_value, max_value, allow_nan, allow_infinity)

    bounded = math.isfinite(min_value) and math.isfinite(max_value)
    half_bounded = not bounded and (
        math.isfinite(min_value) or math.isfinite(max_value)
    )

    if bounded:

        def _base_generate() -> float:
            result = self.random.uniform(min_value, max_value)
            # random.uniform can overflow to inf for extreme ranges
            # (e.g. -float_info.max to float_info.max). Clamp to bounds.
            return max(min_value, min(max_value, result))

    elif half_bounded:

        def _base_generate() -> float:
            if allow_infinity and self.random.random() < 0.05:
                return math.inf if max_value == math.inf else -math.inf
            magnitude = abs(_draw_unbounded_float(self.random))
            if math.isfinite(min_value):
                return min_value + magnitude
            else:
                return max_value - magnitude

    elif allow_nan:

        def _base_generate() -> float:
            if self.random.random() < NAN_DRAW_PROBABILITY:
                return _draw_nan(self.random)
            return _draw_unbounded_float(self.random)

    else:

        def _base_generate() -> float:
            return _draw_unbounded_float(self.random)

    generate = _base_generate
    if feature_enabled("edge_case_boosting"):  # needed_for("edge_case_boosting")
        from pbtkit.edge_case_boosting import BOUNDARY_PROBABILITY

        # Candidates: boundaries, zero, infinities, NaN, small integers.
        # Filter to only those valid for this choice type.
        candidates = [
            min_value,
            max_value,
            0.0,
            -0.0,
            1.0,
            -1.0,
            math.inf,
            -math.inf,
            float("nan"),
            sys.float_info.min,
            sys.float_info.max,
            -sys.float_info.max,
        ]
        nasty_floats: list[float] = [v for v in candidates if kind.validate(v)]
        threshold = len(nasty_floats) * BOUNDARY_PROBABILITY

        def _boosted_generate() -> float:
            if self.random.random() < threshold:
                return self.random.choice(nasty_floats)
            return _base_generate()

        generate = _boosted_generate

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
    """Shrink a float choice toward simplest.

    Tries simplest, special value transitions, sign flip, exponent
    reduction, then mantissa binary search within exponent band."""
    # Try simplest first.
    try_replace(kind.simplest)

    # Step 1: Replace special values with simpler ones.
    if math.isnan(value):
        for v in [0.0, math.inf, -math.inf]:
            try_replace(v)
        return
    if math.isinf(value):
        if value < 0:
            try_replace(math.inf)
        try_replace(sys.float_info.max if value > 0 else -sys.float_info.max)

    # Step 2: If negative (or -0.0), try flipping sign.
    if value < 0 or math.copysign(1.0, value) < 0:
        try_replace(-value)

    if not math.isfinite(value):
        return

    # Step 3: Numeric reduction — shrink the integer part toward zero.
    # This catches cases like -8.0 → -5.0 that are obvious numerically
    # but hard to find via index or exponent/mantissa manipulation.
    if abs(value) >= 2.0:
        int_part = int(value)
        bin_search_down(
            0,
            abs(int_part),
            lambda n: try_replace(math.copysign(float(n), value) + (value - int_part)),
        )

    # Step 4: Binary search on the raw index toward simplest.
    # This handles cross-exponent transitions (e.g. -4.0 → -3.0)
    # where reducing the exponent alone doesn't work but a smaller
    # raw index with a different exponent+mantissa combination does.
    current_raw = _float_to_index(value)
    simplest_raw = _float_to_index(kind.simplest)
    bin_search_down(
        simplest_raw,
        current_raw,
        lambda idx: try_replace(_index_to_float(idx)),
    )

    # Step 4: Reduce exponent toward 1023 (binary search on biased_exp).
    # Re-read bits since step 3 may have changed the value.
    bits = struct.unpack("!Q", struct.pack("!d", value))[0]
    sign = bits >> 63
    biased_exp = (bits >> 52) & 0x7FF
    mantissa = bits & ((1 << 52) - 1)

    if biased_exp > 1023:
        bin_search_down(
            1023,
            biased_exp,
            lambda e: try_replace(
                struct.unpack("!d", struct.pack("!Q", (sign << 63) | (e << 52)))[0]
            ),
        )
    elif biased_exp < 1023 and biased_exp > 0:
        # For exponents below 1023, "simpler" means closer to 1023.
        bin_search_down(
            biased_exp,
            1023,
            lambda e: try_replace(
                struct.unpack("!d", struct.pack("!Q", (sign << 63) | (e << 52)))[0]
            ),
        )

    # Step 5: Binary search the mantissa toward 0 within current exponent.
    base_bits = (sign << 63) | (biased_exp << 52)
    try_replace(struct.unpack("!d", struct.pack("!Q", base_bits))[0])
    bin_search_down(
        0,
        mantissa,
        lambda m: try_replace(struct.unpack("!d", struct.pack("!Q", base_bits | m))[0]),
    )


# ---------------------------------------------------------------------------
# Shrink pass
# ---------------------------------------------------------------------------
