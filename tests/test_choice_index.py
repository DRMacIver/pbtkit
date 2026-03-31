"""Hypothesis property tests for ChoiceType.to_index / from_index invariants."""

import math
import struct

import pytest

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from minithesis.bytes import BytesChoice
from minithesis.core import BooleanChoice, IntegerChoice
from minithesis.floats import FloatChoice
from minithesis.text import StringChoice

pytestmark = [pytest.mark.hypothesis]


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


@st.composite
def integer_choices(draw):
    lo = draw(st.integers(min_value=-(2**16), max_value=2**16))
    hi = draw(st.integers(min_value=lo, max_value=lo + draw(st.integers(0, 2**16))))
    return IntegerChoice(lo, hi)


@st.composite
def integer_choice_and_value(draw):
    kind = draw(integer_choices())
    value = draw(st.integers(min_value=kind.min_value, max_value=kind.max_value))
    return kind, value


@st.composite
def boolean_choice_and_value(draw):
    p = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    kind = BooleanChoice(p)
    value = draw(st.booleans())
    return kind, value


@st.composite
def bytes_choices(draw):
    min_size = draw(st.integers(0, 4))
    max_size = draw(st.integers(min_size, min_size + draw(st.integers(0, 4))))
    return BytesChoice(min_size, max_size)


@st.composite
def bytes_choice_and_value(draw):
    kind = draw(bytes_choices())
    length = draw(st.integers(min_value=kind.min_size, max_value=kind.max_size))
    value = draw(st.binary(min_size=length, max_size=length))
    return kind, value


@st.composite
def string_choices(draw):
    min_cp = draw(st.integers(32, 126))
    max_cp = draw(st.integers(min_cp, min(min_cp + 20, 126)))
    min_size = draw(st.integers(0, 3))
    max_size = draw(st.integers(min_size, min_size + draw(st.integers(0, 3))))
    return StringChoice(min_cp, max_cp, min_size, max_size)


@st.composite
def string_choice_and_value(draw):
    kind = draw(string_choices())
    length = draw(st.integers(min_value=kind.min_size, max_value=kind.max_size))
    alphabet = st.sampled_from(
        [
            chr(c)
            for c in range(kind.min_codepoint, kind.max_codepoint + 1)
            if not (0xD800 <= c <= 0xDFFF)
        ]
    )
    value = draw(st.text(alphabet=alphabet, min_size=length, max_size=length))
    return kind, value


@st.composite
def float_choice_and_value(draw):
    lo = draw(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
    hi = draw(st.floats(allow_nan=False, allow_infinity=False, min_value=lo, max_value=lo + 1e6))
    kind = FloatChoice(lo, hi, allow_nan=False, allow_infinity=False)
    value = draw(st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False))
    assume(kind.validate(value))
    return kind, value


# Combine all types
any_choice_and_value = (
    integer_choice_and_value()
    | boolean_choice_and_value()
    | bytes_choice_and_value()
    | string_choice_and_value()
    | float_choice_and_value()
)


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------


@given(data=any_choice_and_value)
@settings(max_examples=500)
def test_from_index_zero_is_simplest(data):
    kind, _ = data
    assert kind.from_index(0) == kind.simplest


@given(data=any_choice_and_value)
@settings(max_examples=500)
def test_from_index_one_is_second_simplest(data):
    """from_index(1) should be the second value in sort_key order
    (if it exists)."""
    kind, _ = data
    result = kind.from_index(1)
    if result is None:
        return  # Single-value type
    # It must be strictly greater than simplest in sort_key
    assert kind.sort_key(result) > kind.sort_key(kind.simplest)


@given(data=any_choice_and_value)
@settings(max_examples=500)
def test_roundtrip_value(data):
    """from_index(to_index(v)) == v for all valid values."""
    kind, value = data
    index = kind.to_index(value)
    back = kind.from_index(index)
    if isinstance(value, float) and math.isnan(value):
        # NaN: check bitwise equality
        assert back is not None
        assert struct.pack("!d", back) == struct.pack("!d", value)
    else:
        assert back == value


@given(data=any_choice_and_value)
@settings(max_examples=500)
def test_to_index_non_negative(data):
    kind, value = data
    assert kind.to_index(value) >= 0


@given(data=any_choice_and_value, extra=st.integers(min_value=0, max_value=100))
@settings(max_examples=500)
def test_from_index_then_to_index_le(data, extra):
    """to_index(from_index(i)) <= i when from_index(i) is not None."""
    kind, _ = data
    index = extra
    value = kind.from_index(index)
    if value is not None:
        assert kind.to_index(value) <= index


@given(data=any_choice_and_value)
@settings(max_examples=500)
def test_order_preserving(data):
    """If sort_key(x) <= sort_key(y) then to_index(x) <= to_index(y)."""
    kind, x = data
    # Generate a second valid value by bumping the index
    idx_x = kind.to_index(x)
    y = kind.from_index(idx_x + 1)
    if y is not None:
        assert kind.sort_key(x) <= kind.sort_key(y)
        assert kind.to_index(x) <= kind.to_index(y)
