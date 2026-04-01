# This file is part of Pbtkit, which may be found at
# https://github.com/DRMacIver/pbtkit
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
from random import Random

import pytest

import pbtkit.floats

pytestmark = pytest.mark.requires("floats")
import pbtkit.generators as gs
from pbtkit import DirectoryDB, run_test
from pbtkit.core import PbtkitState, Status
from pbtkit.core import TestCase as TC
from pbtkit.database import SerializationTag
from pbtkit.floats import _MAX_FINITE_INDEX, FloatChoice, _draw_unbounded_float


def test_floats_bounded():
    @run_test(database={})
    def _(tc):
        f = tc.draw(gs.floats(0.0, 1.0, allow_nan=False))
        assert 0.0 <= f <= 1.0


def test_floats_unbounded(monkeypatch):
    # Boost NaN probability so we reliably cover _draw_nan.
    monkeypatch.setattr(pbtkit.floats, "NAN_DRAW_PROBABILITY", 0.5)

    @run_test(database={}, max_examples=200)
    def _(tc):
        tc.draw(gs.floats())


def test_floats_shrinks_to_zero(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            f = tc.draw(gs.floats(allow_nan=False))
            assert f == 0.0

    captured = capsys.readouterr()
    # Should shrink to a small non-zero float
    assert " = " in captured.out


def test_floats_bounded_shrinks(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            f = tc.draw(gs.floats(1.0, 10.0, allow_nan=False))
            assert f < 5.0

    captured = capsys.readouterr()
    # Should find some float >= 5.0
    assert " = " in captured.out


def test_floats_no_nan():
    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.draw(gs.floats(allow_nan=False))
        assert not math.isnan(f)


def test_floats_no_infinity():
    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.draw(gs.floats(allow_infinity=False, allow_nan=False))
        assert math.isfinite(f)


def test_floats_negative_range():
    @run_test(database={})
    def _(tc):
        f = tc.draw(gs.floats(-10.0, -1.0, allow_nan=False))
        assert -10.0 <= f <= -1.0


def test_floats_shrinks_negative(capsys):
    """Floats in a negative-only range shrink toward the bound closest to 0."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            f = tc.draw(gs.floats(-10.0, -1.0, allow_nan=False))
            assert f > -5.0

    captured = capsys.readouterr()
    assert " = " in captured.out


def test_floats_shrinks_truncates(capsys):
    """Float shrinker tries to remove fractional parts."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            f = tc.draw(gs.floats(0.0, 100.0, allow_nan=False))
            assert f <= 1.0

    captured = capsys.readouterr()
    # Should shrink to a simple value (integer float)
    assert " = " in captured.out


def test_floats_half_bounded():
    """Test with one finite and one infinite bound, exercising
    the unbounded generation path with clamping."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.draw(gs.floats(min_value=0.0, allow_nan=False, allow_infinity=False))
        assert f >= 0.0
        assert math.isfinite(f)

    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.draw(gs.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
        assert f <= 0.0
        assert math.isfinite(f)


@pytest.mark.requires("database")
def test_floats_database_round_trip(tmpdir):
    db = DirectoryDB(tmpdir)
    count = 0

    def run():
        with pytest.raises(AssertionError):

            @run_test(database=db)
            def _(test_case):
                nonlocal count
                count += 1
                f = test_case.draw(gs.floats(0.0, 10.0, allow_nan=False))
                assert f < 5.0

    run()
    prev_count = count

    run()
    assert count == prev_count + 2


def test_floats_shrinks_large_or_nan():
    """Floats with extreme values shrink toward simpler ones."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            f = tc.draw(gs.floats())
            assert not math.isnan(f) and abs(f) < 1e300


def test_floats_shrinks_scientific():
    """Float with scientific notation shrinks the exponent."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            f = tc.draw(gs.floats(allow_nan=False))
            assert abs(f) < 1e10


def test_floats_shrinks_negative_exponent():
    """Floats with negative exponents shrink the exponent."""

    def tf(tc):
        f = tc.draw_float(allow_nan=False)
        # Interesting if very small (forces scientific notation to persist)
        if 0 < f < 1e-100:
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([1e-200]))
    assert state.result is not None
    state.shrink()
    # Should have shrunk the exponent toward a smaller magnitude
    v = state.result[0].value
    assert 0 < v < 1e-100


def test_floats_half_bounded_min():
    """Half-bounded range with finite min generates correctly."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.draw(gs.floats(min_value=0.0, allow_infinity=False))
        assert f >= 0.0
        assert math.isfinite(f)


def test_floats_half_bounded_max():
    """Half-bounded range with finite max generates correctly."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.draw(gs.floats(max_value=0.0, allow_infinity=False))
        assert f <= 0.0
        assert math.isfinite(f)


def test_floats_half_bounded_with_infinity():
    """Half-bounded range can generate infinity."""
    found_inf = False

    @run_test(database={}, max_examples=1000)
    def _(tc):
        nonlocal found_inf
        f = tc.draw(gs.floats(min_value=0.0))
        if math.isinf(f):
            found_inf = True

    assert found_inf


def test_floats_shrinks_non_canonical():
    """Floats that aren't their canonical string representation
    get canonicalized during shrinking."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            f = tc.draw(gs.floats(0.0, 10.0, allow_nan=False))
            # Interesting for any non-zero value; tests that
            # non-canonical floats get cleaned up.
            assert f == 0.0


def test_floats_shrinks_nan_only():
    """When NaN is the only interesting value, it stays as NaN."""

    def tf(tc):
        f = tc.draw_float()
        if math.isnan(f):
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([math.nan]))
    assert state.result is not None
    state.shrink()
    assert math.isnan(state.result[0].value)


def test_floats_shrinks_nan_to_simpler():
    """When NaN and other values are interesting, NaN shrinks
    to the simpler alternative."""

    def tf(tc):
        f = tc.draw_float()
        if math.isnan(f) or math.isinf(f):
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([math.nan]))
    assert state.result is not None
    state.shrink()
    assert state.result[0].value == math.inf


def test_floats_shrinks_neg_inf():
    """Negative infinity shrinks to positive infinity if that's
    also interesting."""

    def tf(tc):
        f = tc.draw_float()
        if math.isinf(f):
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([-math.inf]))
    assert state.result is not None
    state.shrink()
    assert state.result[0].value == math.inf


def test_floats_shrinks_neg_inf_to_finite():
    """-inf shrinks through inf to a large finite value."""

    def tf(tc):
        f = tc.draw_float(allow_nan=False)
        if abs(f) > 1e300:
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([-math.inf]))
    assert state.result is not None
    state.shrink()
    v = state.result[0].value
    assert math.isfinite(v) and abs(v) > 1e300


def test_floats_shrinks_inf_to_finite():
    """Infinity shrinks to a large finite value when only finite
    values > some threshold are interesting."""

    def tf(tc):
        f = tc.draw_float(allow_nan=False)
        if f > 1e300:
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([math.inf]))
    assert state.result is not None
    state.shrink()
    v = state.result[0].value
    assert math.isfinite(v) and v > 1e300


@pytest.mark.requires("database")
def test_floats_deserialize_truncated():
    """Truncated float in database is handled gracefully."""
    db = {"_": bytes([SerializationTag.FLOAT, 0x40, 0x09])}

    @run_test(database=db, max_examples=1)
    def _(test_case):
        pass


def test_floats_shrinks_large_exponent():
    """Floats like 1e+20 (no decimal point in string) are parsed
    and shrunk correctly."""

    def tf(tc):
        f = tc.draw_float(allow_nan=False)
        # Interesting if >= 1e15 — forces the shrinker to work
        # with scientific notation values.
        if f >= 1e15:
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([1e20]))
    assert state.result is not None
    state.shrink()
    v = state.result[0].value
    assert v >= 1e15


def test_floats_simplest_positive_range():
    """FloatChoice.simplest for a range not containing 0."""
    assert FloatChoice(1.0, 10.0, False, True).simplest == 1.0
    assert FloatChoice(-10.0, -1.0, False, True).simplest == -1.0
    assert FloatChoice(-1.0, 1.0, False, True).simplest == 0.0


def test_floats_validate_edge_cases():
    """FloatChoice.validate handles edge cases."""
    kind = FloatChoice(-math.inf, math.inf, True, True)
    assert kind.validate(math.nan)
    assert kind.validate(math.inf)
    assert kind.validate(-math.inf)
    assert kind.validate(0.0)
    assert not kind.validate("not a float")  # type: ignore[arg-type]

    no_nan = FloatChoice(-math.inf, math.inf, False, True)
    assert not no_nan.validate(math.nan)

    no_inf = FloatChoice(-math.inf, math.inf, True, False)
    assert not no_inf.validate(math.inf)
    assert not no_inf.validate(-math.inf)

    bounded = FloatChoice(0.0, 1.0, False, False)
    assert not bounded.validate(2.0)
    assert bounded.validate(0.5)


def test_floats_sort_key_ordering():
    """Float sort key produces the expected ordering."""
    kind = FloatChoice(-math.inf, math.inf, True, True)
    # Finite < inf < -inf < NaN
    assert kind.sort_key(0.0) < kind.sort_key(math.inf)
    assert kind.sort_key(math.inf) < kind.sort_key(-math.inf)
    assert kind.sort_key(-math.inf) < kind.sort_key(math.nan)
    # Simpler finite values first
    assert kind.sort_key(1.0) < kind.sort_key(2.0)
    assert kind.sort_key(1.0) < kind.sort_key(1.5)
    assert kind.sort_key(1.0) < kind.sort_key(-1.0)


def test_floats_shrinks_small_positive():
    """Floats < 1 reach step 4 with int_part '0', exercising
    the skip-integer-shrinking branch."""

    def tf(tc):
        f = tc.draw_float(0.0, 1.0, allow_nan=False)
        if 0.01 < f < 0.5:
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([0.3]))
    assert state.result is not None
    state.shrink()
    v = state.result[0].value
    assert 0.01 < v < 0.5


def test_shrinks_float_with_large_fractional():
    """Floats with many fractional digits use shrink_by_tens to
    reduce the reversed fractional part by factors of 10."""

    def tf(tc):
        f = tc.draw_float(0.0, 0.5, allow_nan=False)
        # Require a specific range that forces a non-trivial fractional.
        if 0.001 < f < 0.5:
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 1)
    state.test_function(TC.for_choices([0.123456789]))
    assert state.result is not None
    state.shrink()
    v = state.result[0].value
    assert 0.001 < v < 0.5


def test_draw_unbounded_float_rejects_nan():
    """_draw_unbounded_float rejects NaN via rejection sampling."""
    rnd = Random(0)
    # Call enough times that we're very likely to have rejected
    # at least one NaN internally.
    for _ in range(10000):
        f = _draw_unbounded_float(rnd)
        assert not math.isnan(f)


@pytest.mark.requires("shrinking.index_passes")
def test_float_index_subnormals():
    """Subnormals get high indices (most complex)."""
    fc = FloatChoice(float("-inf"), float("inf"), False, False)
    # 5e-324 is the smallest positive subnormal
    idx = fc.to_index(5e-324)
    # It should have a very high index (after all normals)
    assert idx > fc.to_index(1e300)
    # Roundtrip
    back = fc.from_index(idx)
    assert back == 5e-324


@pytest.mark.requires("shrinking.index_passes")
def test_float_index_bounded_simplest():
    """Bounded ranges find the correct simplest via power-of-2 search."""
    # Range where simplest is found via the power-of-2 search, not boundaries.
    fc = FloatChoice(0.5, 2.0, False, False)
    assert fc.simplest == 1.0
    assert fc.to_index(1.0) == 0


@pytest.mark.requires("shrinking.index_passes")
def test_float_from_index_inf():
    """from_index returns inf/nan for high indices."""
    fc = FloatChoice(float("-inf"), float("inf"), True, True)
    assert fc.from_index(_MAX_FINITE_INDEX + 1) == math.inf
    assert fc.from_index(_MAX_FINITE_INDEX + 2) == -math.inf
    v = fc.from_index(_MAX_FINITE_INDEX + 3)
    assert v is not None and math.isnan(v)


@pytest.mark.requires("shrinking.index_passes")
def test_float_from_index_past_max():
    """from_index returns None for huge indices."""
    fc = FloatChoice(0.0, 1.0, False, False)
    assert fc.from_index(10**20) is None
    # Also test a bounded range where base + index exceeds MAX.
    fc2 = FloatChoice(1e300, 2e300, False, False)
    assert fc2.from_index(_MAX_FINITE_INDEX) is None


@pytest.mark.requires("shrinking.index_passes")
def test_float_from_index_out_of_bounded_range():
    """from_index returns None for indices that map to floats outside
    the bounded range (e.g. negative floats for a positive-only range)."""
    fc = FloatChoice(1.0, 2.0, False, False)
    # Index 1 maps to -0.0 in the global ordering (simplest base is 1.0,
    # raw base + 1 = the next raw index which is -1.0). Validate fails.
    assert fc.from_index(1) is None


@pytest.mark.requires("shrinking.mutation")
def test_float_from_index_none_paths():
    """from_index returns None for various out-of-range cases.

    Covers paths that are only reachable when index_passes is enabled
    but also need coverage when only mutation is enabled."""
    fc = FloatChoice(float("-inf"), float("inf"), False, False)
    # offset == 1, allow_infinity=False: returns None (not math.inf)
    assert fc.from_index(_MAX_FINITE_INDEX + 1) is None
    # offset > 3, allow_nan=False: returns None
    assert fc.from_index(10**20) is None
    # bounded range where base + index exceeds _MAX_FINITE_INDEX
    fc2 = FloatChoice(1e300, 2e300, False, False)
    assert fc2.from_index(_MAX_FINITE_INDEX) is None


def test_float_simplest_with_inf_bounds():
    """simplest works when bounds are infinite."""
    fc = FloatChoice(float("-inf"), float("inf"), False, False)
    assert fc.simplest == 0.0
    fc2 = FloatChoice(1.0, float("inf"), False, False)
    assert fc2.simplest == 1.0
    fc3 = FloatChoice(float("-inf"), -1.0, False, False)
    assert fc3.simplest == -1.0


def test_float_simplest_tiny_range():
    """simplest for a tiny range where no power of 2 is in range."""
    fc = FloatChoice(1.5, 1.75, False, False)
    assert fc.simplest == 1.5


def test_float_simplest_subnormal_range():
    """simplest for a subnormal-only range exhausts the exp_rank search."""
    # 1e-323 has mantissa > 1, so its index exceeds the last normal
    # exponent's base_idx, causing the power-of-2 loop to exhaust.
    fc = FloatChoice(1e-323, 2e-323, False, False)
    assert fc.simplest == 1e-323


def test_float_simplest_finds_power_of_two():
    """simplest finds a power of 2 inside the range when boundaries
    don't have the smallest index."""
    fc = FloatChoice(0.5, 2.0, False, False)
    assert fc.simplest == 1.0  # 2^0, found by power-of-2 search


def test_float_negative_zero_simplest():
    """When only -0.0 is valid, simplest returns -0.0."""
    # A range that contains -0.0 but not 0.0 is impossible with finite bounds.
    # But we can test via the validate path.
    fc = FloatChoice(-1.0, 0.0, False, False)
    # 0.0 validates (0.0 >= -1.0 and 0.0 <= 0.0), so simplest is 0.0
    assert fc.simplest == 0.0


def test_float_shrinks_across_exponent_boundary():
    """The float shrinker must find values across exponent boundaries.
    E.g. shrinking -4.0 (exp=1025, mantissa=0) toward -2.0 requires
    finding -3.0 (exp=1024, mantissa=2^51) or -2.0...004 (exp=1024,
    mantissa=1). Binary search on raw index handles this.
    Regression: shrinker was stuck at -4.0 because exponent and mantissa
    searches couldn't cross the boundary independently."""

    def tf(tc):
        v0 = tc.draw(gs.floats(allow_nan=False, allow_infinity=False))
        if v0 < -2.0:
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    v = state.result[0].value
    # Should find the smallest float < -2.0, which is -2.0 - 1 ULP.
    assert -3.0 < v < -2.0


def test_float_choice_unit():
    # Under (exponent_rank, mantissa, sign) ordering:
    # index 0 = 0.0, index 1 = -0.0, index 2 = 1.0, ...
    assert FloatChoice(-10.0, 10.0, False, False).unit == -0.0
    # Negative-only range: simplest has smallest exp_rank in range.
    fc = FloatChoice(-10.0, -5.0, False, False)
    assert fc.simplest == -5.0  # exp_rank=3, simpler than -8.0 (exp_rank=5)
    # Single-value range: falls back to simplest.
    assert FloatChoice(5.0, 5.0, False, False).unit == 5.0
