# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
from random import Random

import pytest

import minithesis.floats

pytestmark = pytest.mark.requires("floats")
from minithesis import DirectoryDB, run_test
from minithesis.core import SerializationTag, Status, MinithesisState
from minithesis.core import TestCase as TC
from minithesis.floats import FloatChoice, _draw_unbounded_float
from minithesis.generators import floats


def test_floats_bounded():
    @run_test(database={})
    def _(tc):
        f = tc.any(floats(0.0, 1.0, allow_nan=False))
        assert 0.0 <= f <= 1.0


def test_floats_unbounded(monkeypatch):
    # Boost NaN probability so we reliably cover _draw_nan.
    monkeypatch.setattr(minithesis.floats, "NAN_DRAW_PROBABILITY", 0.5)

    @run_test(database={}, max_examples=200)
    def _(tc):
        tc.any(floats())


def test_floats_shrinks_to_zero(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            f = tc.any(floats(allow_nan=False))
            assert f == 0.0

    captured = capsys.readouterr()
    # Should shrink to a small non-zero float
    assert "any(floats" in captured.out


def test_floats_bounded_shrinks(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            f = tc.any(floats(1.0, 10.0, allow_nan=False))
            assert f < 5.0

    captured = capsys.readouterr()
    # Should find some float >= 5.0
    assert "any(floats(" in captured.out


def test_floats_no_nan():
    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.any(floats(allow_nan=False))
        assert not math.isnan(f)


def test_floats_no_infinity():
    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.any(floats(allow_infinity=False, allow_nan=False))
        assert math.isfinite(f)


def test_floats_negative_range():
    @run_test(database={})
    def _(tc):
        f = tc.any(floats(-10.0, -1.0, allow_nan=False))
        assert -10.0 <= f <= -1.0


def test_floats_shrinks_negative(capsys):
    """Floats in a negative-only range shrink toward the bound closest to 0."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            f = tc.any(floats(-10.0, -1.0, allow_nan=False))
            assert f > -5.0

    captured = capsys.readouterr()
    assert "any(floats(" in captured.out


def test_floats_shrinks_truncates(capsys):
    """Float shrinker tries to remove fractional parts."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            f = tc.any(floats(0.0, 100.0, allow_nan=False))
            assert f <= 1.0

    captured = capsys.readouterr()
    # Should shrink to a simple value (integer float)
    assert "any(floats(" in captured.out


def test_floats_half_bounded():
    """Test with one finite and one infinite bound, exercising
    the unbounded generation path with clamping."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.any(floats(min_value=0.0, allow_nan=False, allow_infinity=False))
        assert f >= 0.0
        assert math.isfinite(f)

    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.any(floats(max_value=0.0, allow_nan=False, allow_infinity=False))
        assert f <= 0.0
        assert math.isfinite(f)


def test_floats_database_round_trip(tmpdir):
    db = DirectoryDB(tmpdir)
    count = 0

    def run():
        with pytest.raises(AssertionError):

            @run_test(database=db)
            def _(test_case):
                nonlocal count
                count += 1
                f = test_case.any(floats(0.0, 10.0, allow_nan=False))
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
            f = tc.any(floats())
            assert not math.isnan(f) and abs(f) < 1e300


def test_floats_shrinks_scientific():
    """Float with scientific notation shrinks the exponent."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            f = tc.any(floats(allow_nan=False))
            assert abs(f) < 1e10


def test_floats_shrinks_negative_exponent():
    """Floats with negative exponents shrink the exponent."""

    def tf(tc):
        f = tc.draw_float(allow_nan=False)
        # Interesting if very small (forces scientific notation to persist)
        if 0 < f < 1e-100:
            tc.mark_status(Status.INTERESTING)

    state = MinithesisState(Random(0), tf, 1)
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
        f = tc.any(floats(min_value=0.0, allow_infinity=False))
        assert f >= 0.0
        assert math.isfinite(f)


def test_floats_half_bounded_max():
    """Half-bounded range with finite max generates correctly."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        f = tc.any(floats(max_value=0.0, allow_infinity=False))
        assert f <= 0.0
        assert math.isfinite(f)


def test_floats_half_bounded_with_infinity():
    """Half-bounded range can generate infinity."""
    found_inf = False

    @run_test(database={}, max_examples=1000)
    def _(tc):
        nonlocal found_inf
        f = tc.any(floats(min_value=0.0))
        if math.isinf(f):
            found_inf = True

    assert found_inf


def test_floats_shrinks_non_canonical():
    """Floats that aren't their canonical string representation
    get canonicalized during shrinking."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            f = tc.any(floats(0.0, 10.0, allow_nan=False))
            # Interesting for any non-zero value; tests that
            # non-canonical floats get cleaned up.
            assert f == 0.0


def test_floats_shrinks_nan_only():
    """When NaN is the only interesting value, it stays as NaN."""

    def tf(tc):
        f = tc.draw_float()
        if math.isnan(f):
            tc.mark_status(Status.INTERESTING)

    state = MinithesisState(Random(0), tf, 1)
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

    state = MinithesisState(Random(0), tf, 1)
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

    state = MinithesisState(Random(0), tf, 1)
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

    state = MinithesisState(Random(0), tf, 1)
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

    state = MinithesisState(Random(0), tf, 1)
    state.test_function(TC.for_choices([math.inf]))
    assert state.result is not None
    state.shrink()
    v = state.result[0].value
    assert math.isfinite(v) and v > 1e300


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

    state = MinithesisState(Random(0), tf, 1)
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

    state = MinithesisState(Random(0), tf, 1)
    state.test_function(TC.for_choices([0.3]))
    assert state.result is not None
    state.shrink()
    v = state.result[0].value
    assert 0.01 < v < 0.5


def test_draw_unbounded_float_rejects_nan():
    """_draw_unbounded_float rejects NaN via rejection sampling."""
    rnd = Random(0)
    # Call enough times that we're very likely to have rejected
    # at least one NaN internally.
    for _ in range(10000):
        f = _draw_unbounded_float(rnd)
        assert not math.isnan(f)
