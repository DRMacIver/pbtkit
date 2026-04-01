# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
import struct
from random import Random

import pytest

import minithesis.core as core
import minithesis.generators as gs
from minithesis import Generator, Unsatisfiable, run_test
from minithesis.bytes import BytesChoice
from minithesis.caching import CachedTestFunction, _cache_key
from minithesis.core import (
    Frozen,
    IntegerChoice,
    Status,
)
from minithesis.core import MinithesisState as State
from minithesis.core import TestCase as TC
from minithesis.database import DirectoryDB
from minithesis.floats import FloatChoice
from minithesis.shrinking.index_passes import lower_and_bump


@pytest.mark.parametrize("seed", range(10))
def test_finds_small_list_even_with_bad_lists(capsys, seed):
    """Minithesis can't really handle shrinking arbitrary
    monadic bind, but length parameters are a common case
    of monadic bind that it has a little bit of special
    casing for. This test ensures that that special casing
    works.

    The problem is that if you generate a list by drawing
    a length and then drawing that many elements, you can
    end up with something like ``[1001, 0, 0]`` then
    deleting those zeroes in the middle is a pain. minithesis
    will solve this by first sorting those elements, so that
    we have ``[0, 0, 1001]``, and then lowering the length
    by two, turning it into ``[1001]`` as desired.
    """

    with pytest.raises(AssertionError):

        @Generator
        def bad_list(test_case):
            n = test_case.choice(10)
            return [test_case.choice(10000) for _ in range(n)]

        @run_test(database={}, random=Random(seed))
        def _(test_case):
            ls = test_case.any(bad_list)
            assert sum(ls) <= 1000

    captured = capsys.readouterr()

    assert captured.out.strip() == "any(bad_list): [1001]"


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_reduces_additive_pairs(capsys):

    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=10000)
        def _(test_case):
            m = test_case.choice(1000)
            n = test_case.choice(1000)
            assert m + n <= 1000

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 1",
        "choice(1000): 1000",
    ]


@pytest.mark.requires("database")
def test_reuses_results_from_the_database(tmpdir):
    db = DirectoryDB(tmpdir)
    count = 0

    def run():
        with pytest.raises(AssertionError):

            @run_test(database=db)
            def _(test_case):
                nonlocal count
                count += 1
                assert test_case.choice(10000) < 10

    run()

    assert len(tmpdir.listdir()) == 1
    prev_count = count

    run()

    assert len(tmpdir.listdir()) == 1
    assert count == prev_count + 2


def test_test_cases_satisfy_preconditions():
    @run_test()
    def _(test_case):
        n = test_case.choice(10)
        test_case.assume(n != 0)
        assert n != 0


def test_error_on_too_strict_precondition():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(test_case):
            test_case.choice(10)
            test_case.reject()


def test_error_on_unbounded_test_function(monkeypatch):
    monkeypatch.setattr(core, "BUFFER_SIZE", 10)
    with pytest.raises(Unsatisfiable):

        @run_test(max_examples=5)
        def _(test_case):
            while True:
                test_case.choice(10)


@pytest.mark.requires("caching")
def test_function_cache():
    def tf(tc):
        if tc.choice(1000) >= 200:
            tc.mark_status(Status.INTERESTING)
        if tc.choice(1) == 0:
            tc.reject()

    state = State(Random(0), tf, 100)

    cache = CachedTestFunction(state.test_function)

    assert cache([1, 1]) == Status.VALID
    assert cache([1]) == Status.EARLY_STOP
    assert cache([1000]) == Status.INTERESTING
    assert cache([1000]) == Status.INTERESTING
    assert cache([1000, 1]) == Status.INTERESTING

    assert state.calls == 2


@pytest.mark.requires("caching")
def test_cache_key_distinguishes_negative_zero():
    """_cache_key must distinguish 0.0 from -0.0 even though they are
    equal in Python. Otherwise the cache conflates them and the shrinker
    can't replace -0.0 with 0.0."""
    assert _cache_key(0.0) != _cache_key(-0.0)


@pytest.mark.requires("caching")
def test_cache_key_distinguishes_nan_variants():
    """_cache_key must distinguish different NaN bit patterns, which
    Python considers equal (nan == nan is False, but for dict purposes
    the same object is used)."""
    nan1 = float("nan")
    # Create a NaN with a different bit pattern.
    bits = struct.unpack("!Q", struct.pack("!d", nan1))[0] ^ 1
    nan2 = struct.unpack("!d", struct.pack("!Q", bits))[0]
    assert math.isnan(nan1) and math.isnan(nan2)
    assert _cache_key(nan1) != _cache_key(nan2)


@pytest.mark.requires("caching")
@pytest.mark.requires("floats")
def test_cache_distinguishes_negative_zero_in_lookup():
    """The cache must store separate entries for 0.0 and -0.0 so that
    looking up a sequence containing 0.0 doesn't return the result
    for a sequence that used -0.0 (or vice versa)."""

    def tf(tc):
        v = tc.any(gs.floats(allow_nan=False, allow_infinity=False))
        if v >= 0.0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    cache = CachedTestFunction(state.test_function)

    # Record both variants.
    assert cache([0.0]) == Status.INTERESTING
    assert cache([-0.0]) == Status.INTERESTING

    # Lookup must return the correct nodes for each.
    result_pos = cache.lookup([0.0])
    result_neg = cache.lookup([-0.0])
    assert result_pos is not None
    assert result_neg is not None
    assert math.copysign(1.0, result_pos[1][0].value) == 1.0
    assert math.copysign(1.0, result_neg[1][0].value) == -1.0


def test_prints_a_top_level_weighted(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            assert test_case.weighted(0.5)

    captured = capsys.readouterr()
    assert captured.out.strip() == "weighted(0.5): False"


def test_errors_when_using_frozen():
    tc = TC.for_choices([0])
    tc.status = Status.VALID

    with pytest.raises(Frozen):
        tc.mark_status(Status.INTERESTING)

    with pytest.raises(Frozen):
        tc.choice(10)

    with pytest.raises(Frozen):
        tc.forced_choice(10)


def test_errors_on_too_large_choice():
    tc = TC.for_choices([0])
    with pytest.raises(ValueError):
        tc.choice(2**64)


def test_can_choose_full_64_bits():
    @run_test()
    def _(tc):
        tc.choice(2**64 - 1)


def test_integer_choice_simplest():
    assert IntegerChoice(-10, 10).simplest == 0
    assert IntegerChoice(5, 100).simplest == 5
    assert IntegerChoice(-100, -5).simplest == -5


def test_integer_choice_unit():
    assert IntegerChoice(-10, 10).unit == 1
    assert IntegerChoice(5, 100).unit == 6
    # When simplest is at the top of the range, unit is simplest - 1.
    assert IntegerChoice(-100, -5).unit == -6
    # Single-value range: unit falls back to simplest.
    assert IntegerChoice(5, 5).unit == 5


@pytest.mark.requires("floats")
def test_float_choice_unit():
    # Under (exponent_rank, mantissa, sign) ordering:
    # index 0 = 0.0, index 1 = -0.0, index 2 = 1.0, ...
    assert FloatChoice(-10.0, 10.0, False, False).unit == -0.0
    # Negative-only range: simplest has smallest exp_rank in range.
    fc = FloatChoice(-10.0, -5.0, False, False)
    assert fc.simplest == -5.0  # exp_rank=3, simpler than -8.0 (exp_rank=5)
    # Single-value range: falls back to simplest.
    assert FloatChoice(5.0, 5.0, False, False).unit == 5.0


@pytest.mark.requires("bytes")
def test_bytes_choice_unit():
    # Second-simplest in sort_key order: next byte value, not next length.
    assert BytesChoice(0, 10).unit == b"\x01"
    assert BytesChoice(3, 10).unit == b"\x00\x00\x01"


def test_value_punning_on_type_change():
    """When replaying a choice sequence and the type at a position
    changes (e.g., from FloatChoice to BooleanChoice), the value
    is punned: simplest→simplest, anything else→unit."""

    def tf(tc):
        # Branch 0 = booleans, branch 1 = integers
        branch = tc.draw_integer(0, 1)
        if branch == 0:
            tc.weighted(0.5)
        else:
            tc.draw_integer(0, 100)
        tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    # The result should have branch=0 (simplest).
    # If the initial result had branch=1 with integer value 50,
    # switching to branch=0 puns 50 → BooleanChoice.unit (True).
    assert state.result is not None
    assert state.result[0].value == 0  # branch = simplest


def test_forced_choice_bounds():
    with pytest.raises(ValueError):

        @run_test(database={})
        def _(tc):
            tc.forced_choice(2**64)


@pytest.mark.requires("database")
def test_malformed_database_entry():
    """Malformed database entries are silently ignored."""
    db = {"_": b"\xff\xff\xff"}

    @run_test(database=db, max_examples=1)
    def _(test_case):
        pass


@pytest.mark.requires("database")
def test_empty_database_entry():
    """Empty database entries produce an empty replay."""
    db = {"_": b""}

    @run_test(database=db, max_examples=1)
    def _(test_case):
        pass


@pytest.mark.requires("database")
@pytest.mark.parametrize(
    "data",
    [
        b"\x01",  # Boolean tag with no value byte
        b"\x00\x01\x02",  # Integer tag with only 3 of 8 bytes
        b"\x02\x00\x00",  # Bytes tag with truncated length header
        b"\x02\x00\x00\x00\x05\x01",  # Bytes tag claiming length 5 but only 1 byte
    ],
)
def test_truncated_database_entry(data):
    """Truncated database entries are silently ignored."""
    db = {"_": data}

    @run_test(database=db, max_examples=1)
    def _(test_case):
        pass


@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_with_type_change():
    """Lower-and-bump pass handles the case where decrementing an
    integer choice changes the type of the next choice (e.g. one_of
    where branches draw different types)."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            # Branch 0 draws a boolean, branch 1 draws an integer.
            # Lower-and-bump will decrement the branch index and
            # find a BooleanChoice at the next position.
            value = tc.any(gs.one_of(gs.booleans(), gs.integers(0, 100)))
            assert isinstance(value, bool) or value <= 50


def test_shrinking_mixed_choice_types_no_sort_crash():
    """Sorting pass should not crash when the result has a mix of
    IntegerChoice and BooleanChoice nodes and shrinking changes
    which type is at a given position.

    Regression test for a TypeError found by minismith."""

    def tf(tc):
        # Mix integer and boolean choices — shrinking may change
        # which type appears at each position.
        x = tc.choice(3)
        if x > 0:
            tc.weighted(0.5)
            tc.weighted(0.5)
        tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None


def test_shrinking_stale_indices_no_redistribute_crash():
    """Redistribute and lower-and-bump passes should not crash when
    shrinking changes the result length, making pre-computed
    indices stale.

    Regression test for an IndexError found by minismith."""

    def tf(tc):
        # Variable-length sequence: the number of integers drawn
        # depends on a prior choice, so shrinking that choice
        # changes the result length mid-pass.
        n = tc.draw_integer(2, 8)
        vals = [tc.draw_integer(0, 100) for _ in range(n)]
        tc.weighted(0.5)  # Mix in a non-integer choice
        if sum(vals) > 150 and len(vals) >= 3:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 500)
    state.run()
    assert state.result is not None


@pytest.mark.requires("collections")
def test_sorting_pass_survives_type_changes_from_lists():
    """Sorting insertion-sort must not crash when a successful swap
    changes the result so that choice types at pre-computed indices
    shift. Regression for AssertionError in sorting.py found by minismith."""

    with pytest.raises(AssertionError):

        @run_test(max_examples=1, database={}, quiet=True, random=Random(0))
        def _(tc):
            v0 = tc.any(gs.lists(gs.booleans(), max_size=10))
            v1 = tc.any(gs.lists(gs.integers(0, 0), max_size=10))
            assert len(v0) == len(v1)


@pytest.mark.requires("collections")
def test_sorting_full_sort_survives_stale_indices():
    """Sorting full-sort path must not crash when a prior group's
    sort shortens the result, making indices for the next group
    invalid. Regression for IndexError in sorting.py found by minismith."""

    try:

        @run_test(max_examples=1, database={}, quiet=True, random=Random(1))
        def _(tc):
            v0 = tc.any(gs.lists(gs.integers(0, 12), max_size=10))
            tc.any(gs.booleans())
            if not (len(v0) == 0 or v0[0] > 0):
                raise AssertionError
            if len(v0) > 2:
                if not (len(v0) == 0):
                    raise AssertionError
    except AssertionError:
        pass


@pytest.mark.requires("collections")
def test_sorting_stale_filter_with_punning():
    """Sorting stale-index filter must handle the case where punning
    changes node types so that a group has fewer than 2 members.
    Regression for AssertionError in sorting.py found by shrink comparison."""

    @gs.composite
    def pair(tc):
        a = tc.any(gs.booleans())
        b = tc.any(gs.booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.any(gs.lists(gs.integers(0, 0), max_size=10))
        v1 = tc.any(
            gs.integers(0, 0).flat_map(lambda x: gs.lists(gs.booleans(), max_size=1))
        )
        tc.any(pair())
        if len(v0) != len(v1):
            tc.mark_status(Status.INTERESTING)

    for seed in range(5):
        state = State(Random(seed), tf, 200)
        state.run()


@pytest.mark.requires("collections")
def test_unique_list_shrinks_using_negative_values():
    """Unique signed integer lists should shrink to use negative values
    when that gives smaller absolute values (e.g. [0,1,-1,2,-2] not [0,1,2,3,4]).
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.lists(gs.integers(-10, 10), max_size=5, unique=True))
        if len(v0) >= 5:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    # Extract the integer values from the list choices (skip booleans)
    int_values = [n.value for n in state.result if isinstance(n.kind, IntegerChoice)]
    assert int_values == [0, 1, -1, 2, -2]


@pytest.mark.requires("bytes")
@pytest.mark.requires("collections")
@pytest.mark.requires("text")
@pytest.mark.requires("shrinking.index_passes")
def test_bytes_increment_shortens_sequence():
    """Growing a bytes value by one byte can eliminate subsequent choices,
    producing a shorter (and thus simpler) overall sequence.
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.binary(max_size=20))
        v1 = tc.any(
            gs.dictionaries(
                gs.integers(0, 0),
                gs.text(min_codepoint=32, max_codepoint=126, max_size=20),
                max_size=5,
            )
        )
        if len(v0) + len(v1) >= 20:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    # Should shrink to just a 20-byte binary + empty dict (2 choices),
    # not 19 bytes + dict entry (5 choices).
    assert len(state.result) == 2


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_explores_new_range():
    """When decrementing an integer changes the range of a non-adjacent
    later integer, lower_and_bump should explore the new range via
    absolute power-of-2 values at various gaps.
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.sampled_from([32, 46]))
        v1 = tc.any(gs.sampled_from([32, 46]))
        v2 = tc.any(gs.integers(-abs(v0) - 1, abs(v0) + 1))
        v3 = tc.any(gs.integers(-abs(v2) - 1, abs(v2) + 1))
        if v2 == v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    values = [n.value for n in state.result]
    assert values == [0, 0, 32, 0]


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_tries_negative_values():
    """lower_and_bump should try negative absolute powers of 2 when
    exploring a new range, not just positive ones.
    Regression for shrink quality found by minismith."""

    @gs.composite
    def pair(tc):
        a = tc.any(gs.booleans())
        b = tc.any(gs.booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.any(pair())
        v1 = tc.any(pair())
        v2 = tc.any(gs.one_of(gs.integers(0, 0), gs.booleans()))
        if len(v0) <= 0:
            tc.mark_status(Status.INTERESTING)
        v3 = tc.any(gs.integers(-1, 1))
        if v2:
            tc.mark_status(Status.INTERESTING)
        if not v2 and v3 < 0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    values = [n.value for n in state.result]
    # one_of index 0 (gs.integers(0,0)) with v3=-1 is simpler than
    # one_of index 1 (booleans=True) with v3=0
    assert values == [False, False, False, False, 0, 0, -1]


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_increment_to_max_shortens_via_sampled_from():
    """try_shortening_via_increment should try max_value, not just +1.
    For gs.sampled_from([1, 1, 0]), index 2 maps to 0 which triggers an
    early exit (1 choice instead of 2).
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.sampled_from([1, 1, 0]))
        if v0 <= 0:
            tc.mark_status(Status.INTERESTING)
        v1 = tc.any(gs.booleans())
        if v1:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    assert len(state.result) == 1


@pytest.mark.requires("collections")
def test_redistribute_stale_indices_after_type_change():
    """redistribute_integers must handle stale indices when previous passes
    change the result structure, causing a node that was IntegerChoice to
    become BooleanChoice. Regression for AssertionError found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.booleans())
        v1 = tc.any(gs.booleans().map(lambda x: int(x)))
        v2 = tc.any(gs.integers(1, 7).filter(lambda x: x % 2 == 0))
        v3 = tc.any(gs.booleans())
        v4 = tc.any(gs.one_of(gs.integers(0, 0), gs.booleans()))
        if v0:
            tc.mark_status(Status.INTERESTING)

    # Should not crash.
    state = State(Random(0), tf, 1000)
    state.run()


@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_targets_booleans():
    """lower_and_bump should try bumping boolean targets, not just
    integer ones. Decrementing an integer while bumping a boolean
    from False to True can produce a simpler overall sort_key.
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.integers(0, 1))
        v1 = tc.any(gs.booleans())
        if v0 >= 1:
            tc.mark_status(Status.INTERESTING)
        if v1:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    values = [n.value for n in state.result]
    # v0=0 + v1=True is simpler than v0=1 + v1=False
    # (sort_key position 0: (0,F) < (1,F))
    assert values[0] == 0


@pytest.mark.requires("floats")
@pytest.mark.requires("shrinking.index_passes")
def test_float_increment_shortens_via_negative():
    """Making a float negative can trigger an earlier check and shorten
    the overall choice sequence. try_shortening_via_increment should
    try negative float values.
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.booleans())
        v1 = tc.any(gs.floats(allow_nan=False, allow_infinity=False))
        v2 = tc.any(gs.booleans())
        if v1 < 0.0:
            tc.mark_status(Status.INTERESTING)
        tc.any(gs.booleans())
        if v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    assert len(state.result) == 3


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_increment_with_dependent_continuation():
    """try_shortening_via_increment must pass prefix_nodes so that
    value punning maps simplest→simplest when the continuation
    changes type (e.g. list boolean → integer).
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.integers(0, 0))
        v1 = tc.any(gs.booleans())
        v2 = tc.any(gs.integers(0, 0))
        v3 = tc.any(gs.lists(gs.integers(-21, -1), max_size=10, unique=True))
        if len(v3) != 0:
            tc.mark_status(Status.INTERESTING)
        if v1:
            v4 = tc.any(gs.integers(v0, v0 + 1))
            if v0 + v4 <= 0:
                tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    # Should shrink to 5 choices (via v1=True path) not 6 (via non-empty list)
    assert len(state.result) == 5


@pytest.mark.requires("bytes")
@pytest.mark.requires("shrinking.advanced_bytes_passes")
def test_redistribute_bytes_between_pairs():
    """When two bytes values share a total length constraint, the shrinker
    should redistribute to make the first empty and the second full.
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.binary(max_size=20))
        v1 = tc.any(gs.binary(max_size=20))
        if len(v0) + len(v1) >= 20:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    bytes_values = [n.value for n in state.result if isinstance(n.kind, BytesChoice)]
    # First bytes should be empty, second should carry all the length.
    assert bytes_values[0] == b""


@pytest.mark.requires("bytes")
def test_redistribute_bytes_respects_max_size():
    """redistribute_bytes must skip transfers that exceed max_size."""

    def tf(tc):
        v0 = tc.any(gs.binary(min_size=5, max_size=10))
        v1 = tc.any(gs.binary(max_size=8))
        if len(v0) + len(v1) >= 15:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None


@pytest.mark.requires("text")
def test_string_sorts_characters_when_possible():
    """String shrinking should sort characters by codepoint.
    Sorting '0e0' produces '00e' (smaller codepoints first)."""

    def tf(tc):
        v0 = tc.any(gs.text(min_codepoint=32, max_codepoint=126, max_size=20))
        if len(v0) >= 3 and "e" in v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    assert state.result[0].value == "00e"


@pytest.mark.requires("bytes")
def test_bytes_sorts_when_order_matters():
    """Bytes shrinking should sort bytes when the test depends on order."""

    def tf(tc):
        v0 = tc.any(gs.binary(min_size=3, max_size=3))
        # Only interesting if the bytes are NOT already sorted but contain 0x01.
        if b"\x01" in v0 and v0 != bytes(sorted(v0)):
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    # Sorting would make v0 sorted, which violates the condition.
    # So the swap should fail, covering the failure branch.
    assert state.result is not None


@pytest.mark.requires("bytes")
@pytest.mark.requires("collections")
@pytest.mark.requires("text")
@pytest.mark.requires("floats")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_with_float_target():
    """lower_and_bump should try float values (1.0, -1.0, etc.) when
    the target is FloatChoice. Making a string shorter while making a
    float non-zero can produce a simpler overall result."""

    def tf(tc):
        v0 = tc.any(gs.text(min_codepoint=32, max_codepoint=126, max_size=20))
        v1 = tc.any(gs.floats(allow_nan=False, allow_infinity=False))
        if len(v0) >= 4:
            tc.mark_status(Status.INTERESTING)
        if v1 != 0.0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    # Prefer empty string with non-zero float (simpler at position 0)
    assert state.result[0].value == ""


@pytest.mark.requires("floats")
@pytest.mark.requires("collections")
def test_redistribute_stale_indices_with_one_of():
    """redistribute_integers must handle stale indices when one_of
    changes the result structure during shrinking.
    Regression for AssertionError in redistribute found by minismith."""

    def tf(tc):
        v0 = tc.any(
            gs.one_of(
                gs.booleans(),
                gs.integers(0, 0),
                gs.integers(2, 2).filter(lambda x: x > 0),
            )
        )
        v1 = tc.any(gs.integers(0, 0))
        if v0:
            tc.mark_status(Status.INTERESTING)

    # Should not crash.
    state = State(Random(0), tf, 1000)
    state.run()


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_stale_j_after_replace():
    """lower_and_bump must handle j going out of bounds when a replace
    shortens the result during the bytes/string bump loop.
    Regression for AssertionError in lower_and_bump found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.booleans())
        tc.any(gs.booleans())
        tc.any(gs.booleans())
        tc.any(gs.lists(gs.integers(0, 0), max_size=10).filter(lambda x: len(x) > 0))
        tc.any(
            gs.integers(-54, -32).flat_map(
                lambda n: gs.lists(
                    gs.integers(0, 100),
                    min_size=abs(n) % 5,
                    max_size=abs(n) % 5 + 1,
                )
            )
        )
        if v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()


@pytest.mark.requires("shrinking.mutation")
def test_mutation_with_single_value_adjacent():
    """mutate_and_shrink handles adjacent single-value choices where
    from_index(1) returns None."""

    def tf(tc):
        v0 = tc.any(gs.booleans())
        tc.any(gs.integers(5, 5))  # single-value, from_index(1) = None
        if v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None


@pytest.mark.requires("floats")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_with_float_source_gaps():
    """lower_and_bump must handle from_index returning None when the
    float source has index gaps (bounded range with interleaved signs)."""

    def tf(tc):
        v0 = tc.any(gs.floats(min_value=1.0, max_value=2.0, allow_nan=False))
        v1 = tc.any(gs.booleans())
        if v0 > 1.5 and v1:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None


@pytest.mark.requires("floats")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_with_bounded_float_target():
    """lower_and_bump must skip invalid float bump values when the
    float's range doesn't include 1.0 or -1.0."""

    def tf(tc):
        v0 = tc.any(gs.integers(0, 5))
        v1 = tc.any(gs.floats(min_value=0.0, max_value=0.5, allow_nan=False))
        if v0 >= 3 and v1 > 0.0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None


@pytest.mark.requires("bytes")
@pytest.mark.requires("collections")
@pytest.mark.requires("text")
def test_sort_insertion_stale_indices():
    """Sorting insertion sort must handle stale indices when a swap
    changes the result structure (e.g. shortening via value punning).
    Regression for IndexError in sorting.py found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.lists(gs.integers(0, 20), max_size=10, unique=True))
        v1 = tc.any(
            gs.dictionaries(
                gs.text(min_codepoint=32, max_codepoint=126, max_size=5),
                gs.booleans(),
                max_size=5,
            )
        )
        v2 = tc.any(gs.lists(gs.booleans(), max_size=10))
        v3 = tc.any(gs.binary(max_size=20))
        v4 = tc.any(gs.booleans())
        if len(v0) != 0:
            tc.mark_status(Status.INTERESTING)
        if len(v2) != len(v3):
            tc.mark_status(Status.INTERESTING)

    # Should not crash. Try multiple seeds to exercise sorting edge cases.
    for seed in range(5):
        state = State(Random(seed), tf, 1000)
        state.run()


@pytest.mark.requires("text")
@pytest.mark.requires("shrinking.advanced_string_passes")
def test_string_length_redistribution():
    """When two strings share a total length constraint (len(v0)+len(v1) >= N),
    the shrinker should redistribute length to make the first string as short
    as possible, even though shortening v0 requires lengthening v1.
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(gs.text(min_codepoint=32, max_codepoint=126, max_size=20))
        v1 = tc.any(gs.text(min_codepoint=32, max_codepoint=126, max_size=20))
        if len(v0) + len(v1) >= 30:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    v0_len = len(state.result[0].value)
    # Optimal: v0 as short as possible (10 chars, since v1 max is 20).
    assert v0_len == 10


@pytest.mark.requires("bytes")
@pytest.mark.requires("shrinking.advanced_bytes_passes")
def test_bytes_length_redistribution():
    """When two bytes values share a total length constraint, the shrinker
    should redistribute to make the first as short as possible.
    Parallel test to test_string_length_redistribution — bytes and strings
    share the same shrinking infrastructure and often have the same bugs."""

    def tf(tc):
        v0 = tc.any(gs.binary(max_size=20))
        v1 = tc.any(gs.binary(max_size=20))
        if len(v0) + len(v1) >= 30:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    v0_len = len(state.result[0].value)
    # Optimal: v0 as short as possible (10 bytes, since v1 max is 20).
    assert v0_len == 10


@pytest.mark.requires("bytes")
def test_bytes_redistribution_moves_all():
    """When the second bytes value can absorb everything from the first,
    redistribution should move as much as possible. The min_size on v0
    prevents the value shrinker from emptying it directly, forcing
    redistribution to do the work."""

    def tf(tc):
        v0 = tc.any(gs.binary(min_size=3, max_size=10))
        v1 = tc.any(gs.binary(max_size=20))
        if len(v0) + len(v1) >= 10:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    # v0 can't go below min_size=3, so optimal is v0=3 bytes.
    assert len(state.result[0].value) == 3


@pytest.mark.requires("floats")
@pytest.mark.requires("collections")
def test_negative_zero_shrinks_to_positive_zero():
    """The shrinker should prefer 0.0 over -0.0 since sort_key(0.0) <
    sort_key(-0.0). The cache must distinguish them despite 0.0 == -0.0
    in Python.
    Regression for shrink quality found by minismith."""

    @gs.composite
    def pair(tc):
        a = tc.any(gs.booleans())
        b = tc.any(gs.booleans())
        return (a, b)

    def tf(tc):
        tc.any(pair())
        tc.any(pair())
        v2 = tc.any(
            gs.one_of(
                gs.floats(allow_nan=False, allow_infinity=False),
                gs.floats(allow_nan=False, allow_infinity=False),
                gs.nothing(),
            )
        )
        v3 = tc.any(gs.booleans())
        v4 = tc.any(gs.booleans())
        if not (((v4) or (v2 > 0.0)) and (v2 >= 0.0)):
            tc.mark_status(Status.INTERESTING)

    state = State(Random(120), tf, 100)
    state.run()
    assert state.result is not None
    float_val = state.result[5].value
    assert isinstance(float_val, float)
    assert math.copysign(1.0, float_val) == 1.0, f"Expected 0.0 but got {float_val!r}"


@pytest.mark.requires("bytes")
@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_stale_kind_after_replace():
    """lower_and_bump must validate values against the CURRENT kind at
    position j, not the kind from before the replace. A replace can
    change types via value punning (e.g. BytesChoice → BooleanChoice).
    Regression for TypeError in sort_key found by minismith."""

    @gs.composite
    def pair(tc):
        a = tc.any(gs.booleans())
        b = tc.any(gs.booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.any(gs.lists(gs.booleans(), max_size=10))
        tc.any(gs.booleans())
        tc.any(gs.binary(max_size=20))
        tc.any(pair())
        tc.any(pair())
        if len(v0) != 0:
            tc.mark_status(Status.INTERESTING)

    # Should not crash.
    state = State(Random(0), tf, 100)
    state.run()


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.mutation")
def test_one_of_switches_to_shorter_branch():
    """When one_of branch 0 (lists) produces a truthy value in 4 choices
    but branch 1 (booleans via nested one_of) can do it in 3, the
    shrinker should find the shorter branch.

    The difficulty: switching the outer one_of index from 0 to 1 requires
    setting the inner index AND the value to non-zero simultaneously —
    a 3-position compound change.

    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(
            gs.one_of(
                gs.lists(gs.integers(0, 0), max_size=10),
                gs.one_of(gs.integers(0, 0), gs.booleans()),
            )
        )
        if v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(1), tf, 100)
    state.run()
    assert state.result is not None
    # Optimal: branch 1 → inner branch 1 (booleans) → True = 3 choices.
    # Suboptimal: branch 0 → list [0] = 4 choices.
    assert len(state.result) == 3


def test_bind_deletion_valid_but_not_shorter():
    """bind_deletion must correctly detect when a replacement produces
    a VALID test case that isn't shorter (no excess choices to delete).
    This requires the cache to populate test_case.nodes on hit."""

    def tf(tc):
        n = tc.draw_integer(0, 10)
        vals = [tc.draw_integer(0, 100) for _ in range(n)]
        if sum(vals) > 200:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 2000)
    state.run()
    assert state.result is not None


@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_j_past_end_after_shortening():
    """lower_and_bump must handle j becoming invalid when a decrement+zero
    attempt shortens the result.

    The test draws a count n then n values. For gap=1, idx=0: i=0 (count),
    j=1. Decrementing the count and zeroing everything after produces a
    shorter result (fewer draws). With n going from 3→2 and rest zeroed,
    n becomes 0 via value punning and the result shortens to 1 node,
    making j=1 invalid."""

    def tf(tc):
        n = tc.draw_integer(0, 5)
        for _ in range(n):
            tc.draw_integer(0, 10)
        tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 500)
    # Seed: n=3, 3 extra draws → 4 nodes. gap=1 shrinks n progressively,
    # and at gap=2 the j target falls past the shortened result.
    tc = TC.for_choices([3, 5, 5, 5])
    state.test_function(tc)
    assert state.result is not None
    assert len(state.result) == 4
    lower_and_bump(state)
    assert state.result is not None
    # Shrinks to n=0 (1 node) since lower_and_bump now tries
    # decrementing to simplest, not just index-1.
    assert len(state.result) == 1
