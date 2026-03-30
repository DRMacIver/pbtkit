# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from random import Random

import pytest

import minithesis.core as core
from minithesis import Generator, Unsatisfiable, run_test
from minithesis.bytes import BytesChoice
from minithesis.caching import CachedTestFunction
from minithesis.core import (
    Frozen,
    IntegerChoice,
    Status,
)
from minithesis.core import MinithesisState as State
from minithesis.core import TestCase as TC
from minithesis.database import DirectoryDB
from minithesis.floats import FloatChoice
from minithesis.generators import booleans, composite, integers, lists, one_of


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
    assert FloatChoice(-10.0, 10.0, False, False).unit == 1.0
    # When simplest is at the top, unit goes down.
    assert FloatChoice(-10.0, -5.0, False, False).unit == -6.0
    # Single-value range: falls back to simplest.
    assert FloatChoice(5.0, 5.0, False, False).unit == 5.0


@pytest.mark.requires("bytes")
def test_bytes_choice_unit():
    assert BytesChoice(0, 10).unit == b"\x00"
    assert BytesChoice(3, 10).unit == b"\x00\x00\x00\x00"


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
            value = tc.any(one_of(booleans(), integers(0, 100)))
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
            v0 = tc.any(lists(booleans(), max_size=10))
            v1 = tc.any(lists(integers(0, 0), max_size=10))
            assert len(v0) == len(v1)


@pytest.mark.requires("collections")
def test_sorting_full_sort_survives_stale_indices():
    """Sorting full-sort path must not crash when a prior group's
    sort shortens the result, making indices for the next group
    invalid. Regression for IndexError in sorting.py found by minismith."""

    try:

        @run_test(max_examples=1, database={}, quiet=True, random=Random(1))
        def _(tc):
            v0 = tc.any(lists(integers(0, 12), max_size=10))
            tc.any(booleans())
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

    @composite
    def pair(tc):
        a = tc.any(booleans())
        b = tc.any(booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.any(lists(integers(0, 0), max_size=10))
        v1 = tc.any(integers(0, 0).flat_map(lambda x: lists(booleans(), max_size=1)))
        tc.any(pair())
        if len(v0) != len(v1):
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()


@pytest.mark.requires("collections")
def test_unique_list_shrinks_using_negative_values():
    """Unique signed integer lists should shrink to use negative values
    when that gives smaller absolute values (e.g. [0,1,-1,2,-2] not [0,1,2,3,4]).
    Regression for shrink quality found by minismith."""

    def tf(tc):
        v0 = tc.any(lists(integers(-10, 10), max_size=5, unique=True))
        if len(v0) >= 5:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    # Extract the integer values from the list choices (skip booleans)
    int_values = [n.value for n in state.result if isinstance(n.kind, IntegerChoice)]
    assert int_values == [0, 1, -1, 2, -2]


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
