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
from minithesis.core import (
    CachedTestFunction,
    Frozen,
    Status,
)
from minithesis.core import MinithesisState as State
from minithesis.core import TestCase as TC
from minithesis.database import DirectoryDB
from minithesis.generators import integers


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


def test_function_cache():
    def tf(tc):
        if tc.choice(1000) >= 200:
            tc.mark_status(Status.INTERESTING)
        if tc.choice(1) == 0:
            tc.reject()

    state = State(Random(0), tf, 100)

    cache = CachedTestFunction(state.test_function)

    assert cache([1, 1]) == Status.VALID
    assert cache([1]) == Status.OVERRUN
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
