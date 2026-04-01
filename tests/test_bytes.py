# This file is part of Pbtkit, which may be found at
# https://github.com/DRMacIver/pbtkit
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from pbtkit import run_test
from pbtkit.bytes import BytesChoice

pytestmark = pytest.mark.requires("bytes")
import pbtkit.generators as gs
from pbtkit.database import DirectoryDB


def test_finds_short_binary(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(test_case):
            b = test_case.draw(gs.binary(max_size=10))
            assert len(b) < 1

    captured = capsys.readouterr()
    assert r"b'\x00'" in captured.out


def test_shrinks_bytes_to_minimal(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            b = test_case.draw(gs.binary(min_size=1, max_size=5))
            assert 0xFF not in b

    captured = capsys.readouterr()
    assert r"b'\xff'" in captured.out


def test_binary_respects_size_bounds():
    @run_test(database={})
    def _(test_case):
        b = test_case.draw(gs.binary(min_size=2, max_size=4))
        assert 2 <= len(b) <= 4


def test_shrinks_bytes_with_constraints(capsys):
    """When the simplest bytes value (all zeros at min_size) doesn't
    trigger the failure, the shrinker falls back to shortening and
    shrinking individual byte values."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            b = test_case.draw(gs.binary(min_size=2, max_size=10))
            assert sum(b) <= 10

    captured = capsys.readouterr()
    # Should find 2 bytes summing to 11. The exact byte distribution
    # varies because the shrinker can't redistribute value between bytes.
    output = captured.out.strip()
    assert " = " in output
    value = eval(output.split(" = ", 1)[1])
    assert len(value) == 2
    assert sum(value) == 11


@pytest.mark.requires("database")
def test_mixed_types_database_round_trip(tmpdir):
    """Database round-trip works for all choice types (integer,
    boolean, and bytes)."""
    db = DirectoryDB(tmpdir)
    count = 0

    def run():
        with pytest.raises(AssertionError):

            @run_test(database=db)
            def _(test_case):
                nonlocal count
                count += 1
                b = test_case.draw(gs.binary(max_size=10))
                test_case.weighted(0.5)
                assert len(b) < 1

    run()
    prev_count = count

    run()
    assert count == prev_count + 2


def test_shrinks_bytes_to_simplest(capsys):
    """When the simplest bytes value itself triggers the failure,
    the shrinker finds it immediately."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(test_case):
            b = test_case.draw(gs.binary(max_size=10))
            assert sum(b) > 0

    captured = capsys.readouterr()
    assert "b''" in captured.out


@pytest.mark.requires("shrinking.index_passes")
def test_bytes_from_index_out_of_range():
    """from_index past max_index returns None."""
    bc = BytesChoice(0, 2)
    assert bc.from_index(bc.max_index + 1) is None


@pytest.mark.requires("shrinking.mutation")
def test_bytes_from_index_past_end():
    """from_index returns None for indices past all length buckets.

    BytesChoice(0, 2).max_index == 65792 (to_index(b"\\xff\\xff")),
    so from_index(65793) exhausts all length buckets and returns None."""
    bc = BytesChoice(0, 2)
    assert bc.from_index(65793) is None


@pytest.mark.requires("targeting")
def test_targeting_with_bytes():
    """Targeting skips non-integer nodes without crashing."""
    max_score = 0

    @run_test(database={}, max_examples=200)
    def _(test_case):
        nonlocal max_score
        test_case.draw(gs.binary(max_size=5))
        n = test_case.choice(100)
        test_case.target(n)
        max_score = max(n, max_score)

    assert max_score == 100


def test_bytes_choice_unit():
    # Second-simplest in sort_key order: next byte value, not next length.
    assert BytesChoice(0, 10).unit == b"\x01"
    assert BytesChoice(3, 10).unit == b"\x00\x00\x01"
