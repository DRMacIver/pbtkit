# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from minithesis import run_test

pytestmark = pytest.mark.requires("bytes")
from minithesis.core import DirectoryDB
from minithesis.generators import binary


def test_finds_short_binary(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(test_case):
            b = test_case.any(binary(max_size=10))
            assert len(b) < 1

    captured = capsys.readouterr()
    assert captured.out.strip() == r"any(binary(min_size=0, max_size=10)): b'\x00'"


def test_shrinks_bytes_to_minimal(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            b = test_case.any(binary(min_size=1, max_size=5))
            assert 0xFF not in b

    captured = capsys.readouterr()
    assert captured.out.strip() == r"any(binary(min_size=1, max_size=5)): b'\xff'"


def test_binary_respects_size_bounds():
    @run_test(database={})
    def _(test_case):
        b = test_case.any(binary(min_size=2, max_size=4))
        assert 2 <= len(b) <= 4


def test_shrinks_bytes_with_constraints(capsys):
    """When the simplest bytes value (all zeros at min_size) doesn't
    trigger the failure, the shrinker falls back to shortening and
    shrinking individual byte values."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            b = test_case.any(binary(min_size=2, max_size=10))
            assert sum(b) <= 10

    captured = capsys.readouterr()
    # Should find 2 bytes summing to 11.
    output = captured.out.strip()
    assert "binary(" in output
    assert r"\x0b" in output


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
                b = test_case.any(binary(max_size=10))
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
            b = test_case.any(binary(max_size=10))
            assert sum(b) > 0

    captured = capsys.readouterr()
    assert captured.out.strip() == "any(binary(min_size=0, max_size=10)): b''"


@pytest.mark.requires("targeting")
def test_targeting_with_bytes():
    """Targeting skips non-integer nodes without crashing."""
    max_score = 0

    @run_test(database={}, max_examples=200)
    def _(test_case):
        nonlocal max_score
        test_case.any(binary(max_size=5))
        n = test_case.choice(100)
        test_case.target(n)
        max_score = max(n, max_score)

    assert max_score == 100
