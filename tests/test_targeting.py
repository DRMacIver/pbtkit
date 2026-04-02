# This file is part of Pbtkit, which may be found at
# https://github.com/DRMacIver/pbtkit
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from random import Random

import pytest

from pbtkit import run_test

pytestmark = pytest.mark.requires("targeting")


@pytest.mark.parametrize("max_examples", range(1, 100))
def test_max_examples_is_not_exceeded(max_examples):
    """Targeting has a number of places it checks for
    whether we've exceeded the generation limits. This
    makes sure we've checked them all.
    """
    calls = 0

    @run_test(database={}, random=Random(0), max_examples=max_examples)
    def _(tc):
        nonlocal calls
        m = 10000
        n = tc.choice(m)
        calls += 1
        tc.target(n * (m - n))

    assert calls == max_examples


@pytest.mark.parametrize("seed", range(100))
def test_finds_a_local_maximum(seed):
    """Targeting has a number of places it checks for
    whether we've exceeded the generation limits. This
    makes sure we've checked them all.
    """

    with pytest.raises(AssertionError):

        @run_test(database={}, random=Random(seed), max_examples=200, quiet=True)
        def _(tc):
            m = tc.choice(1000)
            n = tc.choice(1000)
            score = -((m - 500) ** 2 + (n - 500) ** 2)
            tc.target(score)
            assert m != 500 or n != 500


def test_can_target_a_score_upwards_to_interesting(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            n = test_case.choice(1000)
            m = test_case.choice(1000)
            score = n + m
            test_case.target(score)
            assert score < 2000

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 1000",
        "choice(1000): 1000",
    ]


def test_can_target_a_score_upwards_without_failing():
    max_score = 0

    @run_test(database={}, max_examples=1000)
    def _(test_case):
        nonlocal max_score
        n = test_case.choice(1000)
        m = test_case.choice(1000)
        score = n + m
        test_case.target(score)
        max_score = max(score, max_score)

    assert max_score == 2000


def test_targeting_when_most_do_not_benefit(capsys):
    with pytest.raises(AssertionError):
        big = 10000

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            test_case.choice(1000)
            test_case.choice(1000)
            score = test_case.choice(big)
            test_case.target(score)
            assert score < big

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 0",
        "choice(1000): 0",
        f"choice({big}): {big}",
    ]


def test_targeting_adjust_avoids_negative_values():
    """Targeting adjust must handle choices near zero — attempting
    step=-1 on value 0 should not produce a negative choice."""

    @run_test(database={}, max_examples=200, random=Random(0))
    def _(test_case):
        # First choice will often be 0 (boundary boost). Targeting
        # tries step=-1, which would make it negative — must be skipped.
        n = test_case.choice(10)
        test_case.target(n)


def test_can_target_a_score_downwards(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(test_case):
            n = test_case.choice(1000)
            m = test_case.choice(1000)
            score = n + m
            test_case.target(-score)
            assert score > 0

    captured = capsys.readouterr()

    assert [c.strip() for c in captured.out.splitlines()] == [
        "choice(1000): 0",
        "choice(1000): 0",
    ]
