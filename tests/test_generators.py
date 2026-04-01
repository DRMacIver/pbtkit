# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import pytest

import minithesis.generators as gs
from minithesis import Unsatisfiable, run_test
from minithesis.collections import many

pytestmark = pytest.mark.requires("collections")


class Failure(Exception):
    pass


def test_mapped_possibility():
    @run_test()
    def _(tc):
        n = tc.any(gs.integers(0, 5).map(lambda n: n * 2))
        assert n % 2 == 0


def test_selected_possibility():
    @run_test()
    def _(tc):
        n = tc.any(gs.integers(0, 5).filter(lambda n: n % 2 == 0))
        assert n % 2 == 0


def test_bound_possibility():
    @run_test()
    def _(tc):
        m, n = tc.any(
            gs.integers(0, 5).flat_map(
                lambda m: gs.tuples(
                    gs.just(m),
                    gs.integers(m, m + 10),
                )
            )
        )

        assert m <= n <= m + 10


def test_cannot_witness_nothing():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(tc):
            tc.any(gs.nothing())


def test_cannot_witness_empty_one_of():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(tc):
            tc.any(gs.one_of())


def test_one_of_single():
    @run_test()
    def _(tc):
        n = tc.any(gs.one_of(gs.integers(0, 10)))
        assert 0 <= n <= 10


def test_can_draw_mixture():
    @run_test()
    def _(tc):
        m = tc.any(gs.one_of(gs.integers(-5, 0), gs.integers(2, 5)))
        assert -5 <= m <= 5
        assert m != 1


@pytest.mark.requires("targeting")
def test_target_and_reduce(capsys):
    """This test is very hard to trigger without targeting,
    and targeting will tend to overshoot the score, so we
    will see multiple interesting test cases before
    shrinking."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            m = tc.choice(100000)
            tc.target(m)
            assert m <= 99900

    captured = capsys.readouterr()

    assert captured.out.strip() == "choice(100000): 99901"


def test_impossible_weighted():
    with pytest.raises(Failure):

        @run_test(database={})
        def _(tc):
            tc.choice(1)
            for _ in range(10):
                if tc.weighted(0.0):
                    assert False
            if tc.choice(1):
                raise Failure()


def test_guaranteed_weighted():
    with pytest.raises(Failure):

        @run_test(database={})
        def _(tc):
            if tc.weighted(1.0):
                tc.choice(1)
                raise Failure()
            else:
                assert False


def test_size_bounds_on_list():
    @run_test(database={})
    def _(tc):
        ls = tc.any(gs.lists(gs.integers(0, 10), min_size=1, max_size=3))
        assert 1 <= len(ls) <= 3


def test_fixed_size_list():
    @run_test(database={})
    def _(tc):
        ls = tc.any(gs.lists(gs.integers(0, 10), min_size=3, max_size=3))
        assert len(ls) == 3


def test_many_with_small_max():
    """Exercise the many() geometric distribution with a small
    max_size, which triggers the p_continue refinement path."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        ls = tc.any(gs.lists(gs.integers(0, 10), max_size=2))
        assert len(ls) <= 2


def test_many_reject():
    """Test that many()'s reject() mechanism works: too many
    rejections force the collection to stop."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        result = []
        # Small range (0-2) makes duplicates frequent, so
        # force_stop will trigger after enough rejections.
        elems = many(tc, min_size=0, max_size=10)
        while elems.more():
            v = tc.draw_integer(0, 2)
            if v in result:
                elems.reject()
            else:
                result.append(v)
        assert len(result) == len(set(result))


def test_many_reject_unsatisfiable():
    """If too many rejections happen before reaching min_size,
    the test case is marked invalid."""
    with pytest.raises(Unsatisfiable):

        @run_test(database={}, max_examples=200)
        def _(tc):
            # min_size=5 but we reject everything, so we can
            # never reach min_size.
            elems = many(tc, min_size=5, max_size=10)
            while elems.more():
                elems.reject()


def test_sampled_from():
    @run_test(database={})
    def _(tc):
        v = tc.any(gs.sampled_from(["a", "b", "c"]))
        assert v in ("a", "b", "c")


def test_sampled_from_shrinks_to_first(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            v = tc.any(gs.sampled_from(["a", "b", "c"]))
            assert v != "a"

    captured = capsys.readouterr()
    assert "'a'" in captured.out


def test_sampled_from_single():
    @run_test(database={})
    def _(tc):
        assert tc.any(gs.sampled_from(["only"])) == "only"


def test_sampled_from_empty():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(tc):
            tc.any(gs.sampled_from([]))


def test_booleans():
    @run_test(database={})
    def _(tc):
        b = tc.any(gs.booleans())
        assert isinstance(b, bool)


def test_composite():
    @gs.composite
    def pairs(tc):
        x = tc.any(gs.integers(0, 10))
        y = tc.any(gs.integers(x, 10))
        return (x, y)

    @run_test(database={})
    def _(tc):
        x, y = tc.any(pairs())
        assert x <= y <= 10


def test_composite_with_args():
    @gs.composite
    def bounded_int(tc, max_val):
        return tc.any(gs.integers(0, max_val))

    @run_test(database={})
    def _(tc):
        n = tc.any(bounded_int(5))
        assert 0 <= n <= 5


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_composite_shrinks(capsys):
    @gs.composite
    def pairs(tc):
        x = tc.any(gs.integers(0, 100))
        y = tc.any(gs.integers(0, 100))
        return (x, y)

    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x, y = tc.any(pairs())
            assert x + y < 100

    captured = capsys.readouterr()
    assert "100, 0" in captured.out or "0, 100" in captured.out


def test_unique_lists():
    @run_test(database={})
    def _(tc):
        ls = tc.any(gs.lists(gs.integers(0, 10), unique=True, max_size=5))
        assert len(ls) == len(set(ls))


def test_unique_lists_shrinks(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            ls = tc.any(gs.lists(gs.integers(0, 100), unique=True))
            assert len(ls) < 3

    captured = capsys.readouterr()
    assert "lists(" in captured.out


def test_unique_by():
    @run_test(database={})
    def _(tc):
        ls = tc.any(
            gs.lists(gs.integers(0, 100), unique_by=lambda x: x % 10, max_size=5)
        )
        keys = [x % 10 for x in ls]
        assert len(keys) == len(set(keys))


def test_dictionaries():
    @run_test(database={})
    def _(tc):
        d = tc.any(gs.dictionaries(gs.integers(0, 10), gs.integers(0, 100), max_size=5))
        assert isinstance(d, dict)
        assert len(d) <= 5
        for k, v in d.items():
            assert 0 <= k <= 10
            assert 0 <= v <= 100


def test_dictionaries_shrinks(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            d = tc.any(gs.dictionaries(gs.integers(0, 10), gs.integers(0, 100)))
            assert sum(d.values()) <= 100

    captured = capsys.readouterr()
    assert "dictionaries(" in captured.out


def test_dictionaries_size_bounds():
    @run_test(database={})
    def _(tc):
        d = tc.any(
            gs.dictionaries(
                gs.integers(0, 10), gs.integers(0, 100), min_size=1, max_size=3
            )
        )
        assert 1 <= len(d) <= 3
