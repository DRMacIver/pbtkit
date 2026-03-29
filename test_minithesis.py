# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import defaultdict
from random import Random

import pytest

import minithesis as mt
from generators import (
    binary,
    booleans,
    composite,
    integers,
    just,
    lists,
    nothing,
    one_of,
    sampled_from,
    tuples,
)
from hypothesis import HealthCheck, given, note, reject, settings
from hypothesis import strategies as st
from minithesis import CachedTestFunction, DirectoryDB, Frozen, Generator, Status
from minithesis import TestCase as TC
from minithesis import TestingState as State
from minithesis import Unsatisfiable, run_test


@pytest.fixture(autouse=True)
def _isolate_database(tmp_path, monkeypatch):
    """Ensure each test gets a fresh default database directory
    so tests don't leak state via .minithesis-cache."""
    monkeypatch.setattr(mt, "_DEFAULT_DATABASE_PATH", str(tmp_path / "cache"))


@Generator
def list_of_integers(test_case):
    result = []
    while test_case.weighted(0.9):
        result.append(test_case.choice(10000))
    return result


@pytest.mark.parametrize("seed", range(10))
def test_finds_small_list(capsys, seed):

    with pytest.raises(AssertionError):

        @run_test(database={}, random=Random(seed))
        def _(test_case):
            ls = test_case.any(lists(integers(0, 10000)))
            assert sum(ls) <= 1000

    captured = capsys.readouterr()

    assert (
        captured.out.strip()
        == "any(lists(integers(min_value=0, max_value=10000), min_size=0, max_size=inf)): [1001]"
    )


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
    monkeypatch.setattr(mt, "BUFFER_SIZE", 10)
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


def test_mapped_possibility():
    @run_test()
    def _(tc):
        n = tc.any(integers(0, 5).map(lambda n: n * 2))
        assert n % 2 == 0


def test_selected_possibility():
    @run_test()
    def _(tc):
        n = tc.any(integers(0, 5).filter(lambda n: n % 2 == 0))
        assert n % 2 == 0


def test_bound_possibility():
    @run_test()
    def _(tc):
        m, n = tc.any(
            integers(0, 5).flat_map(
                lambda m: tuples(
                    just(m),
                    integers(m, m + 10),
                )
            )
        )

        assert m <= n <= m + 10


def test_cannot_witness_nothing():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(tc):
            tc.any(nothing())


def test_cannot_witness_empty_one_of():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(tc):
            tc.any(one_of())


def test_one_of_single():
    @run_test()
    def _(tc):
        n = tc.any(one_of(integers(0, 10)))
        assert 0 <= n <= 10


def test_can_draw_mixture():
    @run_test()
    def _(tc):
        m = tc.any(one_of(integers(-5, 0), integers(2, 5)))
        assert -5 <= m <= 5
        assert m != 1


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
        ls = tc.any(lists(integers(0, 10), min_size=1, max_size=3))
        assert 1 <= len(ls) <= 3


def test_fixed_size_list():
    @run_test(database={})
    def _(tc):
        ls = tc.any(lists(integers(0, 10), min_size=3, max_size=3))
        assert len(ls) == 3


def test_many_with_small_max():
    """Exercise the many() geometric distribution with a small
    max_size, which triggers the p_continue refinement path."""

    @run_test(database={}, max_examples=200)
    def _(tc):
        ls = tc.any(lists(integers(0, 10), max_size=2))
        assert len(ls) <= 2


def test_many_reject():
    """Test that many()'s reject() mechanism works: too many
    rejections force the collection to stop."""
    from generators import many

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
    from generators import many

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
        v = tc.any(sampled_from(["a", "b", "c"]))
        assert v in ("a", "b", "c")


def test_sampled_from_shrinks_to_first(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            v = tc.any(sampled_from(["a", "b", "c"]))
            assert v != "a"

    captured = capsys.readouterr()
    assert "'a'" in captured.out


def test_sampled_from_single():
    @run_test(database={})
    def _(tc):
        assert tc.any(sampled_from(["only"])) == "only"


def test_sampled_from_empty():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(tc):
            tc.any(sampled_from([]))


def test_booleans():
    @run_test(database={})
    def _(tc):
        b = tc.any(booleans())
        assert isinstance(b, bool)


def test_composite():
    @composite
    def pairs(tc):
        x = tc.any(integers(0, 10))
        y = tc.any(integers(x, 10))
        return (x, y)

    @run_test(database={})
    def _(tc):
        x, y = tc.any(pairs())
        assert x <= y <= 10


def test_composite_with_args():
    @composite
    def bounded_int(tc, max_val):
        return tc.any(integers(0, max_val))

    @run_test(database={})
    def _(tc):
        n = tc.any(bounded_int(5))
        assert 0 <= n <= 5


def test_composite_shrinks(capsys):
    @composite
    def pairs(tc):
        x = tc.any(integers(0, 100))
        y = tc.any(integers(0, 100))
        return (x, y)

    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x, y = tc.any(pairs())
            assert x + y < 100

    captured = capsys.readouterr()
    assert "100, 0" in captured.out or "0, 100" in captured.out


def test_forced_choice_bounds():
    with pytest.raises(ValueError):

        @run_test(database={})
        def _(tc):
            tc.forced_choice(2**64)


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


def test_malformed_database_entry():
    """Malformed database entries are silently ignored."""
    db = {"_": b"\xff\xff\xff"}

    @run_test(database=db, max_examples=1)
    def _(test_case):
        pass


def test_empty_database_entry():
    """Empty database entries produce an empty replay."""
    db = {"_": b""}

    @run_test(database=db, max_examples=1)
    def _(test_case):
        pass


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


class Failure(Exception):
    pass


@settings(
    suppress_health_check=list(HealthCheck),
    deadline=None,
    report_multiple_bugs=False,
    max_examples=50,
)
@given(st.data())
def test_give_minithesis_a_workout(data):
    seed = data.draw(st.integers(0, 1000))
    rnd = Random(seed)
    max_examples = data.draw(st.integers(1, 100))

    method_call = st.one_of(
        st.tuples(
            st.just("mark_status"),
            st.sampled_from((Status.INVALID, Status.VALID, Status.INTERESTING)),
        ),
        st.tuples(st.just("target"), st.floats(0.0, 1.0)),
        st.tuples(st.just("choice"), st.integers(0, 1000)),
        st.tuples(st.just("weighted"), st.floats(0.0, 1.0)),
    )

    def new_node():
        return [None, defaultdict(new_node)]

    tree = new_node()

    database = {}
    failed = False
    call_count = 0
    valid_count = 0

    try:
        try:

            @run_test(
                max_examples=max_examples,
                random=rnd,
                database=database,
                quiet=True,
            )
            def test_function(test_case):
                node = tree
                depth = 0
                nonlocal call_count, valid_count, failed
                call_count += 1

                while True:
                    depth += 1
                    if node[0] is None:
                        node[0] = data.draw(method_call)
                    if node[0] == ("mark_status", Status.INTERESTING):
                        failed = True
                        raise Failure()
                    if node[0] == ("mark_status", Status.VALID):
                        valid_count += 1
                    name, *rest = node[0]

                    result = getattr(test_case, name)(*rest)
                    node = node[1][result]

        except Failure:
            failed = True
        except Unsatisfiable:
            reject()

        if not failed:
            assert valid_count <= max_examples
            assert call_count <= max_examples * 10
    except Exception as e:

        @note
        def tree_as_code():
            """If the test fails, print out a test that will trigger that
            failure rather than making me hand-edit it into something useful."""

            i = 1
            while True:
                test_name = f"test_failure_from_hypothesis_{i}"
                if test_name not in globals():
                    break
                i += 1

            lines = [
                f"def {test_name}():",
                "    with pytest.raises(Failure):",
                f"        @run_test(max_examples=1000, database={{}}, random=Random({seed}))",
                "        def _(tc):",
            ]

            varcount = 0

            def recur(indent, node):
                nonlocal varcount

                if node[0] is None:
                    lines.append(" " * indent + "tc.reject()")
                    return

                method, *args = node[0]
                if method == "mark_status":
                    if args[0] == Status.INTERESTING:
                        lines.append(" " * indent + "raise Failure()")
                    elif args[0] == Status.VALID:
                        lines.append(" " * indent + "return")
                    elif args[0] == Status.INVALID:
                        lines.append(" " * indent + "tc.reject()")
                    else:
                        lines.append(
                            " " * indent + f"tc.mark_status(Status.{args[0].name})"
                        )
                elif method == "target":
                    lines.append(" " * indent + f"tc.target({args[0]})")
                    recur(indent, *node[1].values())
                elif method == "weighted":
                    cond = f"tc.weighted({args[0]})"
                    assert len(node[1]) > 0
                    if len(node[1]) == 2:
                        lines.append(" " * indent + "if {cond}:")
                        recur(indent + 4, node[1][True])
                        lines.append(" " * indent + "else:")
                        recur(indent + 4, node[1][False])
                    else:
                        if True in node[1]:
                            lines.append(" " * indent + f"if {cond}:")
                            recur(indent + 4, node[1][True])
                        else:
                            assert False in node[1]
                            lines.append(" " * indent + f"if not {cond}:")
                            recur(indent + 4, node[1][False])
                else:
                    varcount += 1
                    varname = f"n{varcount}"
                    lines.append(
                        " " * indent
                        + f"{varname} = tc.{method}({', '.join(map(repr, args))})"
                    )
                    first = True
                    for k, v in node[1].items():
                        if v[0] == ("mark_status", Status.INVALID):
                            continue
                        lines.append(
                            " " * indent
                            + ("if" if first else "elif")
                            + f" {varname} == {k}:"
                        )
                        first = False
                        recur(indent + 4, v)
                    lines.append(" " * indent + "else:")
                    lines.append(" " * (indent + 4) + "tc.reject()")

            recur(12, tree)
            return "\n".join(lines)

        raise e


def test_failure_from_hypothesis_1():
    with pytest.raises(Failure):

        @run_test(max_examples=1000, database={}, random=Random(100))
        def _(tc):
            n1 = tc.weighted(0.0)
            if not n1:
                n2 = tc.choice(511)
                if n2 == 112:
                    n3 = tc.choice(511)
                    if n3 == 124:
                        raise Failure()
                    elif n3 == 93:
                        raise Failure()
                    else:
                        tc.mark_status(Status.INVALID)
                elif n2 == 93:
                    raise Failure()
                else:
                    tc.mark_status(Status.INVALID)


def test_failure_from_hypothesis_2():
    with pytest.raises(Failure):

        @run_test(max_examples=1000, database={}, random=Random(0))
        def _(tc):
            n1 = tc.choice(6)
            if n1 == 6:
                n2 = tc.weighted(0.0)
                if not n2:
                    raise Failure()
            elif n1 == 4:
                n3 = tc.choice(0)
                if n3 == 0:
                    raise Failure()
                else:
                    tc.mark_status(Status.INVALID)
            elif n1 == 2:
                raise Failure()
            else:
                tc.mark_status(Status.INVALID)
