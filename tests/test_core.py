# This file is part of Pbtkit, which may be found at
# https://github.com/DRMacIver/pbtkit
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
import struct
import unittest.mock
from random import Random

import pytest

import pbtkit.core as core
import pbtkit.generators as gs
from pbtkit import Unsatisfiable, run_test
from pbtkit.caching import CachedTestFunction, _cache_key
from pbtkit.core import (
    Frozen,
    IntegerChoice,
    Status,
)
from pbtkit.core import PbtkitState as State
from pbtkit.core import TestCase as TC
from pbtkit.database import DirectoryDB


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
        v = tc.draw(gs.floats(allow_nan=False, allow_infinity=False))
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


# ---------------------------------------------------------------------------
# Core-only tests: exercise paths that don't require any extensions.
# These ensure 100% coverage of the compiled minimal build.
# ---------------------------------------------------------------------------


def test_flat_map_core():
    """flat_map works with core types only."""

    @run_test(database={})
    def _(tc):
        m, n = tc.draw(
            gs.integers(0, 5).flat_map(
                lambda m: gs.tuples(gs.just(m), gs.integers(m, m + 10))
            )
        )
        assert m <= n <= m + 10


def test_filter_core():
    """filter works with core types only."""

    @run_test(database={})
    def _(tc):
        n = tc.draw(gs.integers(0, 10).filter(lambda n: n % 2 == 0))
        assert n % 2 == 0


def test_nothing_core():
    """nothing() raises Unsatisfiable."""
    with pytest.raises(Unsatisfiable):

        @run_test(database={})
        def _(tc):
            tc.draw(gs.nothing())


def test_one_of_empty_core():
    """one_of() with no args raises Unsatisfiable."""
    with pytest.raises(Unsatisfiable):

        @run_test(database={})
        def _(tc):
            tc.draw(gs.one_of())


def test_one_of_single_core():
    """one_of with a single generator passes through."""

    @run_test(database={})
    def _(tc):
        n = tc.draw(gs.one_of(gs.integers(0, 10)))
        assert 0 <= n <= 10


def test_sampled_from_core():
    """sampled_from with basic values."""

    @run_test(database={})
    def _(tc):
        v = tc.draw(gs.sampled_from(["a", "b", "c"]))
        assert v in ("a", "b", "c")


def test_sampled_from_empty_core():
    """sampled_from([]) raises Unsatisfiable."""
    with pytest.raises(Unsatisfiable):

        @run_test(database={})
        def _(tc):
            tc.draw(gs.sampled_from([]))


def test_sampled_from_single_core():
    """sampled_from with one element returns just that."""

    @run_test(database={})
    def _(tc):
        assert tc.draw(gs.sampled_from(["only"])) == "only"


def test_just_core():
    """just(v) always returns v."""

    @run_test(database={})
    def _(tc):
        assert tc.draw(gs.just(42)) == 42


def test_weighted_forced_true():
    """weighted(1.0) always returns True (forced=True path)."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            if tc.weighted(1.0):
                tc.choice(1)
                assert False


def test_draw_silent_does_not_print(capsys):
    """draw_silent returns a value but never prints it, even on the final replay."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            n = tc.draw_silent(gs.integers(0, 10))
            assert n == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == ""


def test_note_prints_on_failing_example(capsys):
    """note() prints during the final failing replay (print_results=True),
    and is a no-op during generation runs (print_results=False)."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            tc.note("hello from note")
            assert False

    captured = capsys.readouterr()
    assert "hello from note" in captured.out


@pytest.mark.requires("database")
def test_database_round_trip_with_booleans(tmp_path):
    """Database serializes and deserializes boolean values."""
    db = DirectoryDB(str(tmp_path))
    count = 0

    def run():
        with pytest.raises(AssertionError):

            @run_test(database=db)
            def _(tc):
                nonlocal count
                count += 1
                b = tc.weighted(0.5)
                assert not b

    run()
    prev = count
    run()
    assert count == prev + 2


def test_map_core():
    """Generator.map works with core types."""

    @run_test(database={})
    def _(tc):
        n = tc.draw(gs.integers(0, 5).map(lambda n: n * 2))
        assert n % 2 == 0


def test_delete_chunks_stale_index():
    """delete_chunks must handle i going past the end of the result
    when a successful deletion shortens it."""

    def tf(tc):
        # Always interesting, variable length. With many nodes,
        # chunk deletions succeed and shorten the result, causing
        # the loop index i to go past the new end.
        n = tc.draw_integer(0, 30)
        for _ in range(n):
            tc.draw_integer(0, 10)
        tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 200)
    state.run()
    assert state.result is not None
    # Shrinks to n=0, one node.
    assert len(state.result) == 1


def test_run_test_with_preseeded_result():
    """Exercise the run_test path where state.result is not None
    before state.run() (normally set by database hooks)."""
    original_init = State.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        tc = TC.for_choices([42])
        self.test_function(tc)

    with unittest.mock.patch.object(State, "__init__", patched_init):
        with pytest.raises(AssertionError):

            @run_test(database={}, max_examples=1)
            def _(tc):
                n = tc.draw_integer(0, 100)
                assert n < 42


@pytest.mark.requires("targeting")
def test_targeting_skips_non_integer():
    """Targeting skips non-integer nodes (booleans) without crashing."""
    max_score = 0

    @run_test(database={}, max_examples=200)
    def _(tc):
        nonlocal max_score
        tc.weighted(0.5)  # boolean node
        n = tc.choice(100)
        tc.target(n)
        max_score = max(n, max_score)

    assert max_score == 100


def test_bin_search_down_lo_satisfies():
    """bin_search_down returns lo when f(lo) is True. Use a range
    where simplest != 0 so zero_choices can't get there first."""

    def tf(tc):
        n = tc.draw_integer(5, 100)
        # Always interesting — binary search tries simplest (5) first.
        tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    assert state.result[0].value == 5


def test_sort_key_type_mismatch():
    """sort_key methods handle wrong-type values gracefully.

    During shrinking, type changes in the choice sequence can cause a
    ChoiceNode to have a value that doesn't match its kind."""
    from pbtkit.bytes import BytesChoice
    from pbtkit.floats import FloatChoice
    from pbtkit.text import StringChoice

    assert StringChoice(0, 127, 0, 10).sort_key(42) == (0, ())
    assert BytesChoice(0, 10).sort_key(42) == (0, b"")
    assert FloatChoice(-1.0, 1.0, False, False).sort_key("hello") == (0, 0)


def test_shrink_duplicates_with_stale_indices():
    """Regression: duplication pass crashed with AttributeError:
    'BooleanChoice' object has no attribute 'max_value'.

    Exact program from pbtsmith that triggered the crash."""
    from pbtkit.generators import (
        booleans,
        integers,
        just,
        lists,
        sampled_from,
    )

    @gs.composite
    def _tree_node(tc, depth):
        if depth <= 0 or not tc.weighted(0.9):
            return tc.draw(booleans())
        op = tc.choice(3)
        if op == 0:
            child = tc.draw(_tree_node(depth - 1))
            return ("neg", child)
        if op == 1:
            left = tc.draw(_tree_node(depth - 1))
            right = tc.draw(_tree_node(depth - 1))
            return ("add", left, right)
        if op == 2:
            left = tc.draw(_tree_node(depth - 1))
            right = tc.draw(_tree_node(depth - 1))
            return ("sub", left, right)
        left = tc.draw(_tree_node(depth - 1))
        right = tc.draw(_tree_node(depth - 1))
        return ("mul", left, right)

    def _tree():
        return integers(0, 2).flat_map(lambda d: _tree_node(d))

    def tree_size(node):
        if not isinstance(node, tuple) or len(node) == 0:
            return 1
        return 1 + sum(tree_size(child) for child in node[1:])

    def tree_nodes(node):
        result = [node]
        if isinstance(node, tuple) and len(node) > 0:
            for child in node[1:]:
                result.extend(tree_nodes(child))
        return result

    def tree_leaves(node):
        if not isinstance(node, tuple) or len(node) == 0:
            return 1
        return sum(tree_leaves(child) for child in node[1:])

    class Failure(Exception):
        pass

    try:

        @run_test(max_examples=1, database={}, quiet=True, random=Random(0))
        def _(tc):
            v0 = tc.draw(_tree().filter(lambda t: isinstance(t, tuple)))
            _nodes_v0 = tree_nodes(v0)
            tc.draw(_tree().filter(lambda t: isinstance(t, tuple)))
            tc.draw(lists(booleans(), max_size=tree_leaves(v0) + 1))
            if not (tree_size(v0) < 5):
                raise Failure("tree_size(v0) < 5")
            v3 = tc.draw(sampled_from(_nodes_v0) if _nodes_v0 else just(0))
            if not isinstance(v3, tuple):
                raise Failure("isinstance(v3, tuple)")
    except (Unsatisfiable, Failure):
        pass
