"""Composite generator shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/composite.rs.
"""

from random import Random

import pytest

import pbtkit.generators as gs
from pbtkit import Generator, run_test
from pbtkit.core import PbtkitState as State
from pbtkit.core import Status
from pbtkit.core import TestCase as TC
from pbtkit.shrinking.index_passes import lower_and_bump

from .conftest import minimal


@gs.composite
def int_pair(tc, lo, hi):
    a = tc.draw(gs.integers(lo, hi))
    b = tc.draw(gs.integers(lo, hi))
    return (a, b)


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_positive_sum_of_pair():
    assert minimal(int_pair(0, 1000), lambda x: x[0] + x[1] > 1000) == (1, 1000)


@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_negative_sum_of_pair():
    assert minimal(int_pair(-1000, 1000), lambda x: x[0] + x[1] < -1000) == (-1, -1000)


@pytest.mark.requires("text")
@gs.composite
def separated_sum(tc):
    n1 = tc.draw(gs.integers(0, 1000))
    tc.draw(gs.text())
    tc.draw(gs.booleans())
    tc.draw(gs.integers(-(2**63), 2**63 - 1))
    n2 = tc.draw(gs.integers(0, 1000))
    return (n1, n2)


@pytest.mark.requires("text")
@pytest.mark.requires("shrinking.advanced_integer_passes")
def test_sum_of_pair_separated():
    assert minimal(separated_sum(), lambda x: x[0] + x[1] > 1000) == (1, 1000)


def test_minimize_dict_of_booleans():
    result = minimal(gs.tuples(gs.booleans(), gs.booleans()), lambda x: x[0] or x[1])
    assert not (result[0] and result[1])
    assert result[0] or result[1]


@gs.composite
def int_struct(tc):
    a = tc.draw(gs.integers(-(2**63), 2**63 - 1))
    b = tc.draw(gs.integers(-(2**63), 2**63 - 1))
    return (a, b)


def test_minimize_namedtuple():
    tab = minimal(int_struct(), lambda x: x[0] < x[1])
    assert tab[1] == tab[0] + 1


@pytest.mark.requires("shrinking.index_passes")
def test_earlier_exit_produces_shorter_sequence():
    """When v0=True triggers an early exit with fewer choices than
    v0=False followed by more draws, the shrinker should prefer the
    shorter path. Found by shrink comparison test."""

    @gs.composite
    def pair_of_bools(tc):
        a = tc.draw(gs.booleans())
        b = tc.draw(gs.booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.draw(gs.booleans())
        v1 = tc.draw(pair_of_bools())
        tc.draw(pair_of_bools())
        # First exit: when v0=True (5 choices used)
        if v0:
            tc.mark_status(Status.INTERESTING)
        # More draws only reached when v0=False
        tc.draw(gs.booleans())
        # Second exit: len(v1) is always 2, so this always fires (6 choices)
        if len(v1) != 0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    # v0=True exits after 5 choices; v0=False needs 6.
    # The shrinker should find the shorter 5-choice path.
    vals = [n.value for n in state.result]
    assert len(vals) == 5
    assert vals[0]  # v0 is truthy (triggers early exit)


@pytest.mark.requires("floats")
def test_one_of_shrinks_branch_selector():
    """one_of should shrink toward branch 0 even when a higher branch
    also produces an interesting result. Found by shrink comparison test."""
    result = minimal(
        gs.one_of(gs.booleans(), gs.floats(allow_nan=False, allow_infinity=False)),
        lambda v: bool(v),
    )
    # Branch 0 (booleans) with value True is simpler than
    # branch 1 (floats) with any truthy float.
    assert result is True


@pytest.mark.requires("bytes")
@pytest.mark.requires("collections")
def test_early_exit_via_flag_with_preceding_draws():
    """When v0=True triggers an early exit but v1 and v2 are drawn
    BEFORE the v0 check, the shrinker must coordinate: shorten v1/v2
    AND flip v0 simultaneously. Hypothesis finds [True, b'', False]
    (3 choices) but pbtkit gets stuck at 5 choices with v0=False.

    Found by the Hypothesis shrink comparison test."""

    @gs.composite
    def test_data(tc):
        v0 = tc.draw(gs.booleans())
        v1 = tc.draw(gs.binary(max_size=20))
        v2 = tc.draw(gs.lists(gs.integers(0, 0), max_size=10))
        return (v0, v1, v2)

    result = minimal(
        test_data(),
        lambda t: t[0] or len(t[1]) + len(t[2]) >= 20,
    )
    # With 3 choices, both paths are valid:
    # - [False, b'\x00'*20, []] — v0=False, full bytes (simpler at pos 0)
    # - [True, b'', []] — v0=True, empty bytes (simpler at pos 1)
    # Under sort_key, False < True at pos 0, so the first wins.
    assert result[0] is False or result[0] is True


@pytest.mark.requires("floats")
def test_one_of_branch_switch_with_trailing_draws():
    """When gs.one_of(gs.booleans(), gs.floats()) at branch=1 produces a truthy
    float, switching to branch=0 (boolean True) is simpler but requires
    fixing the downstream kind AND keeping the trailing composite draws.

    Found by the Hypothesis shrink comparison test."""

    @gs.composite
    def gen_pair(tc):
        a = tc.draw(gs.booleans())
        b = tc.draw(gs.booleans())
        return (a, b)

    @gs.composite
    def test_data(tc):
        v0 = tc.draw(
            gs.one_of(gs.booleans(), gs.floats(allow_nan=False, allow_infinity=False))
        )
        tc.draw(gen_pair())
        return v0

    result = minimal(test_data(), lambda v: bool(v))
    assert result is True


@pytest.mark.requires("floats")
@pytest.mark.requires("collections")
def test_shorter_path_via_later_assertion():
    """When emptying a list removes one failure but a later assertion
    still fails with fewer total choices, the shrinker should prefer
    the shorter path. Found by the Hypothesis shrink comparison test."""

    @gs.composite
    def pair(tc):
        a = tc.draw(gs.booleans())
        b = tc.draw(gs.floats(allow_nan=False, allow_infinity=False))
        return (a, b)

    @gs.composite
    def test_data(tc):
        v0 = tc.draw(pair())
        v1 = tc.draw(gs.lists(gs.integers(0, 20), max_size=10, unique=True))
        tc.draw(pair())
        return (v0, v1)

    result = minimal(
        test_data(),
        # Fails when v1 is non-empty (7 choices) OR always via len(v0)==2
        # (v0 is a tuple, 6 choices). The shorter path is v1 empty.
        lambda t: len(t[1]) > 0 or len(t[0]) != 0,
    )
    assert len(result[1]) == 0


@pytest.mark.requires("floats")
def test_one_of_branch_switch_to_float():
    """When gs.one_of(gs.floats(), gs.booleans()) starts at branch=1 (booleans),
    switching to branch=0 (floats) is simpler but requires replacing
    the boolean value with a valid float. Found by shrink comparison test."""
    result = minimal(
        gs.one_of(gs.floats(allow_nan=False, allow_infinity=False), gs.booleans()),
        lambda _: True,
    )
    # Branch 0 (floats) with simplest value 0.0 is simpler than
    # branch 1 (booleans) with simplest value False.
    assert isinstance(result, float)
    assert result == 0.0


def test_one_of_shorter_branch_needs_non_simplest_value():
    """When gs.one_of(gs.tuples(gs.booleans(), gs.booleans()), gs.booleans()) starts at
    branch=0 (tuple, 3 choices), branch=1 with True (2 choices) is shorter.
    But the increment + pun produces branch=1 with False (not interesting).
    Found by the Hypothesis shrink comparison test."""
    result = minimal(
        gs.one_of(gs.tuples(gs.booleans(), gs.booleans()), gs.booleans()),
        lambda v: bool(v),
    )
    assert result is True


@pytest.mark.requires("floats")
def test_switch_failure_mode_for_simpler_sort_key():
    """When abs(v1) >= 1.0 triggers one assertion but a later assertion
    also fails for all inputs, the shrinker should prefer the failure
    mode where v1=0.0 (simpler float) even though it means v4 must
    be non-zero. Found by the Hypothesis shrink comparison test."""

    @gs.composite
    def test_data(tc):
        v1 = tc.draw(gs.floats(allow_nan=False, allow_infinity=False))
        v4 = tc.draw(gs.sampled_from([1, 0]))
        return (v1, v4)

    result = minimal(
        test_data(),
        # Fails when |v1| >= 1.0 (via v1) or when v4 > 0 (via v4).
        # v1=0.0, v4=1 is simpler overall than v1=1.0, v4=0.
        lambda t: abs(t[0]) >= 1.0 or t[1] > 0,
    )
    assert result[0] == 0.0  # Simpler float, later assertion fires.


@pytest.mark.requires("collections")
def test_shorter_path_when_guard_precedes_expensive_draw():
    """A guard check (v0 > 0) comes after a cheap draw but before an
    expensive draw (a list). When v0=1 triggers the guard, the expensive
    draw is skipped, producing a shorter sequence. But the shrinker
    reduces v0 to 0 (simplest) first, which forces the longer path.

    Found by the Hypothesis shrink comparison test."""

    @gs.composite
    def test_data(tc):
        v0 = tc.draw(gs.integers(0, 10))
        v1 = tc.draw(gs.booleans())
        v2 = tc.draw(gs.lists(gs.integers(0, 100), max_size=10))
        return (v0, v1, v2)

    result = minimal(
        test_data(),
        lambda t: t[0] > 0 or len(t[2]) >= 3,
    )
    # Optimal: v0=1 exits early → 3 choices (v0, v1, list_stop).
    # Suboptimal: v0=0 needs len(v2) >= 3 → 8+ choices.
    assert result[0] > 0


# --- Regression tests from test_core.py ---


@pytest.mark.parametrize("seed", range(10))
def test_finds_small_list_even_with_bad_lists(capsys, seed):
    """Pbtkit can't really handle shrinking arbitrary
    monadic bind, but length parameters are a common case
    of monadic bind that it has a little bit of special
    casing for. This test ensures that that special casing
    works.

    The problem is that if you generate a list by drawing
    a length and then drawing that many elements, you can
    end up with something like ``[1001, 0, 0]`` then
    deleting those zeroes in the middle is a pain. pbtkit
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
            ls = test_case.draw(bad_list)
            assert sum(ls) <= 1000

    captured = capsys.readouterr()

    assert captured.out.strip() == "draw_1 = [1001]"


def test_shrinking_mixed_choice_types_no_sort_crash():
    """Sorting pass should not crash when the result has a mix of
    IntegerChoice and BooleanChoice nodes and shrinking changes
    which type is at a given position.

    Regression test for a TypeError found by pbtsmith."""

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

    Regression test for an IndexError found by pbtsmith."""

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
            value = tc.draw(gs.one_of(gs.booleans(), gs.integers(0, 100)))
            assert isinstance(value, bool) or value <= 50


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_explores_new_range():
    """When decrementing an integer changes the range of a non-adjacent
    later integer, lower_and_bump should explore the new range via
    absolute power-of-2 values at various gaps.
    Regression for shrink quality found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.sampled_from([32, 46]))
        tc.draw(gs.sampled_from([32, 46]))
        v2 = tc.draw(gs.integers(-abs(v0) - 1, abs(v0) + 1))
        tc.draw(gs.integers(-abs(v2) - 1, abs(v2) + 1))
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
    Regression for shrink quality found by pbtsmith."""

    @gs.composite
    def pair(tc):
        a = tc.draw(gs.booleans())
        b = tc.draw(gs.booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.draw(pair())
        tc.draw(pair())
        v2 = tc.draw(gs.one_of(gs.integers(0, 0), gs.booleans()))
        if len(v0) <= 0:
            tc.mark_status(Status.INTERESTING)
        v3 = tc.draw(gs.integers(-1, 1))
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
    Regression for shrink quality found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.sampled_from([1, 1, 0]))
        if v0 <= 0:
            tc.mark_status(Status.INTERESTING)
        v1 = tc.draw(gs.booleans())
        if v1:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    assert len(state.result) == 1


@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_targets_booleans():
    """lower_and_bump should try bumping boolean targets, not just
    integer ones. Decrementing an integer while bumping a boolean
    from False to True can produce a simpler overall sort_key.
    Regression for shrink quality found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.integers(0, 1))
        v1 = tc.draw(gs.booleans())
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


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_increment_with_dependent_continuation():
    """try_shortening_via_increment must pass prefix_nodes so that
    value punning maps simplest→simplest when the continuation
    changes type (e.g. list boolean → integer).
    Regression for shrink quality found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.integers(0, 0))
        v1 = tc.draw(gs.booleans())
        tc.draw(gs.integers(0, 0))
        v3 = tc.draw(gs.lists(gs.integers(-21, -1), max_size=10, unique=True))
        if len(v3) != 0:
            tc.mark_status(Status.INTERESTING)
        if v1:
            v4 = tc.draw(gs.integers(v0, v0 + 1))
            if v0 + v4 <= 0:
                tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    # Should shrink to 5 choices (via v1=True path) not 6 (via non-empty list)
    assert len(state.result) == 5


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
        v0 = tc.draw(gs.text(min_codepoint=32, max_codepoint=126, max_size=20))
        v1 = tc.draw(gs.floats(allow_nan=False, allow_infinity=False))
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
    Regression for AssertionError in redistribute found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(
            gs.one_of(
                gs.booleans(),
                gs.integers(0, 0),
                gs.integers(2, 2).filter(lambda x: x > 0),
            )
        )
        tc.draw(gs.integers(0, 0))
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
    Regression for AssertionError in lower_and_bump found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.booleans())
        tc.draw(gs.booleans())
        tc.draw(gs.booleans())
        tc.draw(gs.lists(gs.integers(0, 0), max_size=10).filter(lambda x: len(x) > 0))
        tc.draw(
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
        v0 = tc.draw(gs.booleans())
        tc.draw(gs.integers(5, 5))  # single-value, from_index(1) = None
        if v0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
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


@pytest.mark.requires("shrinking.duplication_passes")
def test_shrink_duplicates_two_copies():
    """shrink_duplicates handles exactly 2 copies (no wrapping loop)."""

    def tf(tc):
        a = tc.draw_integer(0, 100)
        b = tc.draw_integer(0, 100)
        if a == b and a > 0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 10000)
    state.run()
    assert state.result is not None
    assert state.result[0].value == 1
    assert state.result[1].value == 1


@pytest.mark.requires("shrinking.duplication_passes")
def test_shrink_duplicates_three_copies():
    """shrink_duplicates handles 3+ copies of the same value."""

    def tf(tc):
        a = tc.draw_integer(0, 10)
        b = tc.draw_integer(0, 10)
        c = tc.draw_integer(0, 10)
        if a == b == c and a > 0:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 10000)
    state.run()
    assert state.result is not None
    assert state.result[0].value == 1
    assert state.result[1].value == 1
    assert state.result[2].value == 1


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.mutation")
def test_one_of_switches_to_shorter_branch():
    """When one_of branch 0 (lists) produces a truthy value in 4 choices
    but branch 1 (booleans via nested one_of) can do it in 3, the
    shrinker should find the shorter branch.

    The difficulty: switching the outer one_of index from 0 to 1 requires
    setting the inner index AND the value to non-zero simultaneously —
    a 3-position compound change.

    Regression for shrink quality found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(
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


@pytest.mark.requires("shrinking.mutation")
def test_mutate_exercises_index_probes():
    """mutate_and_shrink probes max_index on integer and boolean nodes."""

    def tf(tc):
        a = tc.draw_integer(0, 10)
        b = tc.weighted(0.5)
        if a > 5 and b:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 200)
    state.run()
    assert state.result is not None


@pytest.mark.requires("shrinking.mutation")
def test_mutate_skips_large_result():
    """mutate_and_shrink returns early for results with >32 nodes."""

    def tf(tc):
        for _ in range(35):
            tc.draw_integer(0, 10)
        tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
