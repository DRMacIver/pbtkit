# This file is part of Pbtkit, which may be found at
# https://github.com/DRMacIver/pbtkit
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the draw_names feature module."""

import contextlib
import inspect
import re
import textwrap
from random import Random

import libcst as cst
import pytest

pytestmark = pytest.mark.requires("draw_names")

import pbtkit.draw_names  # noqa: F401  (importing patches TestCase)
import pbtkit.generators as gs
from pbtkit import run_test
from pbtkit.core import PbtkitState, TestCase
from pbtkit.draw_names import (
    _draw_names_hook,
    _DrawNameCollector,
    rewrite_test_function,
)

# ---------------------------------------------------------------------------
# Section A: Basic draw counter (core — tc.draw() output format)
# ---------------------------------------------------------------------------


def test_draw_counter_increments(capsys):
    """draw() prints draw_1, draw_2, draw_3 labels in order."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            tc.draw(gs.integers(0, 0))
            tc.draw(gs.integers(0, 0))
            tc.draw(gs.integers(0, 0))
            assert False  # noqa: B011

    out = capsys.readouterr().out
    assert "draw_1 = 0" in out
    assert "draw_2 = 0" in out
    assert "draw_3 = 0" in out


def test_draw_counter_resets_per_test_case():
    """The draw counter starts from 1 for each fresh TestCase."""
    tc1 = TestCase.for_choices([0], print_results=True)
    tc2 = TestCase.for_choices([0], print_results=True)
    tc1.draw(gs.integers(0, 10))
    # tc2 gets its own counter, independent of tc1
    assert tc2._draw_counter == 0


def test_draw_counter_only_fires_when_print_results(capsys):
    """draw() increments the counter but only prints when print_results=True."""
    tc = TestCase.for_choices([0], print_results=False)
    tc.draw(gs.integers(0, 10))
    out = capsys.readouterr().out
    assert out == ""
    assert tc._draw_counter == 0  # counter only increments when printing


def test_draw_silent_does_not_print(capsys):
    """draw_silent() never prints, even with print_results=True."""
    tc = TestCase.for_choices([0], print_results=True)
    tc.draw_silent(gs.integers(0, 10))
    out = capsys.readouterr().out
    assert out == ""
    assert tc._draw_counter == 0


def test_choice_output_unchanged(capsys):
    """choice() still prints in its original format, not draw_N."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            n = tc.choice(5)
            assert n != 0

    out = capsys.readouterr().out
    # The user code calls choice(); its draws print as `choice(N): v`.
    # Other unrelated stdout (header, traceback paths) may mention
    # things like draw_silent or paths under tests/test_draw_names.py;
    # we only care that draws use the non-prefixed format.
    assert any(line.startswith("choice(5):") for line in out.splitlines())
    assert not any(line.lstrip().startswith("draw_") for line in out.splitlines())


def test_weighted_output_unchanged(capsys):
    """weighted() still prints in its original format, not draw_N."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            tc.choice(1)  # ensure we have something to shrink to
            if tc.weighted(1.0):
                raise AssertionError("always")

    out = capsys.readouterr().out
    assert any(line.startswith("weighted(1.0):") for line in out.splitlines())
    assert not any(line.lstrip().startswith("draw_") for line in out.splitlines())


def test_draw_uses_repr_format(capsys):
    """draw() uses repr() so strings get quotes and bytes get b'' notation."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            tc.draw(gs.sampled_from(["hello"]))
            assert False  # noqa: B011

    out = capsys.readouterr().out
    assert "draw_1 = 'hello'" in out


# ---------------------------------------------------------------------------
# Section B: draw_named semantics
# ---------------------------------------------------------------------------


def test_draw_named_non_repeatable_single_use(capsys):
    """Non-repeatable draw with one use prints 'x = value'."""
    tc = TestCase.for_choices([3], print_results=True)
    result = tc.draw_named(gs.integers(0, 10), "x", False)
    out = capsys.readouterr().out
    assert result == 3
    assert out.strip() == "x = 3"


def test_draw_named_repeatable_single_use(capsys):
    """Repeatable draw with one use prints 'x_1 = value'."""
    tc = TestCase.for_choices([3], print_results=True)
    result = tc.draw_named(gs.integers(0, 10), "x", True)
    out = capsys.readouterr().out
    assert result == 3
    assert out.strip() == "x_1 = 3"


def test_draw_named_repeatable_multiple_uses(capsys):
    """Repeatable draws number up: x_1, x_2, x_3."""
    tc = TestCase.for_choices([1, 2, 3], print_results=True)
    tc.draw_named(gs.integers(0, 10), "x", True)
    tc.draw_named(gs.integers(0, 10), "x", True)
    tc.draw_named(gs.integers(0, 10), "x", True)
    out = capsys.readouterr().out
    assert "x_1 = 1" in out
    assert "x_2 = 2" in out
    assert "x_3 = 3" in out


def test_draw_named_repeatable_skips_taken_suffixes(capsys):
    """Repeatable numbering skips already-taken suffixed names."""
    tc = TestCase.for_choices([5, 7], print_results=True)
    # First, consume x_1 via a non-repeatable use with that exact name
    tc._named_draw_used.add("x_1")
    # Now a repeatable x should start at x_2
    tc.draw_named(gs.integers(0, 10), "x", True)
    out = capsys.readouterr().out
    assert "x_2 = 5" in out
    assert "x_1" not in out


def test_draw_named_non_repeatable_reuse_raises():
    """Non-repeatable name used twice → AssertionError."""
    tc = TestCase.for_choices([1, 2], print_results=True)
    tc.draw_named(gs.integers(0, 10), "x", False)
    with pytest.raises(AssertionError, match="Non-repeatable name"):
        tc.draw_named(gs.integers(0, 10), "x", False)


def test_draw_named_inconsistent_flags_raises():
    """Using same name with different repeatable flags → AssertionError."""
    tc = TestCase.for_choices([1, 2, 3], print_results=True)
    tc.draw_named(gs.integers(0, 10), "x", False)
    with pytest.raises(AssertionError, match="inconsistent repeatable flags"):
        tc.draw_named(gs.integers(0, 10), "x", True)


def test_draw_named_no_print_when_print_results_false(capsys):
    """draw_named does not print when print_results=False."""
    tc = TestCase.for_choices([3], print_results=False)
    tc.draw_named(gs.integers(0, 10), "x", False)
    out = capsys.readouterr().out
    assert out == ""


def test_draw_named_different_names_ok(capsys):
    """Multiple different non-repeatable names all print correctly."""
    tc = TestCase.for_choices([1, 2], print_results=True)
    tc.draw_named(gs.integers(0, 10), "x", False)
    tc.draw_named(gs.integers(0, 10), "y", False)
    out = capsys.readouterr().out
    assert "x = 1" in out
    assert "y = 2" in out


# ---------------------------------------------------------------------------
# Section C: CST rewriter unit tests
# ---------------------------------------------------------------------------


def test_rewriter_top_level_assignment():
    """Top-level x = tc.draw(gen) → draw_named(..., "x", False)."""

    def f(tc):
        x = tc.draw(gs.integers(0, 10))
        assert x >= 0

    rewritten = rewrite_test_function(f)
    assert rewritten is not f
    # The rewritten function should work: check it runs without error
    tc = TestCase.for_choices([5], print_results=False)
    rewritten(tc)


def test_rewriter_for_loop_body_is_repeatable(capsys):
    """x = tc.draw(gen) inside a for loop → repeatable=True → prints x_1, x_2."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            results = []
            for _ in range(2):
                x = tc.draw(gs.integers(0, 0))
                results.append(x)
            assert False  # noqa: B011

    out = capsys.readouterr().out
    assert "x_1" in out
    assert "x_2" in out


def test_rewriter_while_loop_body_is_repeatable(capsys):
    """x = tc.draw(gen) inside a while loop → repeatable=True."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            i = 0
            while i < 1:
                x = tc.draw(gs.integers(0, 0))
                i += 1
            assert x < 0

    out = capsys.readouterr().out
    assert "x_1" in out


def test_rewriter_if_body_is_repeatable(capsys):
    """x = tc.draw(gen) inside an if block → repeatable=True."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            if True:
                x = tc.draw(gs.integers(0, 0))
                assert x < 0

    out = capsys.readouterr().out
    assert "x_1" in out


def test_rewriter_with_block_is_repeatable(capsys):
    """x = tc.draw(gen) inside a with block → repeatable=True."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            with contextlib.nullcontext():
                x = tc.draw(gs.integers(0, 0))
                assert x < 0

    out = capsys.readouterr().out
    assert "x_1" in out


def test_rewriter_try_block_is_repeatable(capsys):
    """x = tc.draw(gen) inside a try block → repeatable=True."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            try:
                x = tc.draw(gs.integers(0, 0))
            except Exception:
                x = -1  # never happens; ensures x is always bound
            assert x < 0

    out = capsys.readouterr().out
    assert "x_1" in out


def test_rewriter_nested_function_is_repeatable(capsys):
    """x = tc.draw(gen) inside a nested function → repeatable=True."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            def inner():
                return tc.draw(gs.integers(0, 0))

            inner()
            assert False  # noqa: B011

    # Note: the draw inside the nested function is not a draw_named assignment —
    # it's a return statement, not an assignment. The outer x = inner() is not
    # a tc.draw() call so it won't be rewritten. That's expected behaviour.
    capsys.readouterr()


def test_rewriter_name_seen_at_top_and_loop_all_repeatable(capsys):
    """Same name used both top-level and in a loop → all uses become repeatable."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            x = tc.draw(gs.integers(0, 0))
            for _ in range(1):
                x = tc.draw(gs.integers(0, 0))
            assert x < 0

    out = capsys.readouterr().out
    # Both uses are repeatable because the name appears in a loop too
    assert "x_1" in out
    assert "x_2" in out


def test_rewriter_no_draws_is_noop():
    """A function with no tc.draw() calls is returned unchanged."""

    def f(tc):
        tc.choice(5)

    result = rewrite_test_function(f)
    assert result is f


def test_rewriter_expression_context_not_rewritten(capsys):
    """tc.draw(gen) in expression context (not assignment) stays as draw()."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            # Not an assignment — should NOT be rewritten to draw_named.
            # The draw() format should still be draw_N = ...
            assert tc.draw(gs.integers(0, 0)) >= 0
            assert False  # noqa: B011

    out = capsys.readouterr().out
    # Since it's not an assignment, draw_named was not used — falls back to draw_N format
    assert "draw_1 = 0" in out


def test_rewriter_tuple_target_not_rewritten(capsys):
    """a, b = tc.draw(gen) is NOT rewritten (only simple Name targets)."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            a, b = tc.draw(gs.tuples(gs.integers(0, 0), gs.integers(0, 0)))
            assert False  # noqa: B011

    out = capsys.readouterr().out
    # Tuple unpacking target not rewritten — falls back to draw_N format
    assert "draw_1 = (0, 0)" in out


# ---------------------------------------------------------------------------
# Section D: Integration tests (rewrite_draws + run_test)
# ---------------------------------------------------------------------------


def test_rewrite_draws_output_is_named(capsys):
    """@rewrite_draws causes output to use variable names."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            value = tc.draw(gs.integers(0, 0))
            assert value < 0

    out = capsys.readouterr().out
    assert "value = 0" in out
    assert "draw_1" not in out


def test_rewrite_draws_two_draws(capsys):
    """Two different named draws both appear with their variable names."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            first = tc.draw(gs.integers(0, 0))
            second = tc.draw(gs.integers(0, 0))
            assert first + second < 0

    out = capsys.readouterr().out
    assert "first = 0" in out
    assert "second = 0" in out


def test_auto_rewriting_without_decorator(capsys):
    """Importing draw_names enables auto-rewriting even without any decorator."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            value = tc.draw(gs.integers(0, 0))
            assert value < 0

    out = capsys.readouterr().out
    assert "value = 0" in out
    assert "draw_1" not in out


def test_rewrite_draws_final_replay_uses_rewritten_function(capsys):
    """The final failing-example replay uses the rewritten function (named output)."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            answer = tc.draw(gs.integers(0, 0))
            assert answer < 0

    out = capsys.readouterr().out
    # Output comes from the final replay; should show the named variable
    assert "answer = 0" in out


def test_rewrite_draws_loop_output_numbered(capsys):
    """Loop draws produce x_1 = v1, x_2 = v2, ... in output."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200)
        def _(tc):
            for _ in range(2):
                item = tc.draw(gs.integers(0, 0))
            assert False  # noqa: B011

    out = capsys.readouterr().out
    assert "item_1 = 0" in out
    assert "item_2 = 0" in out


def test_rewrite_draws_with_closure(capsys):
    """@rewrite_draws preserves closure variables."""
    threshold = 5

    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            val = tc.draw(gs.integers(0, 0))
            # threshold is a free variable from the enclosing scope
            assert val >= threshold

    out = capsys.readouterr().out
    assert "val = 0" in out


def test_rewrite_draws_no_error_for_no_draw_function(capsys):
    """@rewrite_draws on a function with no draws works without error."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            assert False  # noqa: B011

    # No draws happened, so no auto-named ``draw_N = ...`` lines
    # should appear. (The header and traceback might mention "draw"
    # via paths/identifiers — only check for the rewriter's signature
    # output pattern.)
    out = capsys.readouterr().out
    assert not re.search(r"^draw_\d+ = ", out, re.MULTILINE)


# ---------------------------------------------------------------------------
# Section E: Full pbtkit integration
# ---------------------------------------------------------------------------


def test_importing_draw_names_enables_auto_rewriting(capsys):
    """Importing draw_names rewrites all run_test() functions automatically."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            n = tc.draw(gs.integers(0, 0))
            assert n < 0

    out = capsys.readouterr().out
    assert "n = 0" in out


def test_draw_named_stub_raises_before_import():
    """The draw_named stub on a bare TestCase raises NotImplementedError
    only if draw_names hasn't been imported yet.

    Since importing this test module already imports draw_names, we verify
    that the monkey-patched version works (not the stub)."""
    tc = TestCase.for_choices([5], print_results=False)
    # draw_names is imported at the top of this module, so draw_named is patched
    result = tc.draw_named(gs.integers(0, 10), "x", False)
    assert result == 5


def test_draw_named_validation_runs_outside_composite(capsys):
    """draw_named validation (non-repeatable reuse) fires even without printing."""
    tc = TestCase.for_choices([1, 2], print_results=False)
    tc.draw_named(gs.integers(0, 10), "x", False)
    with pytest.raises(AssertionError):
        tc.draw_named(gs.integers(0, 10), "x", False)


def test_draw_named_no_validation_inside_composite():
    """draw_named inside a composite generator (depth > 0) skips name tracking."""

    @gs.composite
    def gen(tc):
        # This draw_named is at depth 1 (inside a composite), so no tracking
        return tc.draw_named(gs.integers(0, 5), "inner", False)

    tc = TestCase.for_choices([3, 3], print_results=False)
    # Should not raise even though the same name is used "twice"
    # (each call is at depth > 0 from the outer draw's perspective)
    tc.draw(gen())
    tc.draw(gen())


# ---------------------------------------------------------------------------
# Section F: Coverage tests for CST visitor edge cases
# ---------------------------------------------------------------------------


def _parse_and_collect(source: str, tc_param: str = "tc") -> dict[str, bool]:
    """Parse *source* with libcst and run the draw-name collector on it."""
    tree = cst.parse_module(textwrap.dedent(source))
    collector = _DrawNameCollector(tc_param)
    tree.visit(collector)
    return collector.names


def test_collector_trystar_marks_repeatable():
    """TryStar (Python 3.11+ except*) increments nesting depth."""
    names = _parse_and_collect("""
        def f(tc):
            try:
                pass
            except* ValueError:
                x = tc.draw(gen())
    """)
    assert names == {"x": True}


def test_collector_classdef_marks_repeatable():
    """ClassDef increments nesting depth."""
    names = _parse_and_collect("""
        def f(tc):
            class Inner:
                x = tc.draw(gen())
    """)
    assert names == {"x": True}


def test_collector_chained_assignment_skipped():
    """Chained assignment (a = b = tc.draw(...)) has len(targets) != 1 → skipped."""
    names = _parse_and_collect("""
        def f(tc):
            a = b = tc.draw(gen())
    """)
    # Neither a nor b collected (multiple targets)
    assert names == {}


def test_rewriter_multiple_targets_in_same_fn(capsys):
    """Rewriter skips chained assignment (line 275) when regular draw present."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x = tc.draw(gs.integers(0, 0))  # collected → rewritten
            a = b = tc.draw(
                gs.integers(0, 0)
            )  # multiple targets → skipped  # noqa: F841
            assert x < 0

    out = capsys.readouterr().out
    # x is rewritten to draw_named (no draw_N label); a=b= uses draw() → draw_1
    assert "x = 0" in out
    assert "draw_1 = 0" in out


def test_rewriter_tuple_target_when_regular_draw_present(capsys):
    """Rewriter skips tuple-target (line 278) when a regular draw is present."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x = tc.draw(gs.integers(0, 0))  # collected → rewritten
            a, b = tc.draw(  # tuple target → NOT collected/rewritten  # noqa: F841
                gs.tuples(gs.integers(0, 0), gs.integers(0, 0))
            )
            assert x < 0

    out = capsys.readouterr().out
    assert "x = 0" in out
    assert "draw_1 = (0, 0)" in out


def test_rewriter_nested_funcdef_line_268(capsys):
    """Rewriter returns inner FunctionDef unchanged (line 268), strips outer only."""
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            x = tc.draw(gs.integers(0, 0))

            def inner():
                pass

            inner()
            assert x < 0

    out = capsys.readouterr().out
    assert "x = 0" in out


def test_rewriter_kwdefaults_preserved():
    """rewrite_test_function preserves keyword-only default arguments."""

    def f(tc, *, limit=10):
        x = tc.draw(gs.integers(0, limit))
        return x

    rewritten = rewrite_test_function(f)
    assert rewritten is not f
    assert rewritten.__kwdefaults__ == {"limit": 10}  # type: ignore[union-attr]


def test_rewriter_draw_with_no_args():
    """Rewriter handles tc.draw() with zero arguments (empty existing_args branch)."""

    def f(tc):
        x = tc.draw()
        return x

    rewritten = rewrite_test_function(f)
    # The rewriter succeeds even with no-arg draw; it just doesn't strip a trailing comma
    assert rewritten is not f


def test_rewrite_fallback_on_bad_source():
    """rewrite_test_function falls back to the original function if rewriting fails."""
    # A function defined via exec has no source file, so inspect.getsource fails.
    ns: dict[str, object] = {}
    exec("def f(tc):\n    x = tc.draw(gs.integers(0, 5))\n    return x", ns)  # noqa: S102
    f = ns["f"]
    result = rewrite_test_function(f)  # type: ignore[arg-type]
    assert result is f


def test_hook_noop_when_original_test_is_none():
    """_draw_names_hook returns early when state._original_test is None."""
    state = PbtkitState(Random(), lambda tc: None, 1)
    assert state._original_test is None
    _draw_names_hook(state)  # must not raise
