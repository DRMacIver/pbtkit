"""Snapshot tests for the failing-example reporting output of run_test.

Captures stdout from a representative range of failure scenarios so
the format is reviewable in PRs. Covers the multi_bug-enabled and
multi_bug-disabled paths.
"""

from __future__ import annotations

import re
from random import Random

import pytest

import pbtkit.generators as gs
from pbtkit import run_test


def _disable_multi_bug(monkeypatch):
    """Patch DISABLED_MODULES so feature_enabled('multi_bug') returns
    False inside the test. Skips if the feature module isn't available
    (compiled build)."""
    try:
        import pbtkit.features
    except ModuleNotFoundError:
        pytest.skip("pbtkit.features not available in compiled build")
    monkeypatch.setattr(
        pbtkit.features,
        "DISABLED_MODULES",
        pbtkit.features.DISABLED_MODULES | frozenset({"multi_bug"}),
    )


def _normalise(out: str) -> str:
    """Strip absolute paths and lineno noise so snapshots are stable
    across machines and source edits within the same file."""
    # The "Falsifying example" header embeds the origin's path:line.
    out = re.sub(
        r"\(([A-Za-z_]+) at [^:)]+:\d+\)",
        r"(\1 at <file>:NN)",
        out,
    )
    # Traceback frame lines: 'File "..../x.py", line NN, in ...'.
    out = re.sub(
        r'File "[^"]+/([^/"]+)", line \d+',
        r'File "<.../\1>", line NN',
        out,
    )
    # Replace exact pointer carets (e.g. "    ^^^^") with a stable
    # placeholder — column positions shift with cosmetic edits.
    out = re.sub(r"^\s*\^+\s*$", "    <caret>", out, flags=re.MULTILINE)
    return out


# ---------------------------------------------------------------------------
# Single-failure: with and without multi_bug
# ---------------------------------------------------------------------------


@pytest.mark.requires("multi_bug")
@pytest.mark.requires("draw_names")
def test_single_failure_multi_bug_on(snapshot, capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200, random=Random(0))
        def _(tc):
            n = tc.draw(gs.integers(0, 100))
            assert n < 50

    assert _normalise(capsys.readouterr().out) == snapshot


@pytest.mark.requires("draw_names")
def test_single_failure_multi_bug_off(snapshot, capsys, monkeypatch):
    _disable_multi_bug(monkeypatch)

    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=200, random=Random(0))
        def _(tc):
            n = tc.draw(gs.integers(0, 100))
            assert n < 50

    assert _normalise(capsys.readouterr().out) == snapshot


# ---------------------------------------------------------------------------
# Two distinct failures (different exception types and lines)
# ---------------------------------------------------------------------------


@pytest.mark.requires("multi_bug")
@pytest.mark.requires("draw_names")
def test_two_distinct_failures_multi_bug_on(snapshot, capsys):
    with pytest.raises(BaseException):

        @run_test(database={}, max_examples=200, random=Random(0))
        def _(tc):
            n = tc.draw(gs.integers(-100, 100))
            if n < -50:
                raise IndexError("low")
            if n > 50:
                raise ValueError("high")

    assert _normalise(capsys.readouterr().out) == snapshot


@pytest.mark.requires("draw_names")
def test_two_distinct_failures_multi_bug_off(snapshot, capsys, monkeypatch):
    """Without multi_bug only one failure is reported (the
    shortlex-smallest of the two)."""
    _disable_multi_bug(monkeypatch)

    with pytest.raises(BaseException):

        @run_test(database={}, max_examples=200, random=Random(0))
        def _(tc):
            n = tc.draw(gs.integers(-100, 100))
            if n < -50:
                raise IndexError("low")
            if n > 50:
                raise ValueError("high")

    assert _normalise(capsys.readouterr().out) == snapshot


# ---------------------------------------------------------------------------
# Three failures spread across three lines
# ---------------------------------------------------------------------------


@pytest.mark.requires("multi_bug")
@pytest.mark.requires("draw_names")
def test_three_failures_multi_bug_on(snapshot, capsys):
    with pytest.raises(BaseException):

        @run_test(database={}, max_examples=300, random=Random(0))
        def _(tc):
            n = tc.draw(gs.integers(-100, 100))
            if n < -50:
                raise IndexError("low")
            if n > 50:
                raise ValueError("high")
            if n == 0:
                raise KeyError("zero")

    assert _normalise(capsys.readouterr().out) == snapshot


# ---------------------------------------------------------------------------
# Failure with no exception (mark_status without a raise)
# ---------------------------------------------------------------------------


@pytest.mark.requires("multi_bug")
@pytest.mark.requires("draw_names")
def test_mark_status_failure_multi_bug_on(snapshot, capsys):
    from pbtkit.core import Status

    with pytest.raises(Exception):  # noqa: PT011

        @run_test(database={}, max_examples=10, random=Random(0))
        def _(tc):
            n = tc.draw(gs.integers(0, 100))
            if n >= 0:
                tc.mark_status(Status.INTERESTING)

    assert _normalise(capsys.readouterr().out) == snapshot
