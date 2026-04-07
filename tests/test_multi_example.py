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
from pbtkit.multi_example import MultipleFailures, _exception_key

pytestmark = pytest.mark.requires("multi_example")


def test_single_failure_does_not_raise_multiple():
    """A test with only one failure type should raise normally, not MultipleFailures."""
    with pytest.raises(AssertionError):

        @run_test(
            database={},
            random=Random(0),
            max_examples=200,
            quiet=True,
            report_multiple=True,
        )
        def _(tc):
            n = tc.choice(100)
            assert n < 50


def test_no_failure_is_unaffected():
    """A passing test should not be affected by multi_example."""

    @run_test(
        database={},
        random=Random(0),
        max_examples=50,
        report_multiple=True,
    )
    def _(tc):
        tc.choice(100)


def test_multiple_failures_different_exception_types():
    """Different exception types at different locations produce MultipleFailures."""
    with pytest.raises(MultipleFailures) as exc_info:

        @run_test(
            database={},
            random=Random(0),
            max_examples=200,
            quiet=True,
            report_multiple=True,
        )
        def _(tc):
            n = tc.choice(100)
            if n > 80:
                raise ValueError("too high")
            if n < 20:
                raise TypeError("too low")

    assert len(exc_info.value.errors) == 2
    type_names = {type(e).__name__ for e in exc_info.value.errors}
    assert type_names == {"ValueError", "TypeError"}


def test_multiple_failures_same_type_different_lines():
    """Same exception type raised at different lines produces MultipleFailures."""
    with pytest.raises(MultipleFailures) as exc_info:

        @run_test(
            database={},
            random=Random(0),
            max_examples=200,
            quiet=True,
            report_multiple=True,
        )
        def _(tc):
            n = tc.choice(100)
            if n > 80:
                assert False, "too high"
            if n < 20:
                assert False, "too low"

    assert len(exc_info.value.errors) == 2


def test_failures_are_shrunk():
    """Each distinct failure should be shrunk to its simplest form."""
    with pytest.raises(MultipleFailures) as exc_info:

        @run_test(
            database={},
            random=Random(0),
            max_examples=200,
            quiet=True,
            report_multiple=True,
        )
        def _(tc):
            n = tc.choice(1000)
            if n > 80:
                raise ValueError(f"too high: {n}")
            if n < 20:
                raise TypeError(f"too low: {n}")

    errors = exc_info.value.errors
    assert len(errors) == 2

    # Find each error type and check it was shrunk
    for e in errors:
        if isinstance(e, ValueError):
            # Smallest n > 80 is 81
            assert "81" in str(e)
        elif isinstance(e, TypeError):
            # Smallest n < 20 is 0
            assert "0" in str(e)


def test_without_flag_is_normal():
    """Without report_multiple=True, multi-example is not activated."""
    # Without report_multiple, only one failure is reported (the simplest one).
    # The simplest choice value is 0, which triggers TypeError.
    with pytest.raises(TypeError):

        @run_test(database={}, random=Random(0), max_examples=200, quiet=True)
        def _(tc):
            n = tc.choice(100)
            if n > 80:
                raise ValueError("too high")
            if n < 20:
                raise TypeError("too low")


def test_with_precondition():
    """Tests with preconditions (assume) work correctly with multi_example."""
    with pytest.raises(MultipleFailures) as exc_info:

        @run_test(
            database={},
            random=Random(0),
            max_examples=1000,
            quiet=True,
            report_multiple=True,
        )
        def _(tc):
            n = tc.choice(200)
            tc.assume(n % 2 == 0)
            if n > 100:
                raise ValueError("too high")
            if n < 50:
                raise TypeError("too low")

    assert len(exc_info.value.errors) == 2


def test_exception_key_without_traceback():
    """An exception with no traceback is classified by type only."""
    exc = ValueError("test")
    # Manually constructed exception has no traceback
    key = _exception_key(exc)
    assert key == ("ValueError",)


def test_multiple_failures_message_format():
    """MultipleFailures has a useful string representation."""
    exc = MultipleFailures([ValueError("a"), TypeError("b")])
    msg = str(exc)
    assert "Found 2 distinct failures" in msg
    assert "ValueError: a" in msg
    assert "TypeError: b" in msg


def test_multiple_failures_errors_attribute():
    """MultipleFailures.errors contains the original exceptions."""
    v = ValueError("a")
    t = TypeError("b")
    exc = MultipleFailures([v, t])
    assert exc.errors == [v, t]
