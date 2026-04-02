"""Findability regressions from pbtsmith comparison testing.

Each test here was discovered by generating random pbtsmith programs,
running them under Hypothesis as an oracle, and checking whether pbtkit
can also find the failure. When pbtkit fails to find something Hypothesis
can, it's added here as a concrete regression test.
"""

import pytest

from pbtkit import run_test
from pbtkit.generators import floats


class Failure(Exception):
    pass


@pytest.mark.requires("floats")
def test_non_negative_float_is_not_always_positive():
    """Draw a non-negative float, assert it's positive. The counterexample is 0.0.

    Hypothesis finds this easily but pbtkit's filtered float generation
    doesn't produce 0.0 within 1000 examples."""
    with pytest.raises(Failure):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            v0 = tc.draw(
                floats(allow_nan=False, allow_infinity=False).filter(lambda x: x >= 0.0)
            )
            if not (v0 > 0.0):
                raise Failure("v0 > 0.0")
