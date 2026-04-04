"""Findability regressions from pbtsmith comparison testing.

Each test here was discovered by generating random pbtsmith programs,
running them under Hypothesis as an oracle, and checking whether pbtkit
can also find the failure. When pbtkit fails to find something Hypothesis
can, it's added here as a concrete regression test.
"""

import pytest

from pbtkit import run_test
from pbtkit.generators import binary, booleans, floats, integers, just, lists, tuples


class Failure(Exception):
    pass


@pytest.mark.requires("edge_case_boosting")
def test_zero_from_wide_integer_range():
    """Draw an integer from [0, 8191], assert it's positive. Counterexample: 0.

    With uniform generation, P(0 in 1000 draws) ≈ 11.5%. Hypothesis finds this
    easily because it special-cases boundary values."""
    with pytest.raises(Failure):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            v0 = tc.draw(integers(0, 8191))
            if not (v0 > 0):
                raise Failure("v0 > 0")


@pytest.mark.requires("collections")
@pytest.mark.requires("span_mutation")
def test_duplicate_tuples_in_list():
    """Find a list of (int, int) tuples containing a duplicate.

    Range is (0, 184) × (0, 184) = ~34K possible tuples. With ≤10 elements
    per list, a birthday collision is ~0.15% per test case. Hypothesis finds
    this through its duplication pass; pbtkit must get lucky."""
    with pytest.raises(Failure):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            v0 = tc.draw(lists(tuples(integers(0, 184), integers(0, 184)), max_size=10))
            if not (len(v0) == len(set(v0))):
                raise Failure("duplicate tuple")


@pytest.mark.requires("floats")
@pytest.mark.requires("edge_case_boosting")
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


@pytest.mark.requires("bytes")
@pytest.mark.requires("collections")
@pytest.mark.requires("edge_case_boosting")
@pytest.mark.xfail(reason="conjunction of empty bytes + zero is hard to find reliably")
def test_empty_bytes_with_wide_dependent_range():
    """Like test_empty_bytes_with_dependent_condition but with a wider
    dependent range (integers(0, 39)), making the conjunction harder."""
    with pytest.raises(Failure):

        @run_test(database={}, max_examples=5000)
        def _(tc):
            tc.draw(just(False))
            v1 = tc.draw(binary(max_size=20))
            tc.draw(binary(max_size=20))
            v3 = tc.draw(booleans().map(lambda x: int(x)))
            v4 = tc.draw(integers(v3, v3 + 39))
            if len(v1) > 0:
                tc.draw(booleans())
            else:
                if not (v3 + v4 > 0):
                    raise Failure("v3 + v4 > 0")


@pytest.mark.requires("bytes")
@pytest.mark.requires("collections")
@pytest.mark.requires("edge_case_boosting")
def test_empty_bytes_with_dependent_condition():
    """Find failure requiring empty bytes AND zero from mapped booleans.

    Needs binary(max_size=20) to produce b"" so the else branch runs,
    then booleans().map(int) to produce 0 and integers(0, 2) to produce 0."""
    with pytest.raises(Failure):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            tc.draw(just(False))
            v1 = tc.draw(binary(max_size=20))
            tc.draw(binary(max_size=20))
            v3 = tc.draw(booleans().map(lambda x: int(x)))
            v4 = tc.draw(integers(v3, v3 + 2))
            if len(v1) > 0:
                tc.draw(booleans())
            else:
                if not (v3 + v4 > 0):
                    raise Failure("v3 + v4 > 0")
