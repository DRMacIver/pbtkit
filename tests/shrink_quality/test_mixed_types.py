"""Mixed type shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/mixed_types.rs.
"""

import pytest

from minithesis.generators import booleans, composite, integers, lists, one_of, text

from .conftest import minimal

pytestmark = [pytest.mark.requires("text"), pytest.mark.requires("collections")]


# one_of with same-type generators
def test_minimize_one_of_integers():
    for _ in range(10):
        result = minimal(
            one_of(integers(-(2**63), 2**63 - 1), integers(100, 200)),
        )
        assert result == 0


# Mixed types via tagged tuples
def test_minimize_one_of_mixed():
    for _ in range(10):
        result = minimal(
            one_of(
                integers(-(2**63), 2**63 - 1).map(lambda x: ("int", x)),
                text().map(lambda x: ("text", x)),
                booleans().map(lambda x: ("bool", x)),
            ),
        )
        assert result in [("int", 0), ("text", ""), ("bool", False)]


# Mixed list
def test_minimize_mixed_list():
    result = minimal(
        lists(
            one_of(
                integers(-(2**63), 2**63 - 1).map(lambda x: ("int", x)),
                text().map(lambda x: ("text", x)),
            )
        ),
        lambda x: len(x) >= 10,
    )
    assert len(result) == 10
    allowed = {("int", 0), ("text", "")}
    for item in result:
        assert item in allowed


# Mixed flatmap
@composite
def bool_or_text_via_flatmap(tc):
    b = tc.any(booleans())
    if b:
        return ("bool", tc.any(booleans()))
    else:
        return ("text", tc.any(text()))


def test_mixed_list_flatmap():
    result = minimal(
        lists(bool_or_text_via_flatmap()),
        lambda ls: (
            sum(1 for x in ls if x[0] == "bool") >= 3
            and sum(1 for x in ls if x[0] == "text") >= 3
        ),
    )
    assert len(result) == 6
    assert set(result) == {("bool", False), ("text", "")}


# one_of shrinks towards earlier branches
def test_one_of_slip():
    result = minimal(
        one_of(integers(101, 200), integers(0, 100)),
    )
    assert result == 101
