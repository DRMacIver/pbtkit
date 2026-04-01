"""String shrink quality tests.

Ported from Hypothesis via hegel-rust/tests/test_shrink_quality/strings.rs.
"""

import pytest

import minithesis.generators as gs

from .conftest import minimal

pytestmark = [pytest.mark.requires("text"), pytest.mark.requires("collections")]


def test_minimize_string_to_empty():
    assert minimal(gs.text()) == ""


def test_minimize_longer_string():
    result = minimal(gs.text(max_size=50), lambda x: len(x) >= 10)
    assert result == "0" * 10


def test_minimize_longer_list_of_strings():
    assert minimal(gs.lists(gs.text()), lambda x: len(x) >= 10) == [""] * 10
