"""Bytes shrink quality tests."""

from random import Random

import pytest

import pbtkit.generators as gs
from pbtkit.bytes import BytesChoice
from pbtkit.core import PbtkitState as State
from pbtkit.core import Status

pytestmark = pytest.mark.requires("bytes")


@pytest.mark.requires("shrinking.advanced_bytes_passes")
def test_redistribute_bytes_between_pairs():
    """When two bytes values share a total length constraint, the shrinker
    should redistribute to make the first empty and the second full.
    Regression for shrink quality found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.binary(max_size=20))
        v1 = tc.draw(gs.binary(max_size=20))
        if len(v0) + len(v1) >= 20:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    bytes_values = [n.value for n in state.result if isinstance(n.kind, BytesChoice)]
    # First bytes should be empty, second should carry all the length.
    assert bytes_values[0] == b""


def test_redistribute_bytes_respects_max_size():
    """redistribute_bytes must skip transfers that exceed max_size."""

    def tf(tc):
        v0 = tc.draw(gs.binary(min_size=5, max_size=10))
        v1 = tc.draw(gs.binary(max_size=8))
        if len(v0) + len(v1) >= 15:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None


def test_bytes_sorts_when_order_matters():
    """Bytes shrinking should sort bytes when the test depends on order."""

    def tf(tc):
        v0 = tc.draw(gs.binary(min_size=3, max_size=3))
        # Only interesting if the bytes are NOT already sorted but contain 0x01.
        if b"\x01" in v0 and v0 != bytes(sorted(v0)):
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    # Sorting would make v0 sorted, which violates the condition.
    # So the swap should fail, covering the failure branch.
    assert state.result is not None


@pytest.mark.requires("shrinking.advanced_bytes_passes")
def test_bytes_length_redistribution():
    """When two bytes values share a total length constraint, the shrinker
    should redistribute to make the first as short as possible.
    Parallel test to test_string_length_redistribution — bytes and strings
    share the same shrinking infrastructure and often have the same bugs."""

    def tf(tc):
        v0 = tc.draw(gs.binary(max_size=20))
        v1 = tc.draw(gs.binary(max_size=20))
        if len(v0) + len(v1) >= 30:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    v0_len = len(state.result[0].value)
    # Optimal: v0 as short as possible (10 bytes, since v1 max is 20).
    assert v0_len == 10


def test_bytes_redistribution_moves_all():
    """When the second bytes value can absorb everything from the first,
    redistribution should move as much as possible. The min_size on v0
    prevents the value shrinker from emptying it directly, forcing
    redistribution to do the work."""

    def tf(tc):
        v0 = tc.draw(gs.binary(min_size=3, max_size=10))
        v1 = tc.draw(gs.binary(max_size=20))
        if len(v0) + len(v1) >= 10:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 100)
    state.run()
    assert state.result is not None
    # v0 can't go below min_size=3, so optimal is v0=3 bytes.
    assert len(state.result[0].value) == 3


@pytest.mark.requires("collections")
@pytest.mark.requires("text")
@pytest.mark.requires("shrinking.index_passes")
def test_bytes_increment_shortens_sequence():
    """Growing a bytes value by one byte can eliminate subsequent choices,
    producing a shorter (and thus simpler) overall sequence.
    Regression for shrink quality found by pbtsmith."""

    def tf(tc):
        v0 = tc.draw(gs.binary(max_size=20))
        v1 = tc.draw(
            gs.dictionaries(
                gs.integers(0, 0),
                gs.text(min_codepoint=32, max_codepoint=126, max_size=20),
                max_size=5,
            )
        )
        if len(v0) + len(v1) >= 20:
            tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    # Should shrink to just a 20-byte binary + empty dict (2 choices),
    # not 19 bytes + dict entry (5 choices).
    assert len(state.result) == 2


@pytest.mark.requires("collections")
@pytest.mark.requires("shrinking.index_passes")
def test_lower_and_bump_stale_kind_after_replace():
    """lower_and_bump must validate values against the CURRENT kind at
    position j, not the kind from before the replace. A replace can
    change types via value punning (e.g. BytesChoice → BooleanChoice).
    Regression for TypeError in sort_key found by pbtsmith."""

    @gs.composite
    def pair(tc):
        a = tc.draw(gs.booleans())
        b = tc.draw(gs.booleans())
        return (a, b)

    def tf(tc):
        v0 = tc.draw(gs.lists(gs.booleans(), max_size=10))
        tc.draw(gs.booleans())
        tc.draw(gs.binary(max_size=20))
        tc.draw(pair())
        tc.draw(pair())
        if len(v0) != 0:
            tc.mark_status(Status.INTERESTING)

    # Should not crash.
    state = State(Random(0), tf, 100)
    state.run()
