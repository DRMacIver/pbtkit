# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from minithesis import DirectoryDB, run_test

pytestmark = pytest.mark.requires("text")
from minithesis.core import TestCase as TC
from minithesis.database import SerializationTag
from minithesis.generators import text
from minithesis.text import StringChoice


def test_text_basic():
    @run_test(database={})
    def _(tc):
        s = tc.any(text(min_size=1, max_size=5))
        assert 1 <= len(s) <= 5


def test_text_ascii():
    @run_test(database={})
    def _(tc):
        s = tc.any(text(min_codepoint=32, max_codepoint=126))
        assert all(32 <= ord(c) <= 126 for c in s)


def test_text_shrinks_to_short(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={})
        def _(tc):
            s = tc.any(text(min_codepoint=ord("a"), max_codepoint=ord("z")))
            assert len(s) < 1

    captured = capsys.readouterr()
    assert "text(" in captured.out
    # Should shrink to "a" (shortest, simplest character)
    assert captured.out.strip().endswith(": a")


def test_text_shrinks_characters(capsys):
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            s = tc.any(
                text(
                    min_codepoint=ord("a"),
                    max_codepoint=ord("z"),
                    min_size=1,
                    max_size=5,
                )
            )
            assert "z" not in s

    captured = capsys.readouterr()
    assert captured.out.strip().endswith(": z")


def test_text_no_surrogates():
    @run_test(database={}, max_examples=200)
    def _(tc):
        s = tc.any(text(min_codepoint=0xD700, max_codepoint=0xE000))
        for c in s:
            assert not (0xD800 <= ord(c) <= 0xDFFF)


@pytest.mark.requires("database")
def test_text_database_round_trip(tmpdir):
    db = DirectoryDB(tmpdir)
    count = 0

    def run():
        with pytest.raises(AssertionError):

            @run_test(database=db)
            def _(test_case):
                nonlocal count
                count += 1
                s = test_case.any(text(min_size=1, max_size=5))
                assert len(s) < 1

    run()
    prev_count = count

    run()
    assert count == prev_count + 2


def test_draw_string_invalid_range():
    tc = TC.for_choices([])
    with pytest.raises(ValueError):
        tc.draw_string(min_codepoint=200, max_codepoint=100)


def test_string_single_codepoint_unit():
    """StringChoice with a single codepoint returns sensible unit."""
    # Single codepoint '0' (the simplest in key order), variable length.
    kind = StringChoice(48, 48, 0, 5)
    assert kind.unit == "0"  # one char longer
    assert kind.simplest == ""

    # Single codepoint '0', fixed length → unit == simplest (degenerate).
    kind2 = StringChoice(48, 48, 2, 2)
    assert kind2.unit == kind2.simplest

    # Single codepoint 'A' — second_cp is 'A' itself (not '0'), handled
    # by the general path.
    kind3 = StringChoice(65, 65, 0, 5)
    assert kind3.unit == "A"


def test_string_validate():
    """StringChoice.validate rejects non-strings and wrong sizes."""
    kind = StringChoice(32, 126, 1, 5)
    assert kind.validate("abc")
    assert not kind.validate(123)  # type: ignore[arg-type]
    assert not kind.validate("")  # too short
    assert not kind.validate("abcdef")  # too long


def test_string_from_index_out_of_range():
    """from_index past max_index returns None."""
    sc = StringChoice(32, 126, 0, 2)
    assert sc.from_index(sc.max_index + 1) is None


def test_text_unicode_shrinks(capsys):
    """Strings with high codepoints shrink toward the lowest in range."""
    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=1000)
        def _(tc):
            s = tc.any(
                text(min_codepoint=128, max_codepoint=256, min_size=1, max_size=3)
            )
            # Must contain a high character, so the shrinker has to
            # search through codepoints >= 128.
            assert all(ord(c) < 200 for c in s)


@pytest.mark.requires("database")
def test_truncated_string_database_entry():
    """Truncated string entries in database are handled gracefully."""
    for data in [
        bytes([SerializationTag.STRING, 0x00, 0x00]),  # truncated length header
        bytes(
            [SerializationTag.STRING, 0x00, 0x00, 0x00, 0x05, 0x61]
        ),  # length 5 but only 1 byte
    ]:
        db = {"_": data}

        @run_test(database=db, max_examples=1)
        def _(test_case):
            pass
