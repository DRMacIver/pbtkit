"""Database support for pbtkit.

This module provides the Database protocol, DirectoryDB implementation,
serialization for the test database, and lifecycle hooks that load/save
test results. It is imported by the package's __init__.py to register
everything.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Sequence
from enum import IntEnum
from typing import (
    Any,
    Protocol,
)

from pbtkit.core import (
    PbtkitState,
    TestCase,
    setup_hook,
    teardown_hook,
)


class Database(Protocol):
    def __setitem__(self, key: str, value: bytes) -> None: ...

    def get(self, key: str) -> bytes | None: ...

    def __delitem__(self, key: str) -> None: ...


_DEFAULT_DATABASE_PATH = ".pbtkit-cache"


class DirectoryDB:
    """A very basic key/value store that just uses a file system
    directory to store values. You absolutely don't have to copy this
    and should feel free to use a more reasonable key/value store
    if you have easy access to one."""

    def __init__(self, directory: str):
        self.directory = directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

    def __to_file(self, key: str) -> str:
        return os.path.join(
            self.directory, hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        )

    def __setitem__(self, key: str, value: bytes) -> None:
        with open(self.__to_file(key), "wb") as o:
            o.write(value)

    def get(self, key: str) -> bytes | None:
        f = self.__to_file(key)
        if not os.path.exists(f):
            return None
        with open(f, "rb") as i:
            return i.read()

    def __delitem__(self, key: str) -> None:
        try:
            os.unlink(self.__to_file(key))
        except FileNotFoundError:
            raise KeyError()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class SerializationTag(IntEnum):
    INTEGER = 0
    BOOLEAN = 1
    BYTES = 2
    FLOAT = 3
    STRING = 4


def _read_fixed(data: bytes, offset: int, size: int) -> tuple[bytes, int]:
    """Read exactly size bytes from data at offset."""
    if offset + size > len(data):
        raise ValueError("truncated")
    return data[offset : offset + size], offset + size


def _serialize_value(v: Any) -> bytes:
    """Serialize a single choice value to tag + payload bytes."""
    # bool must be checked before int since bool is a subclass of int.
    match v:
        case bool():
            return bytes([SerializationTag.BOOLEAN, int(v)])
        case int():
            return bytes([SerializationTag.INTEGER]) + v.to_bytes(8, "big", signed=True)
        case float():  # needed_for("floats")
            return bytes([SerializationTag.FLOAT]) + struct.pack("!d", v)
        case bytes():  # needed_for("bytes")
            return bytes([SerializationTag.BYTES]) + len(v).to_bytes(4, "big") + v
        case str():  # needed_for("text")
            encoded = v.encode("utf-8")
            return (
                bytes([SerializationTag.STRING])
                + len(encoded).to_bytes(4, "big")
                + encoded
            )
        case _:
            assert False, f"unsupported type: {type(v)}"


def _deserialize_value(data: bytes, offset: int) -> tuple[Any, int]:
    """Deserialize a single choice value. Returns (value, new_offset).
    Raises ValueError/IndexError on malformed data."""
    tag = data[offset]
    offset += 1
    match tag:
        case SerializationTag.INTEGER:
            raw, offset = _read_fixed(data, offset, 8)
            return int.from_bytes(raw, "big", signed=True), offset
        case SerializationTag.BOOLEAN:
            raw, offset = _read_fixed(data, offset, 1)
            return bool(raw[0]), offset
        case SerializationTag.FLOAT:  # needed_for("floats")
            raw, offset = _read_fixed(data, offset, 8)
            return struct.unpack("!d", raw)[0], offset
        case SerializationTag.BYTES:  # needed_for("bytes")
            raw_len, offset = _read_fixed(data, offset, 4)
            length = int.from_bytes(raw_len, "big")
            raw, offset = _read_fixed(data, offset, length)
            return bytes(raw), offset
        case SerializationTag.STRING:  # needed_for("text")
            raw_len, offset = _read_fixed(data, offset, 4)
            length = int.from_bytes(raw_len, "big")
            raw, offset = _read_fixed(data, offset, length)
            return raw.decode("utf-8"), offset
        case _:
            raise ValueError(f"unknown tag: {tag}")


def _serialize_choices(values: Sequence[Any]) -> bytes:
    """Serialize a choice value sequence to bytes for database storage."""
    return b"".join(_serialize_value(v) for v in values)


def _deserialize_choices(data: bytes) -> list | None:
    """Deserialize a choice sequence from bytes. Returns None if
    the data is malformed (e.g. from an old format)."""
    values: list = []
    offset = 0
    try:
        while offset < len(data):
            value, offset = _deserialize_value(data, offset)
            values.append(value)
    except (IndexError, ValueError):
        return None
    return values


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


@setup_hook
def _database_setup(state: PbtkitState) -> None:
    """Load a previous failure from the database before running."""
    db = getattr(state.extras, "database", None)
    if db is None:
        db = DirectoryDB(_DEFAULT_DATABASE_PATH)
    state.extras.database = db
    test_name = state.extras.test_name
    previous_failure = db.get(test_name)
    if previous_failure is not None:
        values = _deserialize_choices(previous_failure)
        if values is not None:
            state.test_function(TestCase.for_choices(values))


@teardown_hook
def _database_teardown(state: PbtkitState) -> None:
    """Save or clear the result in the database after running."""
    db = state.extras.database
    test_name = state.extras.test_name
    if state.result is None:
        try:
            del db[test_name]
        except KeyError:
            pass
    else:
        db[test_name] = _serialize_choices([n.value for n in state.result])
