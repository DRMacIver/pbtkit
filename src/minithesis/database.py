"""Database support for minithesis.

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

from minithesis.core import (
    MinithesisState,
    TestCase,
    setup_hook,
    teardown_hook,
)


class Database(Protocol):
    def __setitem__(self, key: str, value: bytes) -> None: ...

    def get(self, key: str) -> bytes | None: ...

    def __delitem__(self, key: str) -> None: ...


_DEFAULT_DATABASE_PATH = ".minithesis-cache"


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


def _read_length_prefixed(data: bytes, offset: int) -> tuple[bytes, int]:
    """Read a 4-byte-length-prefixed blob from data at offset."""
    raw, offset = _read_fixed(data, offset, 4)
    length = int.from_bytes(raw, "big")
    return _read_fixed(data, offset, length)


# Type-specific serializers/deserializers. Each pair handles one
# SerializationTag. The compiler strips needed_for-decorated
# functions when their feature is disabled.


def _serialize_int(v: int) -> bytes:
    return bytes([SerializationTag.INTEGER]) + v.to_bytes(8, "big", signed=True)


def _deserialize_int(data: bytes, offset: int) -> tuple[int, int]:
    raw, offset = _read_fixed(data, offset, 8)
    return int.from_bytes(raw, "big", signed=True), offset


def _serialize_bool(v: bool) -> bytes:
    return bytes([SerializationTag.BOOLEAN, int(v)])


def _deserialize_bool(data: bytes, offset: int) -> tuple[bool, int]:
    raw, offset = _read_fixed(data, offset, 1)
    return bool(raw[0]), offset


def _serialize_float(v: float) -> bytes:
    return bytes([SerializationTag.FLOAT]) + struct.pack("!d", v)


def _deserialize_float(data: bytes, offset: int) -> tuple[float, int]:
    raw, offset = _read_fixed(data, offset, 8)
    return struct.unpack("!d", raw)[0], offset


def _serialize_bytes(v: bytes) -> bytes:
    return bytes([SerializationTag.BYTES]) + len(v).to_bytes(4, "big") + v


def _deserialize_bytes(data: bytes, offset: int) -> tuple[bytes, int]:
    raw, offset = _read_length_prefixed(data, offset)
    return bytes(raw), offset


def _serialize_str(v: str) -> bytes:
    encoded = v.encode("utf-8")
    return bytes([SerializationTag.STRING]) + len(encoded).to_bytes(4, "big") + encoded


def _deserialize_str(data: bytes, offset: int) -> tuple[str, int]:
    raw, offset = _read_length_prefixed(data, offset)
    return raw.decode("utf-8"), offset


def _serialize_value(v: Any) -> bytes:
    """Serialize a single choice value to tag + payload bytes."""
    # bool must be checked before int since bool is a subclass of int.
    if isinstance(v, bool):
        return _serialize_bool(v)
    if isinstance(v, int):
        return _serialize_int(v)
    if isinstance(v, float):
        return _serialize_float(v)
    if isinstance(v, bytes):
        return _serialize_bytes(v)
    assert isinstance(v, str), f"unsupported type: {type(v)}"
    return _serialize_str(v)


def _deserialize_value(data: bytes, offset: int) -> tuple[Any, int]:
    """Deserialize a single choice value. Returns (value, new_offset).
    Raises ValueError/IndexError on malformed data."""
    tag = data[offset]
    offset += 1
    if tag == SerializationTag.INTEGER:
        return _deserialize_int(data, offset)
    if tag == SerializationTag.BOOLEAN:
        return _deserialize_bool(data, offset)
    if tag == SerializationTag.FLOAT:
        return _deserialize_float(data, offset)
    if tag == SerializationTag.BYTES:
        return _deserialize_bytes(data, offset)
    if tag == SerializationTag.STRING:
        return _deserialize_str(data, offset)
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
def _database_setup(state: MinithesisState) -> None:
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
def _database_teardown(state: MinithesisState) -> None:
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
