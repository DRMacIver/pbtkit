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
from collections.abc import Callable, Sequence
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


# Serialization registry keyed on Python value types (int, bool, etc.).
_SERIALIZERS: dict[type, tuple[int, Callable[[Any], bytes]]] = {}
_DESERIALIZERS: dict[int, Callable[[bytes, int], tuple[Any, int]]] = {}


def register_serializer(
    value_type: type,
    tag: int,
    serialize: Callable[[Any], bytes],
    deserialize: Callable[[bytes, int], tuple[Any, int]],
) -> None:
    """Register serialization for a Python value type.

    serialize(value) -> bytes
    deserialize(data, offset) -> (value, new_offset)
        Should raise IndexError or ValueError on truncated data."""
    _SERIALIZERS[value_type] = (tag, serialize)
    _DESERIALIZERS[tag] = deserialize


def _serialize_choices(values: Sequence[Any]) -> bytes:
    """Serialize a choice value sequence to bytes for database storage."""
    parts: list[bytes] = []
    for v in values:
        tag, serialize = _SERIALIZERS[type(v)]
        parts.append(bytes([tag]) + serialize(v))
    return b"".join(parts)


def _deserialize_choices(data: bytes) -> list | None:
    """Deserialize a choice sequence from bytes. Returns None if
    the data is malformed (e.g. from an old format)."""
    values: list = []
    i = 0
    try:
        while i < len(data):
            tag = data[i]
            i += 1
            if tag not in _DESERIALIZERS:
                return None
            value, i = _DESERIALIZERS[tag](data, i)
            values.append(value)
    except (IndexError, ValueError):
        return None
    return values


def _deserialize_fixed(size: int, convert: Callable[[bytes], Any]) -> Callable:
    """Helper to create a deserializer for fixed-size values."""

    def deserialize(data: bytes, offset: int) -> tuple[Any, int]:
        if offset + size > len(data):
            raise ValueError("truncated")
        return convert(data[offset : offset + size]), offset + size

    return deserialize


def _deserialize_length_prefixed(
    convert: Callable[[bytes], Any],
) -> Callable:
    """Helper to create a deserializer for length-prefixed values."""

    def deserialize(data: bytes, offset: int) -> tuple[Any, int]:
        if offset + 4 > len(data):
            raise ValueError("truncated")
        length = int.from_bytes(data[offset : offset + 4], "big")
        offset += 4
        if offset + length > len(data):
            raise ValueError("truncated")
        return convert(data[offset : offset + length]), offset + length

    return deserialize


# Register all value types. Note: bool must be registered separately
# from int because type(True) is bool, not int.
register_serializer(
    int,
    SerializationTag.INTEGER,
    lambda v: v.to_bytes(8, "big"),
    _deserialize_fixed(8, lambda b: int.from_bytes(b, "big")),
)

register_serializer(
    bool,
    SerializationTag.BOOLEAN,
    lambda v: bytes([int(v)]),
    _deserialize_fixed(1, lambda b: bool(b[0])),
)

register_serializer(
    float,
    SerializationTag.FLOAT,
    lambda v: struct.pack("!d", v),
    _deserialize_fixed(8, lambda b: struct.unpack("!d", b)[0]),
)

register_serializer(
    bytes,
    SerializationTag.BYTES,
    lambda v: len(v).to_bytes(4, "big") + v,
    _deserialize_length_prefixed(lambda b: bytes(b)),
)

register_serializer(
    str,
    SerializationTag.STRING,
    lambda v: (e := v.encode("utf-8"), len(e).to_bytes(4, "big") + e)[1],
    _deserialize_length_prefixed(lambda b: b.decode("utf-8")),
)


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
