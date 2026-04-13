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
import tempfile
from collections.abc import Iterable, Iterator, Sequence
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
from pbtkit.features import feature_enabled


class Database(Protocol):
    """A multi-valued key/value store: each key may hold any number
    of distinct values. Mirrors Hypothesis's ExampleDatabase surface."""

    def save(self, key: str, value: bytes) -> None: ...

    def delete(self, key: str, value: bytes) -> None: ...

    def fetch(self, key: str) -> Iterable[bytes]: ...


_DEFAULT_DATABASE_PATH = ".pbtkit-cache"


class InMemoryDB:
    """Ephemeral in-process store. Useful for tests that want to opt
    out of on-disk persistence without creating a tmp directory."""

    def __init__(self) -> None:
        self._data: dict[str, set[bytes]] = {}

    def save(self, key: str, value: bytes) -> None:
        self._data.setdefault(key, set()).add(value)

    def delete(self, key: str, value: bytes) -> None:
        bucket = self._data.get(key)
        if bucket is None:
            return
        bucket.discard(value)
        if not bucket:
            del self._data[key]

    def fetch(self, key: str) -> Iterator[bytes]:
        return iter(sorted(self._data.get(key, ())))


class DirectoryDB:
    """Directory-backed store with one subdirectory per key and one
    file per value, named by a hash of the bytes.

    This mirrors Hypothesis's DirectoryBasedExampleDatabase layout,
    so multiple distinct failures per test can be persisted side by
    side. You don't have to copy this — swap in any store with the
    same save/delete/fetch interface."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _key_dir(self, key: str) -> str:
        return os.path.join(
            self.directory, hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        )

    def _value_path(self, key: str, value: bytes) -> str:
        return os.path.join(self._key_dir(key), hashlib.sha1(value).hexdigest()[:16])

    def save(self, key: str, value: bytes) -> None:
        key_dir = self._key_dir(key)
        os.makedirs(key_dir, exist_ok=True)
        path = self._value_path(key, value)
        if os.path.exists(path):
            return
        # Atomic write: temp + rename, so a partial write never shows
        # up as a legitimate value.
        fd, tmp = tempfile.mkstemp(dir=key_dir)
        try:
            os.write(fd, value)
        finally:
            os.close(fd)
        os.rename(tmp, path)

    def delete(self, key: str, value: bytes) -> None:
        try:
            os.unlink(self._value_path(key, value))
        except OSError:
            return
        try:
            os.rmdir(self._key_dir(key))
        except OSError:
            pass

    def fetch(self, key: str) -> Iterator[bytes]:
        key_dir = self._key_dir(key)
        try:
            names = sorted(os.listdir(key_dir))
        except OSError:
            return
        for name in names:
            try:
                with open(os.path.join(key_dir, name), "rb") as f:
                    yield f.read()
            except OSError:
                continue


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


def _current_interesting_values(state: PbtkitState) -> set[bytes]:
    """Return the set of serialised choice sequences we want the
    database to hold after this run.

    With multi_bug enabled, that's every per-origin best example.
    Without it, it's just ``state.result`` (or the empty set if
    nothing failed)."""
    if feature_enabled("multi_bug"):
        examples = getattr(state.extras, "interesting_examples", None) or {}
        return {
            _serialize_choices([n.value for n in tc.nodes]) for tc in examples.values()
        }
    else:  # pragma: no cover
        # Dead at runtime when multi_bug is enabled (the gate above
        # always fires). Exercised in the compiled --disable=multi_bug
        # variant via ``just test-compiled``.
        if state.result is None:
            return set()
        return {_serialize_choices([n.value for n in state.result])}


@setup_hook
def _database_setup(state: PbtkitState) -> None:
    """Replay every previously-saved failure for this test. Each one
    is run through ``state.test_function``, so the engine (and any
    test_function_hooks such as multi_bug's _record_origin) see the
    stored failures as fresh results."""
    db = getattr(state.extras, "database", None)
    if db is None:
        db = DirectoryDB(_DEFAULT_DATABASE_PATH)
    elif isinstance(db, dict):
        # ``database={}`` is the standard test idiom for "no on-disk
        # persistence" — treat it as a fresh in-memory store. Any
        # contents the dict happens to hold are ignored (it didn't
        # have a key/value layout in the first place).
        db = InMemoryDB()
    state.extras.database = db
    test_name = state.extras.test_name
    for raw in db.fetch(test_name):
        values = _deserialize_choices(raw)
        if values is not None:
            state.test_function(TestCase.for_choices(values))


@teardown_hook
def _database_teardown(state: PbtkitState) -> None:
    """Sync the database's stored set of failures for this test with
    the state's current set — additions saved, stale entries
    deleted."""
    db = state.extras.database
    test_name = state.extras.test_name
    existing = set(db.fetch(test_name))
    target = _current_interesting_values(state)
    for v in existing - target:
        db.delete(test_name, v)
    for v in target - existing:
        db.save(test_name, v)
