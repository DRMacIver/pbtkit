"""Tests for the multi_bug feature.

Covers InterestingOrigin, per-origin shrinking, generation continuation,
database round-trip of multiple per-origin examples, and the multi-bug
report path. Each test is gated with ``@pytest.mark.requires("multi_bug")``
so the suite still passes under ``PBTKIT_DISABLED=multi_bug``.
"""

from __future__ import annotations

from random import Random

import pytest

from pbtkit import DirectoryDB, run_test
from pbtkit.core import PbtkitState, Status
from pbtkit.core import TestCase as TC

multi_bug = pytest.importorskip("pbtkit.multi_bug")
pytestmark = pytest.mark.requires("multi_bug")


InterestingOrigin = multi_bug.InterestingOrigin


# ---------------------------------------------------------------------------
# InterestingOrigin
# ---------------------------------------------------------------------------


def _origin_for(exc: BaseException) -> InterestingOrigin:
    return InterestingOrigin.from_exception(exc)


def _capture(fn) -> InterestingOrigin:
    """Run *fn* expecting it to raise; return the origin of that exception."""
    try:
        fn()
    except BaseException as e:
        return _origin_for(e)
    raise AssertionError("expected fn to raise")


def test_origin_equality_for_same_failure():
    """Two raises of the same type from the same line yield equal origins."""

    def raiser():
        raise ValueError("x")

    a = _capture(raiser)
    b = _capture(raiser)
    assert a == b
    # And equal origins hash the same so they bucket in dicts.
    assert {a: 1, b: 2} == {a: 2}


def test_origin_distinguishes_exception_type():
    a = _capture(lambda: (_ for _ in ()).throw(ValueError()))
    b = _capture(lambda: (_ for _ in ()).throw(IndexError()))
    assert a != b
    assert a.exc_type is ValueError
    assert b.exc_type is IndexError


def test_origin_distinguishes_lineno():
    def raise_a():
        raise ValueError()  # line A

    def raise_b():
        raise ValueError()  # line B (different lineno)

    a = _capture(raise_a)
    b = _capture(raise_b)
    assert a != b
    assert a.exc_type is ValueError
    assert b.exc_type is ValueError
    assert a.lineno != b.lineno


def test_origin_handles_no_traceback():
    """An exception with no __traceback__ still yields a valid origin
    (filename and lineno are None)."""
    exc = ValueError()
    # exc.__traceback__ is None when never raised.
    o = _origin_for(exc)
    assert o.exc_type is ValueError
    assert o.filename is None
    assert o.lineno is None


# ---------------------------------------------------------------------------
# Per-origin shrinking via run_test
# ---------------------------------------------------------------------------


def test_two_distinct_failures_are_each_recorded_and_shrunk():
    """A test that fails two ways (different lines) records both
    origins, and per-origin shrinking minimises each."""

    captured: list = []

    def runner():
        with pytest.raises(BaseException):

            @run_test(database={}, max_examples=200, random=Random(0))
            def _(tc):
                n = tc.draw_integer(-100, 100)
                if n < -50:
                    raise IndexError("low")  # one line
                if n > 50:
                    raise ValueError("high")  # different line
                # otherwise pass

    runner()

    # We can't directly inspect the state from the @run_test wrapper, so
    # we re-drive a PbtkitState manually to verify the multi-bug machinery.
    def tf(tc):
        n = tc.draw_integer(-100, 100)
        if n < -50:
            raise IndexError("low")
        if n > 50:
            raise ValueError("high")

    def shim(tc):
        try:
            tf(tc)
        except Exception as exc:
            if tc.status is not None:
                raise
            tc.mark_status(
                Status.INTERESTING,
                interesting_origin=InterestingOrigin.from_exception(exc),
            )

    state = PbtkitState(Random(0), shim, max_examples=200)
    state.run()
    examples = state.extras.interesting_examples
    captured.extend(examples)
    # Both origins discovered.
    types = {o.exc_type for o in examples}
    assert IndexError in types
    assert ValueError in types
    # Each origin shrunk to its respective minimum.
    by_type = {o.exc_type: tc for o, tc in examples.items()}
    low = by_type[IndexError].nodes[0].value
    high = by_type[ValueError].nodes[0].value
    assert low == -51
    assert high == 51


def test_per_origin_predicate_rejects_other_origins():
    """Direct unit test: shrink_per_origin's per-origin predicate
    refuses to transition between origins."""

    def tf(tc):
        n = tc.draw_integer(-100, 100)
        if n < -50:
            raise IndexError("low")
        if n > 50:
            raise ValueError("high")

    def shim(tc):
        try:
            tf(tc)
        except Exception as exc:
            if tc.status is not None:
                raise
            tc.mark_status(
                Status.INTERESTING,
                interesting_origin=InterestingOrigin.from_exception(exc),
            )

    state = PbtkitState(Random(0), shim, max_examples=200)
    state.run()
    # After shrinking, the IndexError example value is -51 (smallest |n|).
    by_type = {o.exc_type: tc for o, tc in state.extras.interesting_examples.items()}
    assert by_type[IndexError].nodes[0].value == -51
    # And the ValueError example is +51 — NOT collapsed into IndexError's
    # bucket even though +51 is shortlex-smaller in the global sense.
    assert by_type[ValueError].nodes[0].value == 51


# ---------------------------------------------------------------------------
# Generation continuation
# ---------------------------------------------------------------------------


def test_generation_stops_quickly_when_only_one_origin():
    """Single-origin failures stop early — `last_new_origin_at * 2 + 1`
    bounds generation. The total call budget is well below the
    `max_examples * 10` hard cap, demonstrating that multi_bug doesn't
    just exhaust the budget."""

    def shim(tc):
        try:
            n = tc.draw_integer(0, 1000)
            if n >= 0:
                raise ValueError()
        except Exception as exc:
            if tc.status is not None:
                raise
            tc.mark_status(
                Status.INTERESTING,
                interesting_origin=InterestingOrigin.from_exception(exc),
            )

    state = PbtkitState(Random(0), shim, max_examples=100)
    state.run()
    # state.calls includes shrinking too; the meaningful signal is that
    # we're nowhere near max_examples * 10 = 1000.
    assert state.calls < 200


def test_should_keep_generating_returns_true_with_no_bugs():
    """Until any bug is found, generation continues normally."""
    state = PbtkitState(Random(0), lambda tc: None, max_examples=10)
    # Force the multi_bug branch by initializing extras as if it had run.
    multi_bug._ensure_state(state)
    assert multi_bug.multi_bug_should_keep_generating(state) is True


def test_should_keep_generating_caps_on_total_calls():
    """The max_examples * 10 hard cap fires even mid-discovery."""
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    multi_bug._ensure_state(state)
    state.calls = 100  # >= max_examples * 10
    assert multi_bug.multi_bug_should_keep_generating(state) is False


def test_should_keep_generating_caps_on_valid_examples():
    """The max_examples cap on valid cases also short-circuits."""
    state = PbtkitState(Random(0), lambda tc: None, max_examples=5)
    multi_bug._ensure_state(state)
    state.valid_test_cases = 5
    assert multi_bug.multi_bug_should_keep_generating(state) is False


def test_should_keep_generating_short_circuits_trivial():
    state = PbtkitState(Random(0), lambda tc: None, max_examples=10)
    multi_bug._ensure_state(state)
    state.test_is_trivial = True
    assert multi_bug.multi_bug_should_keep_generating(state) is False


# ---------------------------------------------------------------------------
# state.shrink early-return / fallback
# ---------------------------------------------------------------------------


def test_shrink_no_result_is_a_noop():
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    # state.result is None: shrink returns immediately, no error.
    state.shrink()
    assert state.result is None


def test_record_origin_skips_non_interesting_test_cases():
    """The hook returns early if status isn't INTERESTING."""

    def tf(tc):
        tc.draw_integer(0, 5)  # status will be VALID

    state = PbtkitState(Random(0), tf, max_examples=1)
    state.run()
    # Generation runs but never marks INTERESTING; examples stays empty.
    examples = getattr(state.extras, "interesting_examples", None) or {}
    assert examples == {}


def test_record_origin_handles_mark_status_without_exception():
    """Tests that mark INTERESTING without raising an exception use a
    synthetic origin (BaseException, None, None)."""

    def tf(tc):
        tc.draw_integer(0, 5)
        tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, max_examples=10)
    state.run()
    examples = state.extras.interesting_examples
    assert len(examples) == 1
    origin = next(iter(examples))
    assert origin.exc_type is BaseException
    assert origin.filename is None
    assert origin.lineno is None


# ---------------------------------------------------------------------------
# Database round-trip
# ---------------------------------------------------------------------------


@pytest.mark.requires("database")
def test_database_round_trip_persists_per_origin_examples(tmpdir):
    """Two distinct origins are persisted; reload replays both."""
    db = DirectoryDB(str(tmpdir))

    def runner():
        with pytest.raises(BaseException):

            @run_test(database=db, max_examples=200, random=Random(0))
            def _(tc):
                n = tc.draw_integer(-100, 100)
                if n < -50:
                    raise IndexError("low")
                if n > 50:
                    raise ValueError("high")

    runner()
    # First run: db should have a .multi entry with both origins.
    state1_files = list(tmpdir.listdir())
    assert len(state1_files) == 1

    # Reload: same logic. Both examples should be replayed and re-shrunk.
    runner()
    state2_files = list(tmpdir.listdir())
    assert len(state2_files) == 1


@pytest.mark.requires("database")
def test_save_to_db_clears_when_no_examples(tmpdir):
    """save_to_db deletes the .multi key when interesting_examples is empty."""
    db = DirectoryDB(str(tmpdir))
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    state.extras.test_name = "T"
    state.extras.interesting_examples = {}
    # Pre-populate the key so the delete path is exercised.
    db["T.multi"] = b"junk"
    multi_bug.save_to_db(state, db, "T")
    assert db.get("T.multi") is None


@pytest.mark.requires("database")
def test_save_to_db_clears_when_key_already_absent(tmpdir):
    """The KeyError on stale-delete is swallowed."""
    db = DirectoryDB(str(tmpdir))
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    state.extras.test_name = "T"
    state.extras.interesting_examples = {}
    multi_bug.save_to_db(state, db, "T")  # key doesn't exist; no error
    assert db.get("T.multi") is None


@pytest.mark.requires("database")
def test_load_from_db_skips_missing_key(tmpdir):
    """No-op when the .multi key isn't in the database."""
    db = DirectoryDB(str(tmpdir))
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    multi_bug.load_from_db(state, db, "absent")
    assert state.calls == 0


@pytest.mark.requires("database")
def test_load_from_db_skips_malformed_payload(tmpdir):
    """If the payload doesn't deserialise, no replay happens."""
    db = DirectoryDB(str(tmpdir))
    db["T.multi"] = b"\x00\x00\x00\x01garbage"  # claims 1 entry, malformed
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    multi_bug.load_from_db(state, db, "T")
    assert state.calls == 0


@pytest.mark.requires("database")
def test_serialize_round_trip():
    """_serialize_multi / _deserialize_multi are inverses."""
    sequences = [[1, True], [-7, False]]
    raw = multi_bug._serialize_multi(sequences)
    out = multi_bug._deserialize_multi(raw)
    assert out == sequences


@pytest.mark.requires("database")
def test_deserialize_too_short_returns_none():
    assert multi_bug._deserialize_multi(b"") is None
    assert multi_bug._deserialize_multi(b"\x00\x00") is None


@pytest.mark.requires("database")
def test_deserialize_truncated_size_returns_none():
    """Header claims 1 entry but the size prefix is truncated."""
    assert multi_bug._deserialize_multi(b"\x00\x00\x00\x01\x00") is None


@pytest.mark.requires("database")
def test_deserialize_truncated_body_returns_none():
    """Header claims 1 entry of size 100 but body is shorter."""
    assert multi_bug._deserialize_multi(b"\x00\x00\x00\x01\x00\x00\x00\x64ab") is None


@pytest.mark.requires("database")
def test_deserialize_invalid_inner_payload_returns_none():
    """Header is well-formed but the inner choice sequence has an
    unknown serialization tag."""
    # 1 entry, size 1, body = 0xFF (not a valid tag).
    assert multi_bug._deserialize_multi(b"\x00\x00\x00\x01\x00\x00\x00\x01\xff") is None


@pytest.mark.requires("database")
def test_deserialize_truncated_value_payload_returns_none():
    """Inner choice sequence has a valid tag but truncated body —
    exercises _read_fixed's ValueError('truncated') path inside
    _deserialize_choices."""
    # 1 entry, size 4, body = INTEGER tag (0x00) + only 3 bytes (need 8).
    assert (
        multi_bug._deserialize_multi(
            b"\x00\x00\x00\x01\x00\x00\x00\x04\x00\x01\x02\x03"
        )
        is None
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_multi_bug_report_prints_each_origin():
    """Two origins ⇒ two prints. Drives the report directly with a
    constructed state to avoid depending on RNG luck during generation."""
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    tc1 = TC.for_choices([1])
    state.test_function(tc1)
    tc1.status = Status.INTERESTING  # type: ignore[assignment]
    tc2 = TC.for_choices([2])
    state.test_function(tc2)
    tc2.status = Status.INTERESTING  # type: ignore[assignment]
    o1 = InterestingOrigin(IndexError, "f.py", 1)
    o2 = InterestingOrigin(ValueError, "f.py", 2)
    state.extras.interesting_examples = {o1: tc1, o2: tc2}
    printed: list = []
    state._print_function = lambda t: printed.append(t)
    multi_bug.multi_bug_report(state, quiet=True)
    assert len(printed) == 2


def test_multi_bug_report_falls_back_when_examples_empty():
    """If interesting_examples is empty for any reason but state.result
    is set, the report falls back to printing state.result once."""

    def tf(tc):
        tc.draw_integer(0, 10)

    state = PbtkitState(Random(0), tf, max_examples=1)
    state.extras.interesting_examples = {}
    tc = TC.for_choices([0])
    state.test_function(tc)
    assert tc.nodes  # the draw_integer should have produced one node
    state.result = tc.nodes
    printed: list = []
    state._print_function = lambda t: printed.append(t)
    multi_bug.multi_bug_report(state, quiet=True)
    assert len(printed) == 1


# ---------------------------------------------------------------------------
# shrink_per_origin idempotency / no-op paths
# ---------------------------------------------------------------------------


def test_shrink_per_origin_noop_when_no_examples():
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    multi_bug._ensure_state(state)
    multi_bug.shrink_per_origin(state)  # no-op, no crash
    assert state.result is None


def test_shrink_per_origin_reshrinks_incidentally_improved_origin(monkeypatch):
    """When shrinking origin B incidentally finds a smaller example
    for origin A (via the test_function_hook updating ``examples``
    mid-shrink), A is evicted from ``shrunk`` and re-shrunk."""

    def tf(tc):
        # Drive draws so the resulting nodes have meaningful sort_keys.
        for _ in range(3):
            tc.draw_integer(0, 100)

    state = PbtkitState(Random(0), tf, max_examples=1)
    multi_bug._ensure_state(state)

    o_a = InterestingOrigin(IndexError, "f.py", 1)
    o_b = InterestingOrigin(ValueError, "f.py", 2)

    # Construct TCs with distinct sort_keys: A larger than B, and a
    # smaller "discoverable" alternative for A.
    big_a = TC.for_choices([100, 100, 100])
    state.test_function(big_a)
    big_a.status = Status.INTERESTING  # type: ignore[assignment]
    big_b = TC.for_choices([5])
    state.test_function(big_b)
    big_b.status = Status.INTERESTING  # type: ignore[assignment]
    small_a = TC.for_choices([1])
    state.test_function(small_a)
    small_a.status = Status.INTERESTING  # type: ignore[assignment]

    state.extras.interesting_examples = {o_a: big_a, o_b: big_b}

    invocations: list = []

    class FakeShrinker:
        def __init__(self, state, initial, is_interesting):
            self.current = initial
            invocations.append(initial)

        def shrink(self):
            # On the first shrink (target = B, the shorter sort_key),
            # sneak a smaller example into A — simulating the hook
            # discovering it while we ran B's test_function calls.
            if len(invocations) == 1:
                state.extras.interesting_examples[o_a] = small_a

    monkeypatch.setattr(multi_bug, "Shrinker", FakeShrinker)
    multi_bug.shrink_per_origin(state)
    assert big_b in invocations
    assert small_a in invocations


def test_shrink_per_origin_caps_iterations(monkeypatch):
    """The MAX_RESHRINK_PASSES cap prevents pathological re-shrink
    cycles. Forcing MAX_RESHRINK_PASSES to a low value bounds the loop
    even if every iteration evicts another origin."""

    def tf(tc):
        for _ in range(2):
            tc.draw_integer(0, 10)

    state = PbtkitState(Random(0), tf, max_examples=1)
    multi_bug._ensure_state(state)

    o_a = InterestingOrigin(IndexError, "f.py", 1)
    o_b = InterestingOrigin(ValueError, "f.py", 2)
    a = TC.for_choices([3, 3])
    state.test_function(a)
    a.status = Status.INTERESTING  # type: ignore[assignment]
    b = TC.for_choices([4, 4])
    state.test_function(b)
    b.status = Status.INTERESTING  # type: ignore[assignment]
    state.extras.interesting_examples = {o_a: a, o_b: b}

    invocations: list = []
    counter = {"value": 10}

    class FakeShrinker:
        def __init__(self, state, initial, is_interesting):
            self.current = initial
            invocations.append(initial)

        def shrink(self):
            # Inject a strictly-smaller TC into both origins each
            # iteration, forcing eviction. The decreasing counter
            # guarantees each new TC has a smaller sort_key than the
            # previous, so the eviction predicate fires.
            counter["value"] -= 1
            v = counter["value"]
            for orig in [o_a, o_b]:
                tc = TC.for_choices([v])
                state.test_function(tc)
                tc.status = Status.INTERESTING  # type: ignore[assignment]
                state.extras.interesting_examples[orig] = tc

    monkeypatch.setattr(multi_bug, "Shrinker", FakeShrinker)
    monkeypatch.setattr(multi_bug, "MAX_RESHRINK_PASSES", 3)
    multi_bug.shrink_per_origin(state)
    assert len(invocations) == 3


def test_ensure_state_is_idempotent():
    state = PbtkitState(Random(0), lambda tc: None, max_examples=1)
    multi_bug._ensure_state(state)
    examples = state.extras.interesting_examples
    examples["x"] = "preserved"
    multi_bug._ensure_state(state)  # second call shouldn't reset
    assert state.extras.interesting_examples is examples
    assert examples["x"] == "preserved"


# ---------------------------------------------------------------------------
# Legacy (no-multi_bug) database paths
#
# The standard database setup/teardown hooks have a feature_enabled gate
# that delegates to multi_bug when enabled. With multi_bug always on in
# ``just test``, the legacy paths would be uncovered — we exercise them
# here by temporarily disabling the feature via DISABLED_MODULES.
# ---------------------------------------------------------------------------


def _disable_multi_bug(monkeypatch):
    """Patch DISABLED_MODULES so feature_enabled('multi_bug') returns
    False inside the test. Skips when running against the compiled
    single-file build, which doesn't have ``pbtkit.features`` (the
    feature system is stripped at compile time)."""
    try:
        import pbtkit.features
    except ModuleNotFoundError:
        pytest.skip("pbtkit.features not available in compiled build")
    monkeypatch.setattr(
        pbtkit.features,
        "DISABLED_MODULES",
        pbtkit.features.DISABLED_MODULES | frozenset({"multi_bug"}),
    )


@pytest.mark.requires("database")
def test_legacy_database_setup_replays_single_best(tmpdir, monkeypatch):
    """With multi_bug disabled, _database_setup falls back to reading
    the single ``test_name`` key and replaying it."""
    from pbtkit.database import _database_setup, _serialize_choices

    db = DirectoryDB(str(tmpdir))
    db["legacy_test"] = _serialize_choices([7])
    _disable_multi_bug(monkeypatch)

    captured: list = []

    def tf(tc):
        v = tc.draw_integer(0, 100)
        captured.append(v)

    state = PbtkitState(Random(0), tf, max_examples=1, test_name="legacy_test")
    state.extras.database = db
    _database_setup(state)
    assert captured == [7]


@pytest.mark.requires("database")
def test_legacy_database_setup_handles_missing_key(tmpdir, monkeypatch):
    """Missing key — no replay, no error."""
    from pbtkit.database import _database_setup

    db = DirectoryDB(str(tmpdir))
    _disable_multi_bug(monkeypatch)

    state = PbtkitState(Random(0), lambda tc: None, max_examples=1, test_name="absent")
    state.extras.database = db
    _database_setup(state)
    assert state.calls == 0


@pytest.mark.requires("database")
def test_legacy_database_setup_handles_malformed_payload(tmpdir, monkeypatch):
    """Malformed payload — _deserialize_choices returns None, no replay.
    Uses an integer tag with truncated body so _read_fixed raises and
    _deserialize_choices catches the ValueError."""
    from pbtkit.database import _database_setup

    db = DirectoryDB(str(tmpdir))
    # Tag 0 = INTEGER, expects 8 bytes; we only give 4.
    db["legacy_test"] = b"\x00\x01\x02\x03\x04"
    _disable_multi_bug(monkeypatch)

    state = PbtkitState(
        Random(0), lambda tc: None, max_examples=1, test_name="legacy_test"
    )
    state.extras.database = db
    _database_setup(state)
    assert state.calls == 0


@pytest.mark.requires("database")
def test_legacy_database_teardown_writes_result(tmpdir, monkeypatch):
    """With multi_bug disabled, teardown serialises state.result to the
    single-key DB entry."""
    from pbtkit.database import _database_teardown, _deserialize_choices

    db = DirectoryDB(str(tmpdir))
    _disable_multi_bug(monkeypatch)

    def tf(tc):
        tc.draw_integer(0, 10)
        tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), tf, max_examples=1, test_name="t")
    state.extras.database = db
    state.run()
    _database_teardown(state)
    raw = db.get("t")
    assert raw is not None
    assert _deserialize_choices(raw) is not None


@pytest.mark.requires("database")
def test_legacy_database_teardown_clears_when_no_result(tmpdir, monkeypatch):
    """state.result is None ⇒ delete the entry."""
    from pbtkit.database import _database_teardown

    db = DirectoryDB(str(tmpdir))
    db["t"] = b"old"
    _disable_multi_bug(monkeypatch)

    state = PbtkitState(Random(0), lambda tc: None, max_examples=1, test_name="t")
    state.extras.database = db
    _database_teardown(state)
    assert db.get("t") is None


@pytest.mark.requires("database")
def test_legacy_database_teardown_swallows_missing_key(tmpdir, monkeypatch):
    """Deleting an already-absent key is silently swallowed."""
    from pbtkit.database import _database_teardown

    db = DirectoryDB(str(tmpdir))
    _disable_multi_bug(monkeypatch)

    state = PbtkitState(Random(0), lambda tc: None, max_examples=1, test_name="t")
    state.extras.database = db
    _database_teardown(state)  # KeyError swallowed
    assert db.get("t") is None


@pytest.mark.requires("database")
@pytest.mark.requires("draw_names")
def test_legacy_run_test_reporting_path(monkeypatch, capsys):
    """With multi_bug disabled, the reporting block in ``run_test``
    takes its single-print fallback (the ``else`` branch in
    ``core.py``)."""
    import pbtkit.generators as gs

    _disable_multi_bug(monkeypatch)

    with pytest.raises(AssertionError):

        @run_test(database={}, max_examples=10, random=Random(0), quiet=False)
        def _(tc):
            n = tc.draw(gs.integers(0, 100))
            assert n < 0

    out = capsys.readouterr().out
    assert "n = " in out
