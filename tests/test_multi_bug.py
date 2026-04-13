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
from pbtkit.database import InMemoryDB

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


def test_origin_header_handles_unknown_filename():
    """When an origin has filename=None (rare: exception with no
    traceback), the header reports ``<unknown>`` for the location."""
    o = InterestingOrigin(ValueError, None, None)
    assert "<unknown>" in multi_bug._origin_header(o, 1, 1)


def test_origin_header_handles_synthetic_no_origin():
    """Test cases that mark INTERESTING via mark_status without an
    origin get a header that says ``no origin``."""
    o = InterestingOrigin(BaseException, None, None)
    assert "no origin" in multi_bug._origin_header(o, 1, 1)


def test_origin_header_numbers_multi():
    """When there are multiple origins, the header includes the index."""
    o = InterestingOrigin(ValueError, "f.py", 1)
    header = multi_bug._origin_header(o, 2, 5)
    assert "2 of 5" in header


def test_quiet_suppresses_headers_and_tracebacks(capsys):
    """quiet=True silences the per-origin header and the inline
    traceback printing."""

    def user_test(tc):
        n = tc.draw_integer(0, 10)
        if n >= 0:
            raise AssertionError(f"value was {n}")

    def shim(tc):
        try:
            user_test(tc)
        except Exception as exc:
            if tc.status is not None:
                raise
            tc.mark_status(
                Status.INTERESTING,
                interesting_origin=InterestingOrigin.from_exception(exc),
            )

    state = PbtkitState(Random(0), shim, max_examples=10)
    state.extras.test_name = "T"
    state._original_test = user_test
    state._print_function = user_test
    state.run()
    capsys.readouterr()  # clear

    with pytest.raises(AssertionError):
        multi_bug.multi_bug_report(state, quiet=True)

    out = capsys.readouterr().out
    assert "Falsifying example" not in out
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# Per-origin shrinking via run_test
# ---------------------------------------------------------------------------


def test_two_distinct_failures_are_each_recorded_and_shrunk():
    """A test that fails two ways (different lines) records both
    origins, and per-origin shrinking minimises each."""

    captured: list = []

    def runner():
        with pytest.raises(BaseException):

            @run_test(database=InMemoryDB(), max_examples=200, random=Random(0))
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
    """Two distinct origins are persisted as separate DB entries;
    reload replays both and reconstructs the per-origin dict."""
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
    # First run: the key's subdirectory holds two value files,
    # one per origin.
    key_dirs = list(tmpdir.listdir())
    assert len(key_dirs) == 1
    assert len(list(key_dirs[0].listdir())) == 2

    # Reload: same logic, same shape. Round-trip is stable.
    runner()
    key_dirs = list(tmpdir.listdir())
    assert len(key_dirs) == 1
    assert len(list(key_dirs[0].listdir())) == 2


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
# Feature-disabled path coverage
#
# The reporting block in ``run_test`` and the ``should_keep_generating``
# else branch are dead when multi_bug is enabled at import time
# (the feature_enabled gate is evaluated once at class-creation /
# branch-selection time). ``just test-compiled --disable=multi_bug``
# is the authoritative test for those paths; we monkey-patch here
# so coverage of the report fallback is also exercisable at runtime.
# ---------------------------------------------------------------------------


def _disable_multi_bug(monkeypatch):
    """Patch DISABLED_MODULES so feature_enabled('multi_bug') returns
    False inside the test. Skips when running against the compiled
    single-file build, which doesn't have ``pbtkit.features``."""
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
@pytest.mark.requires("draw_names")
def test_run_test_reporting_fallback(monkeypatch, capsys):
    """With multi_bug disabled at runtime, the ``if feature_enabled(...)
    else:`` reporting block in ``run_test`` takes its single-print
    fallback."""
    import pbtkit.generators as gs

    _disable_multi_bug(monkeypatch)

    with pytest.raises(AssertionError):

        @run_test(database=InMemoryDB(), max_examples=10, random=Random(0), quiet=False)
        def _(tc):
            n = tc.draw(gs.integers(0, 100))
            assert n < 0

    out = capsys.readouterr().out
    assert "n = " in out
