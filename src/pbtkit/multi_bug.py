"""Multi-failure tracking for pbtkit.

When enabled, pbtkit records every distinct failing example, keyed by an
``InterestingOrigin`` derived from ``(exc_type, filename, lineno)`` of the
deepest non-pbtkit frame. Each origin's best example is shrunk
independently, and all are reported and persisted in the database.

Without this feature, pbtkit keeps its single-best-failure behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pbtkit.core import (
    PbtkitState,
    Shrinker,
    Status,
    TestCase,
    setup_hook,
    sort_key,
    test_function_hook,
)

# Continuation budget after the first bug — mirrors Hypothesis.
MAX_EXTRA_CALLS = 1000

# Hard cap on the per-origin shrink driver so a pair of origins that
# keep nudging each other smaller can't loop forever.
MAX_RESHRINK_PASSES = 100


@dataclass(frozen=True)
class InterestingOrigin:
    """A signature for a failing test case used to bucket distinct bugs.

    Two failing test cases with the same exception type raised from the
    same source location (filename + line number of the deepest user
    frame) are considered the same bug; otherwise they are distinct
    and shrunk independently."""

    exc_type: type[BaseException]
    filename: str | None
    lineno: int | None

    @classmethod
    def from_exception(cls, exc: BaseException) -> InterestingOrigin:
        """Build an origin from the deepest frame in *exc*'s traceback."""
        tb = exc.__traceback__
        if tb is None:
            return cls(type(exc), None, None)
        while tb.tb_next is not None:
            tb = tb.tb_next
        return cls(type(exc), tb.tb_frame.f_code.co_filename, tb.tb_lineno)


def _ensure_state(state: PbtkitState) -> None:
    """Initialise multi_bug fields on ``state.extras`` if not already set.

    Done lazily because the standard database setup hook may call
    ``state.test_function`` before our setup hook runs (registration
    order: database registers before multi_bug)."""
    if not hasattr(state.extras, "interesting_examples"):
        state.extras.interesting_examples = {}
        state.extras.first_bug_at = None
        state.extras.last_new_origin_at = None


@setup_hook
def _multi_bug_setup(state: PbtkitState) -> None:
    _ensure_state(state)


@test_function_hook
def _record_origin(state: PbtkitState, test_case: TestCase) -> None:
    """If the test case is interesting, derive its origin and update
    the per-origin best-example dict."""
    if test_case.status != Status.INTERESTING:
        return
    exc = test_case._exception
    if exc is None:
        # Test marked itself INTERESTING via mark_status without raising.
        # Bucket all such cases under a single synthetic origin so they
        # still get tracked (and shrunk as one group, matching today's
        # single-best behaviour for non-exception failures).
        origin = InterestingOrigin(BaseException, None, None)
    else:
        origin = InterestingOrigin.from_exception(exc)
    setattr(test_case, "_origin", origin)
    _ensure_state(state)
    examples = state.extras.interesting_examples
    is_new = origin not in examples
    if is_new or sort_key(test_case.nodes) < sort_key(examples[origin].nodes):
        examples[origin] = test_case
    if state.extras.first_bug_at is None:
        state.extras.first_bug_at = state.calls
    if is_new:
        state.extras.last_new_origin_at = state.calls


def multi_bug_should_keep_generating(state: PbtkitState) -> bool:
    """Replacement for ``PbtkitState.should_keep_generating`` when
    multi_bug is enabled. Keeps generating after the first failure
    until the budget for finding NEW origins is exhausted."""
    if state.test_is_trivial:
        return False
    if state.valid_test_cases >= state.max_examples:
        return False
    if state.calls >= state.max_examples * 10:
        return False
    examples = getattr(state.extras, "interesting_examples", None)
    if not examples:
        return True
    # Continue while we are still plausibly close to discovering more
    # origins. The double-the-most-recent-origin-time clause means tests
    # whose bugs collapse to a single origin stop quickly; the +1000 cap
    # bounds the worst case for genuinely diverse test failures.
    return state.calls < min(
        state.extras.first_bug_at + MAX_EXTRA_CALLS,
        state.extras.last_new_origin_at * 2 + 1,
    )


def shrink_per_origin(state: PbtkitState) -> None:
    """Shrink each origin's best example with a per-origin predicate.

    A predicate of "interesting AND same origin" prevents shrink
    transitions from one origin to another. New origins discovered
    during shrinking get added to ``examples`` (via _record_origin) and
    are picked up on the next loop iteration.

    Called from ``PbtkitState.shrink`` when multi_bug is enabled. Safe
    to call multiple times — fully shrunk examples are at fixpoint and
    further passes make no progress.

    A per-origin shrink can incidentally improve a different origin's
    example (its predicate-failing test case still records via the
    test_function_hook). When that happens we evict the improved origin
    from ``shrunk`` and re-shrink it. Each re-shrink strictly decreases
    that origin's sort_key, so the loop terminates — but the
    ``MAX_RESHRINK_PASSES`` cap bounds pathological cases where
    shrinkers keep nudging each other."""
    examples = getattr(state.extras, "interesting_examples", None)
    if not examples:
        return
    shrunk: set[InterestingOrigin] = set()
    iterations = 0
    while iterations < MAX_RESHRINK_PASSES:
        pending = [o for o in examples if o not in shrunk]
        if not pending:
            break
        iterations += 1
        target = min(pending, key=lambda o: sort_key(examples[o].nodes))
        before = dict(examples)

        def predicate(tc: TestCase, _t: InterestingOrigin = target) -> bool:
            return (
                tc.status == Status.INTERESTING and getattr(tc, "_origin", None) == _t
            )

        sh = Shrinker(state=state, initial=examples[target], is_interesting=predicate)
        sh.shrink()
        examples[target] = sh.current
        shrunk.add(target)
        # If the shrink incidentally improved any *other* origin (its
        # test_function_hook would have updated examples), re-shrink it.
        # Excluding `target` itself is critical — its example always
        # changes after a shrink and we must not loop on that.
        for o, tc in examples.items():
            if o == target or o not in before:
                continue
            if sort_key(tc.nodes) < sort_key(before[o].nodes):
                shrunk.discard(o)

    # Update state.result so the standard database teardown and the
    # reporting fallback see the shortlex-smallest example overall.
    state.result = min(examples.values(), key=lambda tc: sort_key(tc.nodes)).nodes


# ---------------------------------------------------------------------------
# Database integration: store every per-origin example under a side key.
# ---------------------------------------------------------------------------


def _serialize_multi(sequences: Sequence[Sequence[Any]]) -> bytes:
    """Length-prefixed list of choice sequences: u32 count, then for
    each: u32 size + size bytes of _serialize_choices."""
    from pbtkit.database import _serialize_choices

    parts: list[bytes] = [len(sequences).to_bytes(4, "big")]
    for seq in sequences:
        body = _serialize_choices(seq)
        parts.append(len(body).to_bytes(4, "big"))
        parts.append(body)
    return b"".join(parts)


def _deserialize_multi(data: bytes) -> list[list] | None:
    """Inverse of _serialize_multi; returns None on malformed input.

    The explicit length checks cover every truncation path, and
    ``_deserialize_choices`` swallows its own ``IndexError`` /
    ``ValueError`` and returns ``None``, so no try/except is needed
    here — we just propagate the ``None`` if an inner sequence won't
    parse."""
    from pbtkit.database import _deserialize_choices

    if len(data) < 4:
        return None
    count = int.from_bytes(data[:4], "big")
    offset = 4
    sequences: list[list] = []
    for _ in range(count):
        if offset + 4 > len(data):
            return None
        size = int.from_bytes(data[offset : offset + 4], "big")
        offset += 4
        if offset + size > len(data):
            return None
        seq = _deserialize_choices(data[offset : offset + size])
        if seq is None:
            return None
        sequences.append(seq)
        offset += size
    return sequences


def load_from_db(state: PbtkitState, db: Any, test_name: str) -> None:
    """Replay every previously-saved per-origin example. Called by the
    standard database setup hook (``pbtkit.database._database_setup``)
    when multi_bug is enabled, *replacing* the legacy single-best
    replay so we don't double-replay."""
    raw = db.get(test_name + ".multi")
    if raw is None:
        return
    sequences = _deserialize_multi(raw)
    if sequences is None:
        return
    for seq in sequences:
        state.test_function(TestCase.for_choices(seq))


def save_to_db(state: PbtkitState, db: Any, test_name: str) -> None:
    """Persist (or clear) the per-origin examples under
    ``test_name + '.multi'``. Called by the standard database teardown
    hook when multi_bug is enabled."""
    examples = getattr(state.extras, "interesting_examples", None) or {}
    if not examples:
        try:
            del db[test_name + ".multi"]
        except KeyError:
            pass
        return
    sequences = [[n.value for n in tc.nodes] for tc in examples.values()]
    db[test_name + ".multi"] = _serialize_multi(sequences)


# ---------------------------------------------------------------------------
# Reporting: print every per-origin failing example.
# ---------------------------------------------------------------------------


def multi_bug_report(state: PbtkitState, quiet: bool) -> None:
    """Replacement reporter for the standard single-print block in
    ``run_test``. Prints each per-origin example. Falls back to
    ``state.result`` if ``interesting_examples`` is empty for any reason."""
    print_fn = state._print_function
    assert print_fn is not None
    examples = getattr(state.extras, "interesting_examples", None) or {}
    if not examples:
        assert state.result is not None
        print_fn(
            TestCase.for_choices(
                [n.value for n in state.result], print_results=not quiet
            )
        )
        return
    for tc in examples.values():
        print_fn(
            TestCase.for_choices([n.value for n in tc.nodes], print_results=not quiet)
        )
