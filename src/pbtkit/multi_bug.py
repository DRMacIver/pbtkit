"""Multi-failure tracking for pbtkit.

When enabled, pbtkit records every distinct failing example, keyed by an
``InterestingOrigin`` derived from ``(exc_type, filename, lineno)`` of the
deepest non-pbtkit frame. Each origin's best example is shrunk
independently, and all are reported and persisted in the database.

Without this feature, pbtkit keeps its single-best-failure behaviour.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

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
    """If the test case is interesting, bucket it under its
    ``interesting_origin`` (set by core's mark_status). Test cases
    that mark INTERESTING without supplying an origin all share a
    single synthetic bucket — we also write that synthetic value back
    onto the test case so the per-origin shrink predicate matches."""
    if test_case.status != Status.INTERESTING:
        return
    origin = test_case.interesting_origin
    if origin is None:
        origin = InterestingOrigin(BaseException, None, None)
        test_case.interesting_origin = origin
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
            return tc.status == Status.INTERESTING and tc.interesting_origin == _t

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
# Reporting: print every per-origin failing example.
# ---------------------------------------------------------------------------


def multi_bug_report(state: PbtkitState, quiet: bool) -> None:
    """Replacement reporter for the standard single-print block in
    ``run_test``. Prints a labelled section per origin so the user can
    tell distinct failures apart, then replays each example through
    the test function (which prints its draws).

    Each replay typically raises the failing example's exception. We
    catch them so every example gets printed, and re-raise the first
    (shortlex-smallest) one at the end so callers still see a real
    failure."""
    print_fn = state._print_function
    assert print_fn is not None
    examples = getattr(state.extras, "interesting_examples", None) or {}
    if not examples:
        # Fallback: nothing recorded by the hook (only possible if
        # state.result was set out-of-band). Replay it once and let
        # any exception propagate.
        assert state.result is not None
        print_fn(
            TestCase.for_choices(
                [n.value for n in state.result], print_results=not quiet
            )
        )
        return
    items = sorted(examples.items(), key=lambda kv: sort_key(kv[1].nodes))
    total = len(items)
    captured: list[BaseException] = []
    for idx, (origin, tc) in enumerate(items, start=1):
        if not quiet:
            if idx > 1:
                print()
            print(_origin_header(origin, idx, total))
        try:
            print_fn(
                TestCase.for_choices(
                    [n.value for n in tc.nodes], print_results=not quiet
                )
            )
        except BaseException as exc:  # noqa: BLE001
            captured.append(exc)
            if not quiet:
                # Show the traceback inline so the reporting output
                # actually tells the user what went wrong for each
                # example — not just the draws that led to it. Strip
                # pbtkit-internal frames so the output focuses on the
                # user's code.
                trimmed = _user_visible_traceback(exc)
                sys.stdout.write(
                    "".join(traceback.format_exception(type(exc), exc, trimmed))
                )
    if captured:
        # Re-raise the smallest (sorted first). The full set was
        # already printed above with per-example tracebacks, so no
        # information is lost — the re-raise is just so callers that
        # expect an exception (pytest.raises, run_test's outer
        # contract) still see one.
        raise captured[0]


def _user_visible_traceback(exc: BaseException):  # type: ignore[no-untyped-def]
    """Return *exc*'s traceback with pbtkit-internal frames stripped
    off the top.

    When the test raises, the captured traceback goes through
    ``multi_bug.multi_bug_report`` → ``state._print_function`` →
    ``TestCase`` machinery → the user's test function. Everything
    above the user's frame is pbtkit bookkeeping. If the whole stack
    is internal (shouldn't happen, but be safe), we keep it so the
    user sees something rather than nothing."""
    tb = exc.__traceback__
    head = tb
    while head is not None and "/pbtkit/" in head.tb_frame.f_code.co_filename:
        head = head.tb_next
    return head if head is not None else tb


def _origin_header(origin: InterestingOrigin, idx: int, total: int) -> str:
    """Format a one-line header describing a failing example."""
    if origin.exc_type is BaseException:
        # Synthetic origin: the test marked itself INTERESTING via
        # mark_status without supplying a tag. We can't say anything
        # specific.
        location = "no origin"
    else:
        loc = (
            f"{origin.filename}:{origin.lineno}"
            if origin.filename is not None
            else "<unknown>"
        )
        location = f"{origin.exc_type.__name__} at {loc}"
    if total == 1:
        return f"Falsifying example ({location}):"
    return f"Falsifying example {idx} of {total} ({location}):"
