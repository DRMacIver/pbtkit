"""Multi-example support for pbtkit.

When imported, this module enables discovery and reporting of multiple
distinct failures. Pass ``report_multiple=True`` to ``run_test()`` to
activate: generation continues after the first failure to discover
additional failure types, each is shrunk independently, and all are
reported via a ``MultipleFailures`` exception.

A failure's identity is determined by its exception type and the source
location where it was raised. Two failures are considered distinct if
they differ in either of these.
"""

from __future__ import annotations

from pbtkit.core import (
    BUFFER_SIZE,
    PbtkitState,
    Status,
    StopTest,
    TestCase,
    run_phase,
    setup_hook,
    sort_key,
    teardown_hook,
    test_function_hook,
)


class MultipleFailures(Exception):
    """Raised when a test has multiple distinct failures."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        lines = [f"Found {len(errors)} distinct failures:"]
        for i, e in enumerate(errors, 1):
            lines.append(f"  {i}. {type(e).__name__}: {e}")
        super().__init__("\n".join(lines))


def _exception_key(exc: BaseException) -> tuple[str, ...]:
    """Classify an exception by type and source location.

    Returns (exception_type, filename, lineno) from the deepest
    frame in the traceback, or just (exception_type,) if no
    traceback is available."""
    tb = exc.__traceback__
    if tb is None:
        return (type(exc).__name__,)
    while tb.tb_next is not None:
        tb = tb.tb_next
    return (type(exc).__name__, tb.tb_frame.f_code.co_filename, str(tb.tb_lineno))


@setup_hook
def _multi_example_setup(state: PbtkitState) -> None:
    """Wrap the test function to capture exception info on each failure.

    Only installs the wrapper chain when ``report_multiple=True`` is set."""
    if not getattr(state.extras, "report_multiple", False):
        return

    state.extras._multi_results = {}
    # When set, only failures matching this key are counted as interesting.
    state.extras._multi_active_key = None

    # Read the current test function. After draw_names, this is the
    # rewritten version. Without draw_names, it's the original.
    current_test = state._print_function or state._original_test
    assert current_test is not None

    def capturing_wrapper(test_case: TestCase) -> None:
        """Run the test and store any exception on the test case."""
        try:
            current_test(test_case)
        except StopTest:
            raise
        except Exception as e:
            test_case._failure_exception = e
            raise

    def mark_failures_interesting(test_case: TestCase) -> None:
        try:
            capturing_wrapper(test_case)
        except Exception:
            if test_case.status is not None:
                raise
            test_case.mark_status(Status.INTERESTING)

    def filtered_wrapper(test_case: TestCase) -> None:
        """Wraps mark_failures_interesting to filter by active key.

        During per-key shrinking, only failures matching the active key
        are left as INTERESTING. Others are downgraded to VALID so that
        state.consider() rejects them."""
        try:
            mark_failures_interesting(test_case)
        except StopTest:
            active_key = state.extras._multi_active_key
            if active_key is not None and test_case.status == Status.INTERESTING:
                exc = test_case._failure_exception
                assert exc is not None
                key = _exception_key(exc)
                if key != active_key:
                    test_case.status = Status.VALID
            raise

    # Install our wrapper chain, replacing whatever was there before.
    state._set_test_function(filtered_wrapper)


@test_function_hook
def _multi_example_track(state: PbtkitState, test_case: TestCase) -> None:
    """Track each distinct failure by key."""
    if test_case.status != Status.INTERESTING:
        return
    results = getattr(state.extras, "_multi_results", None)
    if results is None:
        return
    # Don't track during per-key shrinking — only during generation
    # and primary shrinking.
    if state.extras._multi_active_key is not None:
        return
    exc = test_case._failure_exception
    assert exc is not None
    key = _exception_key(exc)
    if key not in results or sort_key(test_case.nodes) < sort_key(results[key]):
        results[key] = list(test_case.nodes)


@run_phase
def _multi_example_discover(state: PbtkitState) -> None:
    """Continue generation after the first failure to find distinct failures.

    Only runs when ``report_multiple=True`` was passed to ``run_test()``."""
    if state.result is None:
        return
    if not getattr(state.extras, "report_multiple", False):
        return

    while (
        state.valid_test_cases < state.max_examples
        and state.calls < state.max_examples * 10
    ):
        test_case = TestCase(prefix=[], random=state.random, max_size=BUFFER_SIZE)
        state.test_function(test_case)


def _shrink_for_key(state: PbtkitState, key: tuple[str, ...]) -> None:
    """Shrink a specific failure key independently.

    Sets ``_multi_active_key`` so that the filtered_wrapper only accepts
    failures matching *key*, then runs the normal shrink loop."""
    results = state.extras._multi_results
    nodes = results[key]
    saved_result = state.result
    state.result = list(nodes)

    state.extras._multi_active_key = key
    try:
        state.shrink()
    finally:
        state.extras._multi_active_key = None

    results[key] = state.result
    state.result = saved_result


@teardown_hook
def _multi_example_teardown(state: PbtkitState) -> None:
    """Shrink additional failures and report all distinct ones."""
    if not getattr(state.extras, "report_multiple", False):
        return
    results = getattr(state.extras, "_multi_results", None)
    if results is None or len(results) <= 1 or state.result is None:
        return

    # Find which key corresponds to the primary (already-shrunk) result.
    # The tracking hook always stores the primary result under its key,
    # so this must succeed.
    primary_sk = sort_key(state.result)
    primary_key = next(
        key for key, nodes in results.items() if sort_key(nodes) == primary_sk
    )
    results[primary_key] = state.result

    # Shrink each non-primary failure independently.
    for key in list(results):
        if key == primary_key:
            continue
        _shrink_for_key(state, key)

    # Replay each distinct failure, collect exceptions.
    orig_test = state._print_function or state._original_test
    assert orig_test is not None
    errors: list[Exception] = []
    for key in sorted(results, key=lambda k: sort_key(results[k])):
        nodes = results[key]
        print(f"\nFailure {len(errors) + 1}:")
        try:
            orig_test(
                TestCase.for_choices([n.value for n in nodes], print_results=True)
            )
        except Exception as e:
            errors.append(e)

    # Prevent default replay by clearing the result.
    state.result = None

    assert len(errors) >= 2
    raise MultipleFailures(errors)
