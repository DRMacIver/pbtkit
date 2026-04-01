---
name: debug-shrink
description: Find and fix pbtkit shrink bugs by running pbtsmith fuzzing (both crash tests and shrink quality comparison against Hypothesis). Use this skill when the user says "/debug-shrink", asks to find shrink bugs, wants to run pbtsmith, or wants to iterate on shrink pass quality. Also use proactively after modifying any shrink pass code.
---

# Debug Shrink

Find bugs in pbtkit shrink passes by running two pbtsmith-based test suites, then fix each bug with a regression-test-first workflow.

There are two test suites to run:

1. **test_pbtsmith.py** — generates random pbtkit programs and checks they don't crash internally.
2. **test_shrink_comparison.py** — generates random programs, runs them under both pbtkit and Hypothesis, and asserts pbtkit shrinks at least as well as the worst of 10 Hypothesis runs.

Both find real bugs, but the shrink comparison test is the more important one — it catches shrink quality regressions, not just crashes.

## The loop

Repeat these steps until both test suites are clean:

### Step 1: Run the tests

Run both test suites. Pipe output to temp files (never pipe to tail or head).

```bash
uv run python -m pytest tests/test_shrink_comparison.py -x --no-header --tb=long -q 2>&1 > /tmp/shrink-comparison-output.txt
```

Read the file. If it passes, also run the crash test:

```bash
uv run python -m pytest tests/test_pbtsmith.py -x --no-header --tb=long -q 2>&1 > /tmp/pbtsmith-output.txt
```

If both pass, report success and stop.

### Step 2: Understand the failure

The failing program is shown via Hypothesis `note()` in the output. Extract it.

Classify the failure:

- **Internal error** (AssertionError, IndexError, OverflowError from `src/pbtkit/`): the regression test just needs to run the program without crashing.
- **Shrink quality failure** (pbtkit result is worse than Hypothesis under `sort_key`): need to determine what pbtkit should shrink to and assert on it. The comparison test output shows both the pbtkit result and the worst Hypothesis result with their sort_keys.

### Step 3: Write a regression test

**Put it in `tests/test_core.py`**, not test_pbtsmith.py or test_shrink_comparison.py. Those files have `pytestmark = [pytest.mark.hypothesis]` which excludes them from coverage runs.

For internal errors, use this pattern:

```python
def test_descriptive_name():
    """One-line description. Regression for <ErrorType> in <file> found by pbtsmith."""
    try:
        @run_test(max_examples=<N>, database={}, quiet=True, random=Random(<seed>))
        def _(tc):
            # ... minimal test body ...
            pass
    except (AssertionError, TestFailed):
        pass
```

For shrink quality failures, use the `minimal()` helper from `tests/shrink_quality/conftest.py`, or write a direct test using `PbtkitState`:

```python
def test_descriptive_name():
    """One-line description. Regression for shrink quality found by pbtsmith."""
    def tf(tc):
        # ... test body that marks INTERESTING ...
        tc.mark_status(Status.INTERESTING)

    state = State(Random(0), tf, 1000)
    state.run()
    assert state.result is not None
    values = [n.value for n in state.result]
    # Assert the expected shrunk result
    assert values == [expected, values, here]
```

**Verify the test fails** before any fix is applied:
```bash
uv run python -m pytest tests/test_core.py::test_descriptive_name -x --no-header --tb=short -q
```

### Step 4: Simplify the regression test

The pbtsmith-generated test body is usually more complex than needed. Simplify it:

**Critical rule: reproduce EXACTLY first.** Before simplifying anything, write a
test that reproduces the exact failure using the same seed, max_examples, and
program body from the pbtsmith output. Verify it fails. Only THEN simplify.

**Simplification procedure:**

1. **Reproduce exactly** with the original seed and max_examples from the failure.
2. **Understand what the shrinker produces vs what's optimal.** Run the test and
   check the result. Then manually verify the optimal result is reachable (e.g.
   via `state.replace()`). This tells you what the shrinker needs to find.
3. **Understand WHY the shrinker can't find it.** Trace through what passes try:
   what values does `try_shortening_via_increment` attempt? What does
   `lower_and_bump` try? Often the issue is a compound change (N positions need
   to change simultaneously) that no single pass discovers.
4. **Simplify the program**, keeping the core structural issue intact:
   - Remove draws that aren't involved in the bug.
   - Simplify predicates — replace compound conditions with the minimal trigger.
   - Narrow integer ranges to the smallest that reproduces.
   - Remove if-blocks, assumes, and trailing draws that don't matter.
   - **After each simplification, re-run** to confirm the same failure.
   - If the simplified version passes, the removed part was load-bearing — put it back.
5. **Try seed 0.** If the simplified test fails with seed 0 too, use that (it's
   the conventional default). If not, keep the original seed.
6. **Assert on the specific failure.** Prefer `assert len(state.result) == N` or
   `assert values == [...]` over vague assertions. The reader should see exactly
   what the expected vs actual result is.

To understand what pbtkit shrinks a test to, use this helper:

```bash
PYTHONPATH=src uv run python -c "
from random import Random
from pbtkit.core import PbtkitState, Status, StopTest, TestCase

def test_fn(tc):
    # paste the test body here
    pass

best = [None]
def wrapper(tc):
    test_fn(tc)
    best[0] = [n.value for n in tc.nodes]

state = PbtkitState(Random(0), wrapper, 1000)
state.run()
if state.result:
    tc = TestCase.for_choices([n.value for n in state.result])
    try:
        wrapper(tc)
    except StopTest:
        pass
    print(f'Result choices: {[n.value for n in state.result]}')
    print(f'Result values: {best[0]}')
else:
    print('No interesting result found')
"
```

After each simplification, re-run the test to confirm it still fails the same way.

### Step 5: Verify with canaries

If the fix involves adding a guard or new branch, place an `assert False` canary inside it and verify the regression test (or pbtsmith) hits it:

```python
if stale_index_detected:
    assert False  # canary: verify this path is reachable
    break
```

Run the regression test. If it hits the canary, the branch is reachable and will get coverage. Remove the canary before committing.

If the canary is NOT hit, the branch may be unreachable from existing tests. Options per CLAUDE.md:
- Write a test that covers it (preferred)
- Convert to an assert (if truly unreachable)
- Restructure to eliminate the branch (e.g., recompute stale data instead of guarding)

### Step 6: Fix the bug

Only now, with a minimal failing regression test in hand, write the fix.

**Beware stale reads in shrink passes.** Any call to `state.test_function`,
`state.replace`, or `state.consider` can change `state.result`. After such a
call, ALL previously read values from `state.result` (indices, kinds, values,
lengths) may be stale. Re-read from `state.result` before using any of these.
Common symptoms: `IndexError`, `TypeError` (wrong type at a position),
`AssertionError` from bounds checks. Also beware caching `len(state.result)`
as `max_size` — re-read it each iteration since prior mutations may have
shortened the result.

Common fix patterns for crashes:
- **Stale indices**: Recompute indices inside the loop instead of once upfront.
- **Serialization errors**: Add missing `signed=True`, handle edge cases.
- **Type mismatches after replace**: Convert hard asserts to guards that bail out.

Common fix patterns for shrink quality:
- **Missing shrink pass**: Write a new `@shrink_pass` function and register it.
- **Insufficient redistribution**: Extend an existing pass to handle more cases (e.g., negative values, wider gaps).
- **Pass ordering**: Move a pass earlier/later in the registry.

### Step 7: Verify everything

```bash
just test      # 100% branch coverage
just typecheck # 0 errors
just format    # clean
```

All three must pass before committing.

### Step 8: Commit

Commit the regression test and fix together. The commit message should name the bug and mention pbtsmith:

```
Fix <description of bug>

<Brief explanation of root cause and fix.>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

### Step 9: Loop

Go back to Step 1. Run both test suites again. Repeat until clean.

## Key rules

- **Regression test FIRST.** Never write a fix before the test exists and is verified to fail. This is non-negotiable per project CLAUDE.md.
- **Bytes and strings share infrastructure.** When you write a regression test for a bytes shrink issue, also write the parallel test for strings (and vice versa). The test is valuable even if it already passes — it guards against future regressions in the shared code.
- **tests/test_core.py, not test_pbtsmith.py.** Coverage is only collected from tests that run in `just test` (which uses `-m 'not hypothesis'`).
- **No piping to tail or head.** Pipe to a file and use the Read tool.
- **Simplify before fixing.** A 3-line regression test is worth more than a 30-line one. The simpler the test, the clearer the bug, the better the fix.
- **Crashes before quality.** Either suite can surface crashes. If you see a crash (internal AssertionError, IndexError, etc. from pbtkit source), fix it before tackling shrink quality failures. Crashes indicate a fundamental problem in the shrinker code.
- **Unit test specific components.** When a shrink bug reveals a problem in a specific component (cache, serialization, value punning, etc.), write targeted unit tests for that component in addition to the end-to-end regression test. The unit tests are more focused and easier to reason about.
