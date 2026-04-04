"""Exercise each shrink pass on arbitrary pbtsmith-generated test cases.

Uses Hypothesis to generate a random pbtsmith program, runs it through
the Hypothesis ConjectureData adapter to get a failing choice sequence,
then replays it through pbtkit and runs a single shrink pass. This
catches crashes in individual shrink passes under unusual conditions.
"""

from random import Random

import pytest

from hypothesis import given, note, reject
from hypothesis import strategies as st
from pbtkit.core import SHRINK_PASSES, PbtkitState, Status, TestCase

from .test_pbtsmith import Failure
from .test_pbtsmith import program as pbtsmith_program
from .test_shrink_comparison import ConjectureTestCase, _extract_test_body

pytestmark = [
    pytest.mark.requires("floats"),
    pytest.mark.requires("text"),
    pytest.mark.requires("bytes"),
    pytest.mark.requires("collections"),
    pytest.mark.hypothesis,
]


@pytest.mark.parametrize("shrink_pass", SHRINK_PASSES, ids=lambda p: p.__name__)
@given(st.data())
def test_shrink_passes_can_be_run_in_arbitrary_conditions(shrink_pass, data):
    seed = data.draw(st.integers())

    test_function = _extract_test_body(data.draw(pbtsmith_program()))
    base_test_case = ConjectureTestCase(data.conjecture_data)

    try:
        test_function(base_test_case)
        reject()
    except Failure:
        pass

    note(f"Drawn values: {[n.value for n in base_test_case.nodes]}")

    def mark_failures_interesting(test_case: TestCase) -> None:
        try:
            test_function(test_case)
        except Exception:
            if test_case.status is not None:
                raise
            test_case.mark_status(Status.INTERESTING)

    state = PbtkitState(
        random=Random(seed),
        test_function=mark_failures_interesting,
        max_examples=1,
        test_name="shrinking_test",
    )

    values = [n.value for n in base_test_case.nodes]
    tc = TestCase.for_choices(values, prefix_nodes=base_test_case.nodes)
    state.test_function(tc)
    if state.result is None:
        reject()
    shrink_pass(state)
