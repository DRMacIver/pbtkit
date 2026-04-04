"""Findability comparison: Hypothesis oracle vs pbtkit.

Uses pbtsmith to generate random test programs. Each program is run
under Hypothesis (100 examples) as an oracle. If Hypothesis finds a
failure, pbtkit gets 1000 examples to find the same failure.
"""

from __future__ import annotations

from random import Random

import pytest

from hypothesis import HealthCheck, assume, given, note, settings
from pbtkit.core import (  # noqa: F401
    PbtkitState,
    Status,
    Unsatisfiable,
    run_test,
)

try:
    from pbtkit.generators import (  # noqa: F401
        binary,
        booleans,
        composite,
        dictionaries,
        floats,
        integers,
        just,
        lists,
        nothing,
        one_of,
        sampled_from,
        text,
        tuples,
    )
except (ImportError, NotImplementedError):
    pytest.skip("requires all generator types", allow_module_level=True)

from .test_pbtsmith import Failure, program
from .test_shrink_comparison import _extract_test_body, _run_hypothesis

pytestmark = [
    pytest.mark.requires("floats"),
    pytest.mark.requires("text"),
    pytest.mark.requires("bytes"),
    pytest.mark.requires("collections"),
    pytest.mark.hypothesis,
]


def _pbtkit_finds(test_body, max_examples: int = 5000) -> bool:
    """Return True if pbtkit finds an interesting example."""

    def test_fn(tc):
        try:
            test_body(tc)
        except Failure:
            tc.mark_status(Status.INTERESTING)

    state = PbtkitState(Random(0), test_fn, max_examples)
    state.run()
    return state.result is not None


@given(program())
def test_pbtkit_finds_what_hypothesis_finds(pbtkit_program: str) -> None:
    """If Hypothesis can find a failure in 100 examples, pbtkit should
    find it in 1000."""
    note(pbtkit_program)

    test_body = _extract_test_body(pbtkit_program)

    # Oracle: Hypothesis with 100 examples
    hyp_result = _run_hypothesis(test_body, seed=0)
    assume(hyp_result is not None)

    # Test: pbtkit with 1000 examples
    assert _pbtkit_finds(test_body), (
        "Hypothesis found a failure but pbtkit (1000 examples) did not"
    )
