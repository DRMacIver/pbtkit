"""Compare minithesis shrinking quality against Hypothesis.

Generates random minismith programs, runs them under both a Hypothesis
ConjectureRunner and minithesis, and checks that minithesis produces a
result at least as good as the worst of 10 Hypothesis runs under the
minithesis shrinking order.
"""

from __future__ import annotations

import math
from random import Random

import pytest
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    calc_label_from_name,
)
from hypothesis.internal.conjecture.data import (
    Status as HStatus,
)
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.intervalsets import IntervalSet

from hypothesis import HealthCheck, assume, given, note, settings
from hypothesis import strategies as st
from minithesis.bytes import BytesChoice
from minithesis.core import (  # noqa: F401
    BooleanChoice,
    ChoiceNode,
    Generator,
    IntegerChoice,
    MinithesisState,
    Status,
    StopTest,
    TestCase,
    Unsatisfiable,
    run_test,
    sort_key,
)
from minithesis.floats import FloatChoice

# We need the generators in scope so the exec'd program can reference them.
from minithesis.generators import (  # noqa: F401
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
from minithesis.text import StringChoice

from .test_minismith import Failure, program

# ---------------------------------------------------------------------------
# ConjectureTestCase adapter
# ---------------------------------------------------------------------------


class ConjectureTestCase:
    """Adapts a Hypothesis ConjectureData to the minithesis TestCase interface.

    Each draw method delegates to the corresponding ConjectureData primitive
    and records a ChoiceNode for later comparison under minithesis sort_key.
    The any() method wraps generator calls in start_span/stop_span."""

    def __init__(self, data: ConjectureData):
        self.data = data
        self.nodes: list[ChoiceNode] = []
        self.status: Status | None = None
        self.depth = 0
        self.targeting_score: int | None = None
        self.print_results = False

    def draw_integer(self, min_value: int, max_value: int) -> int:
        result = self.data.draw_integer(min_value=min_value, max_value=max_value)
        self.nodes.append(
            ChoiceNode(IntegerChoice(min_value, max_value), result, False)
        )
        return result

    def weighted(self, p: float, *, forced: bool | None = None) -> bool:
        if p <= 0:
            forced = False
        elif p >= 1:
            forced = True
        if forced is not None:
            result = forced
        else:
            result = self.data.draw_boolean(p=p)
        self.nodes.append(
            ChoiceNode(BooleanChoice(p), bool(result), forced is not None)
        )
        return bool(result)

    def draw_float(
        self,
        min_value: float = -math.inf,
        max_value: float = math.inf,
        *,
        allow_nan: bool = True,
        allow_infinity: bool = True,
    ) -> float:
        result = self.data.draw_float(
            min_value=min_value, max_value=max_value, allow_nan=allow_nan
        )
        if not allow_infinity and math.isinf(result):
            # Hypothesis may generate inf even when we don't want it;
            # reject this test case.
            self.data.mark_invalid("infinity not allowed")
        self.nodes.append(
            ChoiceNode(
                FloatChoice(min_value, max_value, allow_nan, allow_infinity),
                result,
                False,
            )
        )
        return result

    def draw_string(
        self,
        min_codepoint: int = 0,
        max_codepoint: int = 0x10FFFF,
        min_size: int = 0,
        max_size: int = 10,
    ) -> str:
        # Build an IntervalSet for the codepoint range, excluding surrogates.
        ranges = []
        lo = max(min_codepoint, 0)
        hi = min(max_codepoint, 0x10FFFF)
        if lo <= 0xD7FF and hi >= lo:
            ranges.append((lo, min(hi, 0xD7FF)))
        if hi >= 0xE000 and max(lo, 0xE000) <= hi:
            ranges.append((max(lo, 0xE000), hi))
        if not ranges:
            ranges = [(lo, hi)]
        intervals = IntervalSet(ranges)
        result = self.data.draw_string(intervals, min_size=min_size, max_size=max_size)
        self.nodes.append(
            ChoiceNode(
                StringChoice(min_codepoint, max_codepoint, min_size, max_size),
                result,
                False,
            )
        )
        return result

    def draw_bytes(self, min_size: int, max_size: int) -> bytes:
        result = self.data.draw_bytes(min_size=min_size, max_size=max_size)
        self.nodes.append(ChoiceNode(BytesChoice(min_size, max_size), result, False))
        return result

    def choice(self, n: int) -> int:
        return self.draw_integer(0, n)

    def forced_choice(self, n: int) -> int:
        self.nodes.append(ChoiceNode(IntegerChoice(0, n), n, True))
        return n

    def any(self, generator: Generator) -> object:
        label = calc_label_from_name(repr(generator))
        self.data.start_span(label)
        try:
            self.depth += 1
            result = generator.produce(self)
        finally:
            self.depth -= 1
            self.data.stop_span()
        return result

    def assume(self, precondition: bool) -> None:
        if not precondition:
            self.reject()

    def reject(self) -> None:
        self.data.mark_invalid("rejected")

    def mark_status(self, status: Status) -> None:
        if status == Status.INVALID:
            self.data.mark_invalid("rejected")
        elif status == Status.INTERESTING:
            self.data.mark_interesting(("minismith",))
        elif status == Status.EARLY_STOP:
            self.data.mark_invalid("early stop")

    def target(self, score: int) -> None:
        pass  # Ignore targeting for comparison purposes.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_test_body(program_code: str):
    """Extract the test body function from a minismith program string.

    Returns a callable that takes a TestCase-like object and runs the
    test body, raising Failure on assertion failure."""
    # The program may define composite helper functions before the
    # try block. Exec the whole program into a namespace, but replace
    # the run_test invocation with just defining the body function.
    #
    # The program structure is:
    #   [optional @composite defs]
    #   try:
    #       @run_test(...)
    #       def _(tc):
    #           <body>
    #   except ...:
    #       pass
    #
    # We replace everything from 'try:' onwards with just the function def.

    # Find the try: block
    lines = program_code.split("\n")
    preamble = []
    body_lines = []
    in_body = False
    for line in lines:
        if line.strip().startswith("try:"):
            in_body = True
            continue
        if in_body and line.strip().startswith("@run_test"):
            continue
        if in_body and line.strip().startswith("def _(tc):"):
            body_lines.append("def _test_body(tc):")
            continue
        if in_body and line.strip().startswith("except"):
            break
        if in_body:
            body_lines.append(line)
        else:
            preamble.append(line)

    func_source = "\n".join(preamble + body_lines)
    namespace = dict(globals())
    exec(compile(func_source, "<minismith-body>", "exec"), namespace)
    return namespace["_test_body"]


def _run_hypothesis(test_body, seed: int) -> list[ChoiceNode] | None:
    """Run the test body under Hypothesis ConjectureRunner.

    Returns the choice nodes from the shrunk interesting example,
    or None if no interesting example was found."""
    captured_nodes: list[list[ChoiceNode]] = []

    def test_fn(data: ConjectureData) -> None:
        tc = ConjectureTestCase(data)
        try:
            test_body(tc)
        except Failure:
            captured_nodes.append(list(tc.nodes))
            data.mark_interesting(("minismith",))
        except StopTest:
            pass

    runner = ConjectureRunner(
        test_fn,
        settings=settings(
            max_examples=100,
            database=None,
            suppress_health_check=list(HealthCheck),
        ),
        random=Random(seed),
    )
    runner.run()

    if not runner.interesting_examples:
        return None
    # Get the shrunk result's choices and replay to get our nodes.
    result = next(iter(runner.interesting_examples.values()))
    # Replay the shrunk choices through our adapter to get ChoiceNodes.
    replay_data = ConjectureData.for_choices(result.choices)
    tc = ConjectureTestCase(replay_data)
    try:
        test_body(tc)
    except (Failure, Exception):
        pass
    return tc.nodes


def _run_minithesis(test_body, seed: int) -> list[ChoiceNode] | None:
    """Run the test body under minithesis with shrinking.

    Returns the shrunk choice nodes, or None if no interesting
    example was found."""

    def test_fn(tc):
        try:
            test_body(tc)
        except Failure:
            tc.mark_status(Status.INTERESTING)

    state = MinithesisState(Random(seed), test_fn, 100)
    state.run()
    return state.result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.requires("floats"),
    pytest.mark.requires("text"),
    pytest.mark.requires("bytes"),
    pytest.mark.requires("collections"),
    pytest.mark.hypothesis,
]


@given(
    minithesis_program=program(),
    seed1=st.integers(),
    seed2=st.integers(),
)
@settings(
    max_examples=20000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_minithesis_shrinks_at_least_as_well_as_hypothesis(
    minithesis_program: str,
    seed1: int,
    seed2: int,
) -> None:
    """Minithesis should produce a result at least as good as the worst
    of 10 Hypothesis runs, under the minithesis shrinking order."""
    note(minithesis_program)

    test_body = _extract_test_body(minithesis_program)

    hypothesis_result = _run_hypothesis(test_body, seed2)
    assume(hypothesis_result is not None)

    minithesis_result = _run_minithesis(test_body, seed=seed1)
    assume(minithesis_result is not None)

    if sort_key(minithesis_result) > sort_key(hypothesis_result):
        # Replay both results through the test body, capturing the
        # top-level values returned by tc.any() calls.
        def _replay(nodes):
            tc = TestCase.for_choices([n.value for n in nodes], prefix_nodes=nodes)
            draws: list[tuple[str, object]] = []
            original_any = tc.any

            def capturing_any(gen):
                result = original_any(gen)
                if tc.depth == 0:
                    draws.append((gen.name, result))
                return result

            tc.any = capturing_any  # type: ignore[assignment]
            try:
                test_body(tc)
            except (Failure, StopTest, Exception):
                pass
            return draws

        mt_draws = _replay(minithesis_result)
        hy_draws = _replay(hypothesis_result)

        mt_vals = [v for _, v in mt_draws]
        hy_vals = [v for _, v in hy_draws]
        note(
            f"\nMinithesis choices: {[n.value for n in minithesis_result]}"
            f"\nMinithesis values:  {mt_vals!r}"
            f"\nHypothesis choices: {[n.value for n in hypothesis_result]}"
            f"\nHypothesis values:  {hy_vals!r}"
        )

        assert False, (
            f"Minithesis result {[n.value for n in minithesis_result]} "
            f"(sort_key={sort_key(minithesis_result)}) is worse than "
            f"Hypothesis result {[n.value for n in hypothesis_result]} "
            f"(sort_key={sort_key(hypothesis_result)})"
        )
