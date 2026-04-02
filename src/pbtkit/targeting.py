"""Targeting support for pbtkit.

This module adds score-based targeting (hill climbing) to the core
engine. When imported, it patches TestCase with a target() method
and registers a targeting generation type and test-function hook.
"""

from __future__ import annotations

from pbtkit.core import (
    BUFFER_SIZE,
    ChoiceNode,
    IntegerChoice,
    PbtkitState,
    Status,
    TestCase,
    generation_type,
    test_function_hook,
)


def _target(self: TestCase, score: int) -> None:
    """Set a score to maximize. Multiple calls to this function
    will override previous ones.

    The name and idea come from Löscher, Andreas, and Konstantinos
    Sagonas. "Targeted property-based testing." ISSTA. 2017, but
    the implementation is based on that found in Hypothesis,
    which is not that similar to anything described in the paper.
    """
    self.targeting_score = score


TestCase.target = _target


@test_function_hook
def _targeting_hook(state: PbtkitState, test_case: TestCase) -> None:
    """Track the best targeting score seen so far."""
    if test_case.status is not None and test_case.status >= Status.VALID:
        if test_case.targeting_score is not None:
            relevant_info: tuple[int, list[ChoiceNode]] = (
                test_case.targeting_score,
                test_case.nodes,
            )
            if state.best_scoring is None:
                state.best_scoring = relevant_info
            else:
                best, _ = state.best_scoring
                if test_case.targeting_score > best:
                    state.best_scoring = relevant_info


TARGETING_BATCH = 10


@generation_type
def _targeting_generation(state: PbtkitState) -> None:
    """Hill climbing as a generation type.

    Each invocation picks a random index and tries to improve the
    score by adjusting it, running up to TARGETING_BATCH probes."""
    if state.result is not None or state.best_scoring is None:
        return

    def adjust(i: int, step: int) -> bool:
        """Can we improve the score by changing nodes[i] by ``step``?"""
        assert state.best_scoring is not None
        score, nodes = state.best_scoring
        if not isinstance(nodes[i].kind, IntegerChoice):
            return False
        assert nodes[i].value.bit_length() < 64
        if nodes[i].value + step < 0:
            return False
        values = [n.value for n in nodes]
        values[i] += step
        test_case = TestCase(prefix=values, random=state.random, max_size=BUFFER_SIZE)
        state.test_function(test_case)
        assert test_case.status is not None
        return (
            test_case.status >= Status.VALID
            and test_case.targeting_score is not None
            and test_case.targeting_score > score
        )

    for _ in range(TARGETING_BATCH):
        if not state.should_keep_generating():
            return
        assert state.best_scoring is not None
        i = state.random.randrange(0, len(state.best_scoring[1]))
        sign = 0
        for k in [1, -1]:
            if not state.should_keep_generating():
                return
            if adjust(i, k):
                sign = k
                break
        if sign == 0:
            continue

        k = 1
        while state.should_keep_generating() and adjust(i, sign * k):
            k *= 2

        while k > 0:
            while state.should_keep_generating() and adjust(i, sign * k):
                pass
            k //= 2
