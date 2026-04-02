"""Span-based mutation generation hook for pbtkit.

When imported, registers a generation hook that runs after each generated
test case.  It finds pairs of spans with the same label and tries
replacing both with the same content, enabling the engine to discover
duplicated compound structures (e.g. a list of tuples containing two
identical elements) which pure random generation is very unlikely to
produce.

Based on the crossover/mutation approach described in Aschermann et al.,
"NAUTILUS: Fishing for Deep Bugs with Grammars" (NDSS 2019), Example IV.4.
"""

from __future__ import annotations

from collections import defaultdict

from pbtkit.core import (
    BUFFER_SIZE,
    PbtkitState,
    TestCase,
    generation_hook,
)

SPAN_MUTATION_ATTEMPTS = 5


@generation_hook
def _span_mutation_hook(state: PbtkitState, base: TestCase) -> None:
    """After each generated test case, try span-swapping mutations.

    For each attempt: pick a label with 2+ spans, pick two spans,
    and replace both with the same (randomly chosen) content."""
    if not base.spans:
        return

    values = [n.value for n in base.nodes]

    # Group spans by label — only labels with 2+ spans are useful.
    by_label: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for span in base.spans:
        by_label[span[0]].append(span)
    multi = [spans for spans in by_label.values() if len(spans) >= 2]
    if not multi:
        return

    for _ in range(SPAN_MUTATION_ATTEMPTS):
        if not state.should_keep_generating():
            return
        spans = state.random.choice(multi)
        a, b = state.random.sample(sorted(spans, key=lambda s: s[1]), 2)
        # Ensure a comes before b.
        if a[1] > b[1]:
            a, b = b, a

        # Pick one span's choices as the replacement for both.
        donor = state.random.choice([a, b])
        replacement = values[donor[1] : donor[2]]

        attempt = (
            values[: a[1]]
            + replacement
            + values[a[2] : b[1]]
            + replacement
            + values[b[2] :]
        )
        tc = TestCase(prefix=tuple(attempt), random=state.random, max_size=BUFFER_SIZE)
        state.test_function(tc)
