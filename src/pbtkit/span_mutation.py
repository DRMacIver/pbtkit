"""Span-based mutation generation type for pbtkit.

When imported, registers a generation type that generates a fresh random
test case, collects its spans, then tries replacing the contents of one
span with those of another same-labelled span.  This enables the engine
to discover duplicated compound structures (e.g. a list of tuples
containing two identical elements) which pure random generation is very
unlikely to produce.
"""

from __future__ import annotations

from collections import defaultdict

from pbtkit.core import (
    BUFFER_SIZE,
    PbtkitState,
    TestCase,
    generation_type,
)

SPAN_MUTATION_ATTEMPTS = 5


@generation_type
def span_mutation(state: PbtkitState) -> None:
    """Generate a random test case, then try span-swapping mutations on it."""
    # Generate a fresh random test case to get spans.
    base = TestCase(prefix=(), random=state.random, max_size=BUFFER_SIZE)
    state.test_function(base)
    if not state.should_keep_generating():
        return

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
        a, b = state.random.sample(spans, 2)
        # Replace span a's choices with span b's choices.
        replacement = values[b[1] : b[2]]
        attempt = values[: a[1]] + replacement + values[a[2] :]
        tc = TestCase(prefix=tuple(attempt), random=state.random, max_size=BUFFER_SIZE)
        state.test_function(tc)
