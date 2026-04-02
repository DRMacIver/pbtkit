"""Span tracking for pbtkit.

When imported, this module monkey-patches start_span/stop_span onto
TestCase. Spans record (label, start, stop) regions over the choice
sequence, where start and stop are indices into tc.nodes.

draw() automatically wraps each generator call in a span when this
feature is enabled.
"""

from __future__ import annotations

from pbtkit.core import TestCase


def _start_span(self: TestCase, label: str) -> None:
    """Begin a labelled region at the current position in nodes."""
    self._span_stack.append((label, len(self.nodes)))


def _stop_span(self: TestCase) -> None:
    """End the most recent span and record it."""
    label, start = self._span_stack.pop()
    self.spans.append((label, start, len(self.nodes)))


TestCase.start_span = _start_span  # type: ignore[assignment]
TestCase.stop_span = _stop_span  # type: ignore[assignment]
