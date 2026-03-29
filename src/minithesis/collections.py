"""Variable-length collection support for minithesis."""

from __future__ import annotations

import math

from minithesis.minithesis import Status, TestCase


class many:
    """Utility class for drawing variable-length collections. Handles
    the "should I keep drawing more values?" logic using a geometric
    distribution with forced results for min/max constraints.

    Ported from Hypothesis's conjecture.utils.many.

    Usage:
        elements = many(test_case, min_size=0, max_size=10)
        while elements.more():
            result.append(draw_something())
    """

    def __init__(
        self,
        tc: TestCase,
        min_size: int,
        max_size: float,
    ) -> None:
        assert min_size <= max_size
        average_size = min(
            max(min_size * 2, min_size + 5),
            0.5 * (min_size + max_size),
        )
        self.tc = tc
        self.min_size = min_size
        self.max_size = max_size
        desired_extra = average_size - min_size
        max_extra = max_size - min_size
        if desired_extra >= max_extra:
            # Slightly short of 1.0 so as to not force this to always
            # be the maximum size.
            self.p_continue = 0.99
        elif math.isinf(max_size):
            # Geometric distribution with the right expected size.
            self.p_continue = 1.0 - 1.0 / (1 + desired_extra)
        else:
            # If we have a finite max size we slightly undershoot
            # with the geometric estimate. We'd rather produce
            # slightly too large than slightly too small collections,
            # and the exact probability doesn't matter very much, so
            # we bump it up slightly to compensate for the undershoot.
            self.p_continue = 1.0 - 1.0 / (2 + desired_extra)
        self.count = 0
        self.rejections = 0
        self.force_stop = False

    def more(self) -> bool:
        """Should we draw another element?"""
        if self.min_size == self.max_size:
            # Fixed size: draw exactly min_size elements.
            should_continue = self.count < self.min_size
        else:
            if self.force_stop:
                forced = False
            elif self.count < self.min_size:
                forced = True
            elif self.count >= self.max_size:
                forced = False
            else:
                forced = None
            should_continue = self.tc.weighted(self.p_continue, forced=forced)

        if should_continue:
            self.count += 1
            return True
        return False

    def reject(self) -> None:
        """Reject the last drawn element (e.g. because it was a
        duplicate). Decrements the count and may force the
        collection to stop if too many rejections occur."""
        assert self.count > 0
        self.count -= 1
        self.rejections += 1
        if self.rejections > max(3, 2 * self.count):
            if self.count < self.min_size:
                self.tc.mark_status(Status.INVALID)
            else:
                self.force_stop = True
