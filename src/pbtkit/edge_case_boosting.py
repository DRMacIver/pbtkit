"""Edge-case boosting for pbtkit.

When imported, this module boosts the probability of generating
boundary and special values (min, max, zero) in integer and float
generation. Without it, uniform random generation rarely hits exact
boundary values, making certain counterexamples hard to find.

The boost probability is calibrated so that with 1000 draws, each
special value is drawn at least once with P > 0.999.
"""

from __future__ import annotations

# This module is a feature flag — importing it enables edge-case
# boosting in core.py and floats.py. The actual boosting logic
# lives inline in those modules.

# Probability of drawing a special value (boundary or zero) per draw.
# With k special values and n=1000 draws:
#   P(all drawn at least once) = (1 - (1-p)^n)^k
# At p=0.01, k=3: > 0.9999.
BOUNDARY_PROBABILITY = 0.01
