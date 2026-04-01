<div class="ai">

*The following summary was written by an LLM.*

The floats module adds IEEE 754 floating-point generation and shrinking. Floats are tricky because the natural numeric ordering doesn't match the bitwise representation, and special values (NaN, infinities, negative zero) require careful handling.

Generation works by drawing a float from its full range and then clamping to the requested bounds, with special handling for NaN (drawn with a configurable probability) and infinities. The key insight for shrinking is to define a total order on floats that puts simpler values first: 0.0, then small positive integers, then other values — and to shrink within that ordering rather than the raw bit pattern.

The shrink pass uses a lexicographic encoding that maps floats to integers preserving the desired simplicity order, so the generic integer binary-search shrinker can be reused.

</div>
