<div class="ai">

*The following summary was written by an LLM.*

The bytes module adds byte string generation and shrinking. A byte string is represented as a length-prefixed sequence: first a choice for the length (within the allowed bounds), then one integer choice per byte (0–255).

Shrinking tries to minimize both the length and the individual byte values. The length is reduced by the generic chunk-deletion passes operating on the underlying choice sequence, while individual bytes are reduced by the integer binary-search shrinker. This means byte strings naturally shrink toward short sequences of zero bytes.

</div>
