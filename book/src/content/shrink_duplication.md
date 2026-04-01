<div class="ai">

*The following summary was written by an LLM.*

The duplication passes detect repeated structure in the choice sequence and try to exploit it. If the same sub-sequence of choices appears multiple times (e.g., because a list has duplicate elements), the shrinker can try making all copies identical by replacing them with the simplest one, or reducing the number of copies.

This is particularly effective for shrinking lists where the failure condition depends on having multiple similar elements.

</div>
