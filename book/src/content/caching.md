<div class="ai">

*The following summary was written by an LLM.*

The caching module avoids re-running the test function on choice sequences that have already been evaluated. During shrinking, many candidate simplifications turn out to be equivalent to previously-tried sequences. Without caching, the shrinker would redundantly execute the test function on these duplicates.

The cache is keyed on the choice sequence itself (as a tuple of values). When the shrinker proposes a candidate, it checks the cache first. This is particularly valuable for shrink passes that make many small perturbations, since nearby candidates often collide.

</div>
