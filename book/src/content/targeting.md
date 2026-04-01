<div class="ai">

*The following summary was written by an LLM.*

The targeting module adds score-guided generation. A test can call `tc.target(score)` to report a numeric score, and the engine will bias future generation towards inputs that produce higher scores. This is useful for finding edge cases that are hard to reach by uniform random generation alone — for example, finding inputs where a computation is slow, or where a value crosses a critical threshold.

The implementation uses a hill-climbing approach: the engine tracks the best-scoring test case seen so far and mutates it to explore nearby inputs, preferring mutations that increase the score.

</div>
