<div class="ai">

*The following summary was written by an LLM.*

The bind deletion pass handles a structural challenge in choice-sequence shrinking: when a test function uses `flat_map` (or equivalent), the choices made by the right-hand side depend on the value produced by the left-hand side. Deleting choices from the left side can change the boundary between the two, causing the right side's choices to be misinterpreted.

This pass identifies likely boundaries between bound segments and tries deleting chunks that respect those boundaries, allowing the shrinker to remove entire "layers" of a flatmap without corrupting the downstream choices.

</div>
