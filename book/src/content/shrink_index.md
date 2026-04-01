<div class="ai">

*The following summary was written by an LLM.*

The index passes add an abstraction layer for shrinking: each choice type can define a `to_index()` / `from_index()` mapping that converts its value to a non-negative integer rank. The shrinker then operates on these indices, trying to lower them, and converts back to concrete values.

This is particularly powerful for types where the natural numeric value doesn't correspond to simplicity. For example, a float's index might order 0.0 < 1.0 < 2.0 < 0.5 < ..., so lowering the index moves toward "simpler" floats even though the raw value might increase.

The index passes also enable cross-type shrinking: the shrinker can try swapping one choice type for another if the index is compatible, allowing structural simplifications like replacing a float draw with an integer draw.

</div>
