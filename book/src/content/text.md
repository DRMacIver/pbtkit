<div class="ai">

*The following summary was written by an LLM.*

The text module adds Unicode string generation and shrinking. Like bytes, strings are length-prefixed sequences, but each element is a codepoint drawn from the allowed range (with the surrogate range excluded).

The codepoint mapping uses a rank function that assigns lower ranks to "simpler" characters — ASCII before other BMP characters, BMP before supplementary planes — so that the integer shrinker naturally moves toward simpler strings. The surrogate gap (U+D800–U+DFFF) is handled by compacting the valid codepoint space so the shrinker doesn't waste effort on invalid values.

</div>
