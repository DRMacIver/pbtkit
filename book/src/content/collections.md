<div class="ai">

*The following summary was written by an LLM.*

The collections module provides variable-length sequence generation via the `many()` primitive. It uses a geometric distribution controlled by boolean choices to decide when to stop adding elements, with parameters for minimum and maximum size.

The key mechanism is `Many`, which manages a draw loop: each iteration draws a boolean to decide whether to continue, then the caller draws the next element. The probability of continuing is tuned so the expected length matches a reasonable default, and the `reject()` method allows filtering elements without counting them toward the size limit.

This is the foundation for generating lists, sets, dictionaries, and other variable-length collections.

</div>
