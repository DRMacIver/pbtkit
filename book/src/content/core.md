<div class="ai">

*The following summary was written by an LLM.*

The core is a self-contained property-based testing engine. It provides the fundamental loop: generate random test cases, check if they fail, and if so, shrink the failing input to a minimal reproducing example.

The representation is choice-sequence based: instead of generating structured values directly, the test function makes a series of typed choices (integers and booleans) from a `TestCase` object. A test case is fully determined by its choice sequence, and shrinking operates on that sequence rather than on the generated values.

Shrinking works by repeatedly trying to simplify the choice sequence while keeping the test interesting (failing). The basic passes try deleting chunks of choices, zeroing individual values, and binary-searching towards simpler values. This approach means shrinking is type-agnostic at the core level — it doesn't need to know what the choices represent.

</div>
