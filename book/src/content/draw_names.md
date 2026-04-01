<div class="ai">

*The following summary was written by an LLM.*

The draw names module improves the output of failing test cases. Without it, failures print generic labels like `draw_1 = 42, draw_2 = "hello"`. With it, the output uses the actual variable names from the test source code: `x = 42, name = "hello"`.

This works by rewriting the test function's source code at runtime using libcst (a concrete syntax tree library). A two-pass transform first collects all `x = tc.draw(gen)` assignments and determines whether each name is "repeatable" (used inside a loop or other nested scope), then rewrites them to `x = tc.draw_named(gen, "x", repeatable)`. The rewritten function is executed in place of the original.

Names used in loops get numbered suffixes (`x_1`, `x_2`, ...) to distinguish iterations. Names used only once at the top level are printed as-is.

</div>
