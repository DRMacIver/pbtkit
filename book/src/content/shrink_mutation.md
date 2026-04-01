<div class="ai">

*The following summary was written by an LLM.*

The mutation pass adds random perturbation to the shrinking process. While most shrink passes are deterministic (try specific simplifications in a fixed order), the mutation pass randomly modifies individual choices and keeps the result if it's simpler.

This helps escape local minima where deterministic passes get stuck. For example, if reducing choice A only works when choice B is also changed, a random mutation might stumble on the right combination. The pass runs a bounded number of attempts, so it doesn't dominate the shrinking budget.

</div>
