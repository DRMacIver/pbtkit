<div class="ai">

*The following summary was written by an LLM.*

The sorting pass tries to reorder choices so that smaller values come first. This exploits a common pattern: in many test failures, the order of elements doesn't matter, so sorting produces a lexicographically smaller (and thus "simpler") choice sequence that still triggers the failure.

The pass works by attempting to swap adjacent pairs of choice sub-sequences, keeping swaps that maintain the failure while reducing the sort key. This is essentially an insertion sort on the choice sequence, which is effective because most shrinking gains come from local reorderings.

</div>
