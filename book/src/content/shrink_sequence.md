<div class="ai">

*The following summary was written by an LLM.*

The sequence passes provide the foundational shrinking operations that work on contiguous runs of choices. The main pass deletes chunks of the choice sequence at various sizes, trying to find sub-sequences whose removal still produces a failing test. The redistribution pass tries to move value between adjacent choices within a sequence — for example, reducing one list element while increasing another — to find simpler combinations that preserve the failure.

</div>
