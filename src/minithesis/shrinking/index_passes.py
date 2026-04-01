"""Index-based shrink passes for minithesis.

Provides lower_and_bump and try_shortening_via_increment, which use
the to_index/from_index API on ChoiceType for type-generic shrinking.
"""

from __future__ import annotations

from minithesis.core import (
    MinithesisState,
    TestCase,
    shrink_pass,
)


def _indexed_indices(state: MinithesisState) -> list[int]:
    """Return indices of all nodes whose kind supports to_index/from_index."""
    assert state.result is not None
    return [i for i, n in enumerate(state.result) if n.kind.supports_index]


@shrink_pass
def lower_and_bump(state: MinithesisState) -> None:
    """For indexed nodes not at simplest, try decrementing (lowering the
    index) and bumping a later node (raising its index). Uses
    to_index/from_index for type-generic operation.

    Value punning in _make_choice handles the case where decrementing
    changes the type at position j (e.g., one_of branch switch)."""
    assert state.result is not None
    for gap in range(1, min(len(_indexed_indices(state)), 4)):
        idx = 0
        while idx < len(_indexed_indices(state)):
            indices = _indexed_indices(state)
            i = indices[idx]
            node_i = state.result[i]
            kind_i = node_i.kind
            # Skip nodes already at simplest.
            current_idx = kind_i.to_index(node_i.value)
            if current_idx == 0:
                idx += 1
                continue
            # Decrement targets: try simplest (index 0) and previous
            # value (index - 1). Trying simplest first handles cases
            # where intermediate steps don't produce interesting results
            # but the full decrement does (e.g. sampled_from where
            # only index 0 satisfies a downstream constraint).
            decrement_targets = []
            if current_idx > 1:
                v0 = kind_i.from_index(0)
                # from_index(0) is always the simplest — never None
                # for a kind that has any valid values.
                assert v0 is not None
                decrement_targets.append(v0)
            # from_index(current_idx - 1) can be None for bounded float
            # ranges with gaps (e.g. FloatChoice(1.0, 2.0) where
            # index 1 maps to a negative float outside the range).
            v_prev = kind_i.from_index(current_idx - 1)
            match v_prev:
                case None:  # needed_for("floats")
                    pass
                case v if v in decrement_targets:  # needed_for("floats")
                    pass
                case v:
                    decrement_targets.append(v)
            # Find the bump target: the gap'th indexed node after i.
            indices = _indexed_indices(state)
            targets_after_i = [k for k in indices if k > i]
            if gap - 1 >= len(targets_after_i):
                idx += 1
                continue
            j = targets_after_i[gap - 1]

            for new_val in decrement_targets:
                # Run the decrement to observe the kind at position j.
                attempt = list(state.result)
                attempt[i] = attempt[i].with_value(new_val)
                tc = TestCase.for_choices(
                    [n.value for n in attempt], prefix_nodes=attempt
                )
                state.test_function(tc)
                assert tc.status is not None
                # Also try the decrement with everything after i zeroed.
                zeroed = list(attempt)
                for k in range(i + 1, len(zeroed)):
                    zeroed[k] = zeroed[k].with_value(zeroed[k].kind.simplest)
                tc_z = TestCase.for_choices(
                    [n.value for n in zeroed], prefix_nodes=zeroed
                )
                state.test_function(tc_z)

                # When decrementing i changes the kind at j, try
                # zeroing to the NEW kind's simplest (from tc.nodes).
                if tc.status is not None and j < len(tc.nodes):
                    zeroed2_values = [n.value for n in state.result]
                    zeroed2_values[i] = new_val
                    for k in range(i + 1, min(len(zeroed2_values), len(tc.nodes))):
                        zeroed2_values[k] = tc.nodes[k].kind.simplest
                    tc2 = TestCase.for_choices(zeroed2_values)
                    state.test_function(tc2)

                # Try bumping the target by index offsets.
                def _try_bump_j(val: object, _new_val: object = new_val) -> bool:
                    result = state.result
                    assert result is not None
                    return (
                        j < len(result)
                        and result[j].kind.validate(val)
                        and state.replace({i: _new_val, j: val})
                    )

                if j < len(state.result) and state.result[j].kind.supports_index:
                    kind_j = state.result[j].kind
                    target_idx = kind_j.to_index(state.result[j].value)
                    for bump in [1, 2, 4]:
                        bumped = kind_j.from_index(target_idx + bump)
                        if bumped is not None:
                            if _try_bump_j(bumped):
                                break
                    p = 1
                    for _ in range(8):
                        if p > kind_j.max_index:
                            break
                        for abs_idx in [p - 1, p]:
                            _try_bump_j(kind_j.from_index(abs_idx))
                        p *= 2
            idx += 1


@shrink_pass
def try_shortening_via_increment(state: MinithesisState) -> None:
    """Try incrementing each node's index to see if the test takes a
    shorter path (e.g., triggering an earlier assertion).

    A value shrinker can only make values simpler, but sometimes making
    a value LESS simple (e.g., False→True) causes an earlier exit,
    producing a shorter and thus overall simpler choice sequence."""
    assert state.result is not None
    i = 0
    while i < len(state.result):
        node = state.result[i]
        kind = node.kind
        assert kind.supports_index
        current_idx = kind.to_index(node.value)
        # Generate candidates by incrementing the index.
        # Small relative increments plus max_index.
        candidate_indices = [current_idx + d for d in [1, 2, 4, 8, 16]]
        candidate_indices.append(kind.max_index)
        candidates = []
        for target_idx in candidate_indices:
            v = kind.from_index(target_idx)
            if v is not None and v != node.value and v not in candidates:
                candidates.append(v)
        # Also try powers of 2 (and negatives) as values directly.
        # This covers large jumps in index space that exponential
        # index probing misses (e.g. float -128.0 for a test
        # checking v < -86).
        for e in range(11):
            for sign in [1, -1]:
                for v in [sign * (1 << e), sign * float(1 << e)]:
                    if kind.validate(v) and v != node.value and v not in candidates:
                        candidates.append(v)
        if not candidates:
            i += 1
            continue
        for incremented in candidates:
            # Run the test with the incremented value and rest zeroed.
            attempt = list(state.result)
            attempt[i] = attempt[i].with_value(incremented)
            zeroed = list(attempt)
            for j in range(i + 1, len(zeroed)):
                zeroed[j] = zeroed[j].with_value(zeroed[j].kind.simplest)
            tc_zeroed = TestCase.for_choices(
                [n.value for n in zeroed], prefix_nodes=zeroed
            )
            state.test_function(tc_zeroed)
        i += 1
