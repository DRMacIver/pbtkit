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
            # Decrement: try the previous value in index order.
            new_val = kind_i.from_index(current_idx - 1)
            if new_val is None:
                # Index gap (e.g. bounded float range). Skip.
                idx += 1
                continue
            # Find the bump target: the gap'th indexed node after i.
            indices = _indexed_indices(state)
            targets_after_i = [k for k in indices if k > i]
            if gap - 1 >= len(targets_after_i):
                idx += 1
                continue
            j = targets_after_i[gap - 1]
            # Run the decrement to observe the kind at position j.
            attempt = list(state.result)
            attempt[i] = attempt[i].with_value(new_val)
            tc = TestCase.for_choices([n.value for n in attempt], prefix_nodes=attempt)
            state.test_function(tc)
            assert tc.status is not None
            # Also try the decrement with everything after i zeroed.
            zeroed = list(attempt)
            for k in range(i + 1, len(zeroed)):
                zeroed[k] = zeroed[k].with_value(zeroed[k].kind.simplest)
            tc_z = TestCase.for_choices([n.value for n in zeroed], prefix_nodes=zeroed)
            state.test_function(tc_z)

            # Try bumping the target by index offsets.
            # Re-read kind_j before each attempt since prior replaces
            # can change the result structure.
            def _try_bump_j(val: object) -> bool:
                result = state.result
                assert result is not None
                return (
                    j < len(result)
                    and result[j].kind.validate(val)
                    and state.replace({i: new_val, j: val})
                )

            if j < len(state.result) and state.result[j].kind.supports_index:
                kind_j = state.result[j].kind
                target_idx = kind_j.to_index(state.result[j].value)
                # Try index increments: powers of 2 from current position.
                bump = 1
                while bump <= 16:
                    bumped = kind_j.from_index(target_idx + bump)
                    if bumped is not None:
                        if _try_bump_j(bumped):
                            break
                    bump *= 2
                # Try simplest and exponential probe of the index range.
                for abs_idx in range(min(8, kind_j.max_index + 1)):
                    v = kind_j.from_index(abs_idx)
                    if v is not None:
                        _try_bump_j(v)
                # Also probe powers of 2 (and p-1) up to a cap.
                p = 8
                for _ in range(8):
                    if p > kind_j.max_index:
                        break
                    for idx in [p, p - 1]:
                        v = kind_j.from_index(idx)
                        if v is not None:
                            _try_bump_j(v)
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
        candidates = []
        for target_idx in [current_idx + d for d in [1, 2, 4, 8, 16]] + [
            kind.max_index
        ]:
            v = kind.from_index(target_idx)
            if v is not None and v != node.value and v not in candidates:
                candidates.append(v)
        if not candidates:
            i += 1
            continue
        for incremented in candidates:
            # Run the test with the incremented value.
            attempt = list(state.result)
            attempt[i] = attempt[i].with_value(incremented)

            # Try with rest zeroed to simplest.
            zeroed = list(attempt)
            for j in range(i + 1, len(zeroed)):
                zeroed[j] = zeroed[j].with_value(zeroed[j].kind.simplest)
            tc_zeroed = TestCase.for_choices(
                [n.value for n in zeroed], prefix_nodes=zeroed
            )
            state.test_function(tc_zeroed)
            # Try with original values (handles filters that reject simplest).
            tc_orig = TestCase.for_choices(
                [n.value for n in attempt], prefix_nodes=attempt
            )
            state.test_function(tc_orig)
            # Try setting each nearby position to unit (from_index(1)).
            for j in range(i + 1, i + 5):
                if j >= len(state.result):
                    break
                kind_j = state.result[j].kind
                assert kind_j.supports_index
                unit_val = kind_j.from_index(1)
                if unit_val is not None and j < len(zeroed):
                    filled = list(zeroed)
                    filled[j] = filled[j].with_value(unit_val)
                    tc_filled = TestCase.for_choices(
                        [n.value for n in filled], prefix_nodes=filled
                    )
                    state.test_function(tc_filled)
        i += 1
