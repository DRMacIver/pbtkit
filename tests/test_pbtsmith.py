"""Pbtsmith: a random pbtkit program generator.

Uses Hypothesis to generate random but valid pbtkit test programs
as Python source code, then executes them to verify they either succeed
or fail with Failure (never crash internally).

Ported from hegelsmith (which does the same for hegel-rust).

The generated programs exercise pbtkit's full API:
  - integers, booleans, floats, text, binary, lists
  - dependent draws (bounds derived from earlier variables)
  - if blocks with nested draws and assertions
  - tc.assume for filtering
  - genuinely falsifiable predicates (not tautologies)
  - multi-variable and compound boolean predicates
"""

from __future__ import annotations

import math  # noqa: F401  (used by exec'd generated programs)
from random import Random

import pytest

from hypothesis import HealthCheck, given, note, settings
from hypothesis import strategies as st
from pbtkit.core import Unsatisfiable, run_test

# Bare names needed by exec'd generated programs (exec uses globals()).
# These tests exercise all types, so skip the module if any type is disabled.
try:
    from pbtkit.generators import (  # noqa: F401
        binary,
        booleans,
        composite,
        dictionaries,
        floats,
        integers,
        just,
        lists,
        nothing,
        one_of,
        sampled_from,
        text,
        tuples,
    )
except (ImportError, NotImplementedError):
    pytest.skip("requires all generator types", allow_module_level=True)


class Failure(Exception):
    pass


# Tree-walking helpers available to exec'd programs via globals().
def tree_depth(node):  # noqa: F401
    """Return the depth of a tree node (tuple = branch, else = leaf)."""
    if not isinstance(node, tuple) or len(node) == 0:
        return 0
    return 1 + max(tree_depth(child) for child in node[1:])


def tree_size(node):  # noqa: F401
    """Return the number of nodes in a tree."""
    if not isinstance(node, tuple) or len(node) == 0:
        return 1
    return 1 + sum(tree_size(child) for child in node[1:])


def tree_leaves(node):  # noqa: F401
    """Return the number of leaf nodes in a tree."""
    if not isinstance(node, tuple) or len(node) == 0:
        return 1
    return sum(tree_leaves(child) for child in node[1:])


def tree_labels(node):  # noqa: F401
    """Return the set of operator labels used in a tree."""
    if not isinstance(node, tuple) or len(node) == 0:
        return set()
    result = {node[0]}
    for child in node[1:]:
        result |= tree_labels(child)
    return result


def tree_leaf_values(node):  # noqa: F401
    """Return a list of all leaf values in a tree."""
    if not isinstance(node, tuple) or len(node) == 0:
        return [node]
    result = []
    for child in node[1:]:
        result.extend(tree_leaf_values(child))
    return result


def tree_nodes(node):  # noqa: F401
    """Return a list of all nodes in a tree (branches and leaves), pre-order."""
    result = [node]
    if isinstance(node, tuple) and len(node) > 0:
        for child in node[1:]:
            result.extend(tree_nodes(child))
    return result


# ---------------------------------------------------------------------------
# Python type tags and environment tracking
# ---------------------------------------------------------------------------

TYPE_NAMES = [
    "bool",
    "int",
    "float",
    "str",
    "bytes",
    "list_int",
    "list_bool",
    "list_tuple",
]


class VarInfo:
    """A variable in scope."""

    __slots__ = ("name", "typ", "lo", "hi")

    def __init__(self, name: str, typ: str, lo: int = 0, hi: int = 0) -> None:
        self.name = name
        self.typ = typ
        self.lo = lo
        self.hi = hi


class Env:
    """Track variables in scope during program generation."""

    def __init__(self) -> None:
        self.vars: list[VarInfo] = []
        self.next_id = 0

    def fresh_var(self) -> str:
        name = f"v{self.next_id}"
        self.next_id += 1
        return name

    def add(self, name: str, typ: str, lo: int = 0, hi: int = 0) -> VarInfo:
        v = VarInfo(name, typ, lo, hi)
        self.vars.append(v)
        return v

    def save(self) -> int:
        return len(self.vars)

    def restore(self, point: int) -> None:
        del self.vars[point:]

    def of_type(self, *typs: str) -> list[VarInfo]:
        return [v for v in self.vars if v.typ in typs]

    def int_vars(self) -> list[VarInfo]:
        return self.of_type("int")

    def bool_vars(self) -> list[VarInfo]:
        return self.of_type("bool")

    def float_vars(self) -> list[VarInfo]:
        return self.of_type("float")

    def str_vars(self) -> list[VarInfo]:
        return self.of_type("str")

    def bytes_vars(self) -> list[VarInfo]:
        return self.of_type("bytes")

    def collection_vars(self) -> list[VarInfo]:
        return self.of_type(
            "list_int",
            "list_bool",
            "list_tuple",
            "list_tree",
            "list_tree_nodes",
            "bytes",
            "dict",
            "tuple",
        )

    def tree_vars(self) -> list[VarInfo]:
        return self.of_type("tree")

    def has_vars(self) -> bool:
        return len(self.vars) > 0


# ---------------------------------------------------------------------------
# Generator expression rendering
# ---------------------------------------------------------------------------


def gen_expr_code(typ: str, int_lo: int, int_hi: int, *, wide: bool = False) -> str:
    """Return source code for a pbtkit generator expression.

    When *wide* is True, use broader ranges: floats allow infinity,
    text covers full Unicode, and integers use a much larger range.
    """
    if typ == "bool":
        return "booleans()"
    elif typ == "int":
        if wide:
            return f"integers({int_lo}, {int_hi})"
        return f"integers({int_lo}, {int_hi})"
    elif typ == "float":
        if wide:
            return "floats()"
        return "floats(allow_nan=False, allow_infinity=False)"
    elif typ == "str":
        if wide:
            return "text(max_size=20)"
        return "text(min_codepoint=32, max_codepoint=126, max_size=20)"
    elif typ == "bytes":
        return "binary(max_size=20)"
    elif typ == "list_int":
        if wide:
            return f"lists(integers({int_lo}, {int_hi}), max_size=10)"
        return f"lists(integers({int_lo}, {int_hi}), max_size=10)"
    elif typ == "list_bool":
        return "lists(booleans(), max_size=10)"
    elif typ == "list_tuple":
        if wide:
            return f"lists(tuples(integers({int_lo}, {int_hi}), integers({int_lo}, {int_hi})), max_size=10)"
        return f"lists(tuples(integers({int_lo}, {int_hi}), integers({int_lo}, {int_hi})), max_size=10)"
    else:
        raise ValueError(f"Unknown type: {typ}")


# ---------------------------------------------------------------------------
# Predicate generation -- genuinely falsifiable assertions
# ---------------------------------------------------------------------------

# Each gen_*_predicate strategy returns a string like "v0 > 0" that
# is genuinely falsifiable (could be False for some drawn value).


@st.composite
def gen_int_predicate(draw: st.DrawFn, name: str) -> str:
    """Falsifiable predicates for int variables."""
    choice = draw(st.integers(0, 6))
    if choice == 0:
        return f"{name} > 0"
    elif choice == 1:
        t = draw(st.integers(-100, 100))
        return f"{name} < {t}"
    elif choice == 2:
        n = draw(st.integers(2, 10))
        return f"{name} % {n} == 0"
    elif choice == 3:
        return f"{name} >= 0"
    elif choice == 4:
        val = draw(st.integers(-10, 10))
        return f"{name} != {val}"
    elif choice == 5:
        return f"abs({name}) < 50"
    else:
        return f"{name} * {name} > {name}"


@st.composite
def gen_float_predicate(draw: st.DrawFn, name: str) -> str:
    """Falsifiable predicates for float variables."""
    choice = draw(st.integers(0, 6))
    if choice == 0:
        return f"{name} > 0.0"
    elif choice == 1:
        t = draw(st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False))
        return f"{name} < {t}"
    elif choice == 2:
        return f"abs({name}) < 1.0"
    elif choice == 3:
        return f"{name} >= 0.0"
    elif choice == 4:
        # Exercises infinity detection
        return f"not math.isinf({name})"
    elif choice == 5:
        # Exercises NaN detection
        return f"not math.isnan({name})"
    else:
        # Exercises finiteness
        return f"math.isfinite({name})"


@st.composite
def gen_bool_predicate(draw: st.DrawFn, name: str) -> str:
    """Falsifiable predicates for bool variables."""
    if draw(st.booleans()):
        return name
    else:
        return f"not {name}"


@st.composite
def gen_string_predicate(draw: st.DrawFn, name: str) -> str:
    """Falsifiable predicates for str variables."""
    choice = draw(st.integers(0, 7))
    if choice == 0:
        return f"len({name}) == 0"
    elif choice == 1:
        return f"len({name}) > 0"
    elif choice == 2:
        t = draw(st.integers(1, 20))
        return f"len({name}) < {t}"
    elif choice == 3:
        c = draw(st.sampled_from(["a", "e", " ", "0", "A"]))
        return f"'{c}' in {name}"
    elif choice == 6:
        # All unique characters — exercises duplicate finding
        return f"len(set({name})) == len({name})"
    elif choice == 7:
        # ASCII-encodable — exercises non-ASCII finding
        return f"{name}.isascii()"
    elif choice == 4:
        return f"{name}.isascii()"
    else:
        prefix = draw(st.sampled_from(["a", "the", "0"]))
        return f"{name}.startswith('{prefix}')"


@st.composite
def gen_bytes_predicate(draw: st.DrawFn, name: str) -> str:
    """Falsifiable predicates for bytes variables."""
    choice = draw(st.integers(0, 3))
    if choice == 0:
        return f"len({name}) == 0"
    elif choice == 1:
        return f"len({name}) > 0"
    elif choice == 2:
        t = draw(st.integers(1, 10))
        return f"len({name}) < {t}"
    else:
        return f"len({name}) > 3"


@st.composite
def gen_collection_predicate(draw: st.DrawFn, var: VarInfo) -> str:
    """Falsifiable predicates for list/bytes variables."""
    name = var.name
    choice = draw(st.integers(0, 5))
    if choice == 0:
        return f"len({name}) == 0"
    elif choice == 1:
        return f"len({name}) > 0"
    elif choice == 2:
        t = draw(st.integers(1, 10))
        return f"len({name}) < {t}"
    elif choice == 3:
        t = draw(st.integers(1, 5))
        return f"len({name}) > {t}"
    elif choice == 4:
        t = draw(st.integers(0, 5))
        return f"len({name}) == {t}"
    elif var.typ in ("list_int", "list_bool", "list_tuple", "list_tree_nodes"):
        # No duplicate elements — requires finding two identical compound values.
        return f"len({name}) == len(set({name}))"
    else:
        return f"len({name}) > 0"


@st.composite
def gen_predicate_for_var(draw: st.DrawFn, var: VarInfo) -> str:
    """Generate a falsifiable predicate for a single variable."""
    if var.typ == "int":
        return draw(gen_int_predicate(var.name))
    elif var.typ == "float":
        return draw(gen_float_predicate(var.name))
    elif var.typ == "bool":
        return draw(gen_bool_predicate(var.name))
    elif var.typ == "str":
        return draw(gen_string_predicate(var.name))
    elif var.typ == "bytes":
        return draw(gen_bytes_predicate(var.name))
    elif var.typ in ("list_int", "list_bool", "list_tuple", "dict", "tuple"):
        return draw(gen_collection_predicate(var))
    elif var.typ == "tree":
        return draw(gen_tree_predicate(var.name))
    elif var.typ in ("list_tree_nodes", "list_tree"):
        return draw(gen_tree_nodes_predicate(var.name))
    else:
        raise ValueError(f"Unknown type: {var.typ}")


# ---------------------------------------------------------------------------
# Multi-variable predicates
# ---------------------------------------------------------------------------


@st.composite
def gen_multi_value_predicate(draw: st.DrawFn, env: Env) -> str:
    """Generate a predicate involving two or more variables."""
    int_vars = env.int_vars()
    if len(int_vars) >= 2:
        a = draw(st.sampled_from(int_vars))
        b = draw(st.sampled_from([v for v in int_vars if v.name != a.name]))
        # Check if both variables span zero — if so, allow negative sum predicates
        can_negative_sum = (
            a.lo < 0
            and a.hi > 0
            and b.lo < 0
            and b.hi > 0
            and a.lo + b.lo < -max(a.hi, b.hi)
        )
        choice = draw(st.integers(0, 6))
        if choice == 0:
            return f"{a.name} + {b.name} > 0"
        elif choice == 1:
            return f"{a.name} < {b.name}"
        elif choice == 2:
            return f"{a.name} == {b.name}"
        elif choice == 3:
            return f"{a.name} != {b.name}"
        elif choice == 4:
            return f"{a.name} + {b.name} == {b.name} + {a.name}"
        elif choice == 5:
            return f"{a.name} - {b.name} > 0"
        elif can_negative_sum:
            threshold = max(a.hi, b.hi)
            return f"{a.name} + {b.name} >= -{threshold}"
        else:
            return f"{a.name} + {b.name} > 0"
    elif len(int_vars) == 1:
        a = int_vars[0]
        choice = draw(st.integers(0, 2))
        if choice == 0:
            return f"{a.name} + {a.name} > 0"
        elif choice == 1:
            return f"{a.name} * 2 == {a.name} + {a.name}"
        else:
            return f"{a.name} - {a.name} == 0"

    str_vars = env.str_vars()
    if len(str_vars) >= 2:
        a = draw(st.sampled_from(str_vars))
        b = draw(st.sampled_from([v for v in str_vars if v.name != a.name]))
        choice = draw(st.integers(0, 2))
        if choice == 0:
            return f"len({a.name}) < len({b.name})"
        elif choice == 1:
            return f"{a.name} == {b.name}"
        else:
            return f"len({a.name}) + len({b.name}) < 30"

    col_vars = env.collection_vars()
    if len(col_vars) >= 2:
        a = draw(st.sampled_from(col_vars))
        b = draw(st.sampled_from([v for v in col_vars if v.name != a.name]))
        choice = draw(st.integers(0, 1))
        if choice == 0:
            return f"len({a.name}) == len({b.name})"
        else:
            return f"len({a.name}) + len({b.name}) < 20"

    tree_vars = env.tree_vars()
    if len(tree_vars) >= 2:
        a = draw(st.sampled_from(tree_vars))
        b = draw(st.sampled_from([v for v in tree_vars if v.name != a.name]))
        choice = draw(st.integers(0, 2))
        if choice == 0:
            return f"tree_depth({a.name}) == tree_depth({b.name})"
        elif choice == 1:
            return f"tree_size({a.name}) < tree_size({b.name})"
        else:
            return f"tree_labels({a.name}) == tree_labels({b.name})"

    if tree_vars and int_vars:
        t = draw(st.sampled_from(tree_vars))
        iv = draw(st.sampled_from(int_vars))
        choice = draw(st.integers(0, 1))
        if choice == 0:
            return f"tree_depth({t.name}) < abs({iv.name}) + 1"
        else:
            return f"tree_size({t.name}) < abs({iv.name}) + 1"

    # Cross-type: int and collection — containment, count, etc.
    if int_vars and col_vars:
        iv = draw(st.sampled_from(int_vars))
        cv = draw(st.sampled_from(col_vars))
        choice = draw(st.integers(0, 3))
        if choice == 0:
            return f"abs({iv.name}) < len({cv.name}) + 1"
        elif choice == 1 and cv.typ == "list_int":
            return f"{iv.name} in {cv.name}"
        elif choice == 2 and cv.typ == "list_int":
            return f"{cv.name}.count({iv.name}) > 1"
        else:
            return f"abs({iv.name}) < len({cv.name}) + 1"

    # Ultimate fallback: single-variable predicate
    var = draw(st.sampled_from(env.vars))
    return draw(gen_predicate_for_var(var))


# ---------------------------------------------------------------------------
# Compound boolean predicates
# ---------------------------------------------------------------------------


@st.composite
def gen_compound_predicate(draw: st.DrawFn, env: Env, depth: int = 0) -> str:
    """Generate boolean combinations of predicates."""
    if depth >= 2:
        var = draw(st.sampled_from(env.vars))
        return draw(gen_predicate_for_var(var))

    choice = draw(st.integers(0, 4))
    if choice == 0:
        # p1 and p2
        p1 = draw(gen_leaf_or_compound(env, depth + 1))
        p2 = draw(gen_leaf_or_compound(env, depth + 1))
        return f"({p1}) and ({p2})"
    elif choice == 1:
        # p1 or p2
        p1 = draw(gen_leaf_or_compound(env, depth + 1))
        p2 = draw(gen_leaf_or_compound(env, depth + 1))
        return f"({p1}) or ({p2})"
    elif choice == 2:
        # not p
        var = draw(st.sampled_from(env.vars))
        p = draw(gen_predicate_for_var(var))
        return f"not ({p})"
    elif choice == 3:
        # p1 and not p2
        var1 = draw(st.sampled_from(env.vars))
        var2 = draw(st.sampled_from(env.vars))
        p1 = draw(gen_predicate_for_var(var1))
        p2 = draw(gen_predicate_for_var(var2))
        return f"({p1}) and not ({p2})"
    else:
        # (p1 or p2) and p3
        var1 = draw(st.sampled_from(env.vars))
        var2 = draw(st.sampled_from(env.vars))
        var3 = draw(st.sampled_from(env.vars))
        p1 = draw(gen_predicate_for_var(var1))
        p2 = draw(gen_predicate_for_var(var2))
        p3 = draw(gen_predicate_for_var(var3))
        return f"(({p1}) or ({p2})) and ({p3})"


@st.composite
def gen_leaf_or_compound(draw: st.DrawFn, env: Env, depth: int) -> str:
    if draw(st.booleans()):
        var = draw(st.sampled_from(env.vars))
        return draw(gen_predicate_for_var(var))
    else:
        return draw(gen_compound_predicate(env, depth))


# ---------------------------------------------------------------------------
# Computed-value predicates
# ---------------------------------------------------------------------------


@st.composite
def gen_computed_predicate(draw: st.DrawFn, env: Env) -> str:
    """Generate predicates over computed values (sums, lengths, etc.)."""
    list_int_vars = env.of_type("list_int")
    if list_int_vars:
        var = draw(st.sampled_from(list_int_vars))
        n = var.name
        choice = draw(st.integers(0, 3))
        if choice == 0:
            return f"sum({n}) > 0"
        elif choice == 1:
            return f"all(x > 0 for x in {n})"
        elif choice == 2:
            return f"any(x == 0 for x in {n})"
        else:
            return f"len({n}) == 0 or {n}[0] > 0"

    int_vars = env.int_vars()
    if len(int_vars) >= 2:
        a = draw(st.sampled_from(int_vars))
        b = draw(st.sampled_from([v for v in int_vars if v.name != a.name]))
        choice = draw(st.integers(0, 2))
        if choice == 0:
            return f"{a.name} + {b.name} - {b.name} == {a.name}"
        elif choice == 1:
            return f"{a.name} + {b.name} > {a.name}"
        else:
            return f"abs({a.name}) + abs({b.name}) < 1000"

    str_vars = env.str_vars()
    if str_vars:
        var = draw(st.sampled_from(str_vars))
        n = var.name
        choice = draw(st.integers(0, 2))
        if choice == 0:
            return f"len({n}) == len(list({n}))"
        elif choice == 1:
            return f"len({n}.upper()) >= len({n})"
        else:
            return f"len({n}.strip()) <= len({n})"

    list_tree_vars = env.of_type("list_tree")
    if list_tree_vars:
        var = draw(st.sampled_from(list_tree_vars))
        n = var.name
        choice = draw(st.integers(0, 2))
        if choice == 0:
            return f"all(tree_depth(t) < 3 for t in {n})"
        elif choice == 1:
            return f"sum(tree_size(t) for t in {n}) < 20"
        else:
            return f"len(set(tuple(tree_labels(t)) for t in {n})) <= 1"

    tree_vars = env.tree_vars()
    if tree_vars:
        var = draw(st.sampled_from(tree_vars))
        n = var.name
        choice = draw(st.integers(0, 6))
        if choice == 0:
            # depth vs size relationship
            return f"tree_depth({n}) <= tree_size({n})"
        elif choice == 1:
            # leaves vs size
            return f"tree_leaves({n}) <= tree_size({n})"
        elif choice == 2:
            # depth of left vs right subtree (if binary)
            return (
                f"not isinstance({n}, tuple) or len({n}) != 3"
                f" or abs(tree_depth({n}[1]) - tree_depth({n}[2])) < 3"
            )
        elif choice == 3:
            # tree has at least some structure
            return f"tree_depth({n}) > 0 or tree_size({n}) == 1"
        elif choice == 4:
            # sum of integer leaf values
            t = draw(st.integers(5, 30))
            return f"sum(v for v in tree_leaf_values({n}) if isinstance(v, int)) < {t}"
        elif choice == 5:
            # all leaves are the same value
            return (
                f"len(set(v for v in tree_leaf_values({n}) if isinstance(v, int))) <= 1"
            )
        else:
            # max leaf value
            return (
                f"max((v for v in tree_leaf_values({n})"
                f" if isinstance(v, int)), default=0) < 5"
            )

    # Fallback
    var = draw(st.sampled_from(env.vars))
    return draw(gen_predicate_for_var(var))


# ---------------------------------------------------------------------------
# Rich assertion generation
# ---------------------------------------------------------------------------


@st.composite
def gen_rich_assertion(draw: st.DrawFn, env: Env) -> str:
    """Generate a non-trivial assertion statement (one source line)."""
    choice = draw(st.integers(0, 3))
    if choice == 0:
        var = draw(st.sampled_from(env.vars))
        cond = draw(gen_predicate_for_var(var))
    elif choice == 1:
        cond = draw(gen_multi_value_predicate(env))
    elif choice == 2:
        cond = draw(gen_compound_predicate(env))
    else:
        cond = draw(gen_computed_predicate(env))
    return f"if not ({cond}): raise Failure({cond!r})"


# ---------------------------------------------------------------------------
# Draw statements
# ---------------------------------------------------------------------------


@st.composite
def gen_base_generator_expr(draw: st.DrawFn) -> tuple[str, str, int, int]:
    """Generate a (typ, expr, lo, hi) for a base generator."""
    typ = draw(st.sampled_from(TYPE_NAMES))
    wide = draw(st.integers(0, 4)) == 0  # 20% chance of wide mode
    if wide and typ == "int":
        # Sometimes use much larger integer ranges
        int_lo = draw(st.integers(-(2**63), 0))
        int_hi = draw(st.integers(0, 2**63 - 1))
    else:
        int_lo = draw(st.integers(-100, 100))
        int_hi = draw(st.integers(int_lo, int_lo + 200))
    return (typ, gen_expr_code(typ, int_lo, int_hi, wide=wide), int_lo, int_hi)


@st.composite
def gen_generator_expr(draw: st.DrawFn, depth: int = 0) -> tuple[str, str, int, int]:
    """Generate a (typ, expr, lo, hi) for any generator expression.

    depth limits recursion for nested generators."""
    if depth >= 2:
        return draw(gen_base_generator_expr())

    choice = draw(st.integers(0, 15))
    if choice <= 6:
        # Base generators (majority of draws)
        return draw(gen_base_generator_expr())
    elif choice == 7:
        # just(value)
        val_choice = draw(st.integers(0, 3))
        if val_choice == 0:
            v = draw(st.integers(-100, 100))
            return ("int", f"just({v})", v, v)
        elif val_choice == 1:
            return ("bool", f"just({draw(st.booleans())})", 0, 0)
        elif val_choice == 2:
            s = draw(st.text(min_size=0, max_size=5, alphabet="abcde "))
            return ("str", f"just({s!r})", 0, 0)
        else:
            return (
                "float",
                f"just({draw(st.floats(-100, 100, allow_nan=False))})",
                0,
                0,
            )
    elif choice == 8:
        # sampled_from(elements)
        val_choice = draw(st.integers(0, 1))
        if val_choice == 0:
            elems = draw(st.lists(st.integers(-50, 50), min_size=2, max_size=6))
            return ("int", f"sampled_from({elems!r})", min(elems), max(elems))
        else:
            elems = draw(
                st.lists(
                    st.text(min_size=0, max_size=3, alphabet="abc"),
                    min_size=2,
                    max_size=5,
                )
            )
            return ("str", f"sampled_from({elems!r})", 0, 0)
    elif choice == 9:
        # one_of(*generators)
        heterogeneous = draw(st.booleans())
        if heterogeneous:
            # Heterogeneous one_of — different types per branch.
            # Use "bool" as the result type since truthiness predicates
            # (v0, not v0) are safe for any Python type, and bool vars
            # aren't used in type-specific dependent draws.
            sub_exprs = []
            for _ in range(draw(st.integers(2, 4))):
                _, sub_expr, _, _ = draw(gen_generator_expr(depth + 1))
                sub_exprs.append(sub_expr)
            return ("bool", f"one_of({', '.join(sub_exprs)})", 0, 0)
        else:
            # Homogeneous one_of — same output type
            base_typ = draw(st.sampled_from(["int", "bool", "str", "float"]))
            n = draw(st.integers(2, 4))
            sub_exprs = []
            lo_min, hi_max = 0, 0
            for _ in range(n):
                _, sub_expr, lo, hi = draw(gen_generator_expr(depth + 1))
                # Override to ensure same type
                if base_typ == "int":
                    lo = draw(st.integers(-100, 100))
                    hi = draw(st.integers(lo, lo + 200))
                    sub_expr = f"integers({lo}, {hi})"
                    lo_min = min(lo_min, lo) if sub_exprs else lo
                    hi_max = max(hi_max, hi) if sub_exprs else hi
                elif base_typ == "bool":
                    sub_expr = "booleans()"
                elif base_typ == "str":
                    sub_expr = "text(min_codepoint=32, max_codepoint=126, max_size=10)"
                else:
                    sub_expr = "floats(allow_nan=False, allow_infinity=False)"
                sub_exprs.append(sub_expr)
            # Sometimes include nothing() as one branch
            if draw(st.integers(0, 4)) == 0:
                sub_exprs.append("nothing()")
            return (base_typ, f"one_of({', '.join(sub_exprs)})", lo_min, hi_max)
    elif choice == 10:
        # dictionaries(keys, values)
        key_typ = draw(st.sampled_from(["int", "str"]))
        if key_typ == "int":
            lo = draw(st.integers(-50, 50))
            hi = draw(st.integers(lo, lo + 100))
            key_expr = f"integers({lo}, {hi})"
        else:
            key_expr = "text(min_codepoint=32, max_codepoint=126, max_size=5)"
        _, val_expr, _, _ = draw(gen_generator_expr(depth + 1))
        return ("dict", f"dictionaries({key_expr}, {val_expr}, max_size=5)", 0, 0)
    elif choice == 11:
        # tuples(*generators)
        n = draw(st.integers(2, 4))
        sub_exprs = []
        for _ in range(n):
            _, sub_expr, _, _ = draw(gen_generator_expr(depth + 1))
            sub_exprs.append(sub_expr)
        return ("tuple", f"tuples({', '.join(sub_exprs)})", 0, 0)
    elif choice == 12:
        # lists(integers(...), unique=True) — need wide enough range
        lo = draw(st.integers(-100, 0))
        hi = draw(st.integers(lo + 20, lo + 200))
        return (
            "list_int",
            f"lists(integers({lo}, {hi}), max_size=10, unique=True)",
            lo,
            hi,
        )
    elif choice == 13:
        # .filter on a base generator
        typ, expr, lo, hi = draw(gen_base_generator_expr())
        filt = _filter_for_type(draw, typ)
        if filt is not None:
            return (typ, f"{expr}.filter({filt})", lo, hi)
        return (typ, expr, lo, hi)
    elif choice == 14:
        # .map on a base generator
        typ, expr, lo, hi = draw(gen_base_generator_expr())
        mapped = _map_for_type(draw, typ)
        if mapped is not None:
            out_typ, map_fn = mapped
            return (out_typ, f"{expr}.map({map_fn})", 0, 0)
        return (typ, expr, lo, hi)
    else:
        # .flat_map on a base generator
        typ, expr, lo, hi = draw(gen_base_generator_expr())
        fm = _flat_map_for_type(draw, typ)
        if fm is not None:
            out_typ, fm_fn = fm
            return (out_typ, f"{expr}.flat_map({fm_fn})", 0, 0)
        return (typ, expr, lo, hi)


def _filter_for_type(draw: st.DrawFn, typ: str) -> str | None:
    """Return a lambda string for filtering values of the given type."""
    if typ == "int":
        return draw(
            st.sampled_from(
                ["lambda x: x > 0", "lambda x: x % 2 == 0", "lambda x: x != 0"]
            )
        )
    if typ == "float":
        return "lambda x: x >= 0.0"
    if typ == "str":
        return draw(st.sampled_from(["lambda x: len(x) > 0", "lambda x: len(x) < 10"]))
    if typ == "bytes":
        return "lambda x: len(x) > 0"
    if typ in ("list_int", "list_bool"):
        return draw(st.sampled_from(["lambda x: len(x) > 0", "lambda x: len(x) < 8"]))
    return None


def _map_for_type(draw: st.DrawFn, typ: str) -> tuple[str, str] | None:
    """Return (output_type, lambda_string) for mapping values of the given type."""
    if typ == "int":
        return draw(
            st.sampled_from(
                [
                    ("int", "lambda x: abs(x)"),
                    ("int", "lambda x: x * 2"),
                    ("bool", "lambda x: x > 0"),
                    ("str", "lambda x: str(x)"),
                ]
            )
        )
    if typ == "str":
        return draw(
            st.sampled_from(
                [
                    ("int", "lambda x: len(x)"),
                    ("str", "lambda x: x.upper()"),
                ]
            )
        )
    if typ in ("list_int", "list_bool"):
        return ("int", "lambda x: len(x)")
    if typ == "bool":
        return ("int", "lambda x: int(x)")
    return None


def _flat_map_for_type(draw: st.DrawFn, typ: str) -> tuple[str, str] | None:
    """Return (output_type, lambda_string) for flat_map on the given type."""
    if typ == "int":
        return draw(
            st.sampled_from(
                [
                    ("int", "lambda x: integers(0, abs(x) + 1)"),
                    (
                        "str",
                        "lambda x: text(min_codepoint=32, max_codepoint=126,"
                        " max_size=abs(x) % 10 + 1)",
                    ),
                    (
                        "list_bool",
                        "lambda x: lists(booleans(), max_size=abs(x) % 5 + 1)",
                    ),
                    # Length-controlling: the int value determines the exact
                    # list size, creating a strong structural dependency.
                    (
                        "list_int",
                        "lambda n: lists(integers(0, 100),"
                        " min_size=abs(n) % 5, max_size=abs(n) % 5 + 1)",
                    ),
                ]
            )
        )
    if typ == "bool":
        return draw(
            st.sampled_from(
                [
                    ("int", "lambda x: integers(0, 10) if x else integers(-10, 0)"),
                    # Branch on bool to produce different types — exercises
                    # type changes during shrinking.
                    (
                        "str",
                        "lambda x: text(min_codepoint=32, max_codepoint=126, max_size=5)"
                        " if x else just('')",
                    ),
                ]
            )
        )
    return None


@st.composite
def gen_tree_dependent_draw(draw: st.DrawFn, env: Env) -> str:
    """Generate a draw whose bounds depend on an existing tree variable."""
    tree_vars = env.tree_vars()
    if not tree_vars:
        _, code = draw(gen_draw(env))
        return code

    source = draw(st.sampled_from(tree_vars))
    var = env.fresh_var()
    src = source.name

    choice = draw(st.integers(0, 2))
    if choice == 0:
        # Integer bounded by tree depth
        code = f"{var} = tc.draw(integers(0, tree_depth({src}) + 1))"
        env.add(var, "int")
    elif choice == 1:
        # Integer bounded by tree size
        code = f"{var} = tc.draw(integers(0, tree_size({src})))"
        env.add(var, "int")
    else:
        # List bounded by tree leaf count
        code = f"{var} = tc.draw(lists(booleans(), max_size=tree_leaves({src}) + 1))"
        env.add(var, "list_bool")
    return code


@st.composite
def gen_draw(draw: st.DrawFn, env: Env) -> tuple[str, str]:
    """Generate a simple draw statement. Returns (type, code_line)."""
    typ, expr, lo, hi = draw(gen_generator_expr())
    var = env.fresh_var()
    code = f"{var} = tc.draw({expr})"
    env.add(var, typ, lo, hi)
    return (typ, code)


@st.composite
def gen_dependent_draw(draw: st.DrawFn, env: Env) -> str:
    """Generate a draw whose bounds depend on an existing int variable."""
    int_vars = env.int_vars()
    if not int_vars:
        _, code = draw(gen_draw(env))
        return code

    source = draw(st.sampled_from(int_vars))
    var = env.fresh_var()
    src = source.name

    choice = draw(st.integers(0, 3))
    if choice == 0:
        # Integer with dependent bounds: integers(src, src + delta)
        delta = draw(st.integers(1, 50))
        code = f"{var} = tc.draw(integers({src}, {src} + {delta}))"
        env.add(var, "int")
    elif choice == 1:
        # Integer with dependent bounds: integers(-abs(src), abs(src))
        code = f"{var} = tc.draw(integers(max(-abs({src}) - 1, -(2**63)), min(abs({src}) + 1, 2**63 - 1)))"
        env.add(var, "int")
    elif choice == 2:
        # List with dependent max_size
        code = f"{var} = tc.draw(lists(booleans(), max_size=abs({src}) % 10 + 1))"
        env.add(var, "list_bool")
    else:
        # Text with dependent max_size
        code = (
            f"{var} = tc.draw(text(min_codepoint=32, max_codepoint=126, "
            f"max_size=abs({src}) % 20 + 1))"
        )
        env.add(var, "str")
    return code


@st.composite
def gen_draw_or_dependent(draw: st.DrawFn, env: Env) -> str:
    """Generate a draw, possibly dependent on an existing variable."""
    r = draw(st.integers(0, 9))
    int_vars = env.int_vars()
    tree_vars = env.tree_vars()
    if int_vars and r >= 6:
        return draw(gen_dependent_draw(env))
    if tree_vars and r >= 4:
        return draw(gen_tree_dependent_draw(env))
    _, code = draw(gen_draw(env))
    return code


# ---------------------------------------------------------------------------
# Assume statements
# ---------------------------------------------------------------------------


@st.composite
def gen_assume(draw: st.DrawFn, env: Env) -> str:
    """Generate a tc.assume(...) call."""
    int_vars = env.int_vars()
    col_vars = env.collection_vars()
    str_vars = env.str_vars()

    options: list[str] = []
    if int_vars:
        options.append("int")
    if col_vars:
        options.append("col")
    if str_vars:
        options.append("str")

    if not options:
        return "tc.assume(True)"

    which = draw(st.sampled_from(options))
    if which == "int":
        var = draw(st.sampled_from(int_vars))
        return f"tc.assume(abs({var.name}) < 10000)"
    elif which == "col":
        var = draw(st.sampled_from(col_vars))
        return f"tc.assume(len({var.name}) < 100)"
    else:
        var = draw(st.sampled_from(str_vars))
        return f"tc.assume(len({var.name}) < 100)"


# ---------------------------------------------------------------------------
# If blocks
# ---------------------------------------------------------------------------


@st.composite
def gen_if_condition(draw: st.DrawFn, env: Env) -> str:
    """Generate a condition for an if block."""
    choice = draw(st.integers(0, 4))
    if choice == 0:
        bvars = env.bool_vars()
        if bvars:
            var = draw(st.sampled_from(bvars))
            return var.name
    elif choice == 1:
        bvars = env.bool_vars()
        if bvars:
            var = draw(st.sampled_from(bvars))
            return f"not {var.name}"
    elif choice == 2:
        ivars = env.int_vars()
        if ivars:
            var = draw(st.sampled_from(ivars))
            return draw(gen_int_predicate(var.name))
    elif choice == 3:
        cvars = env.collection_vars()
        if cvars:
            var = draw(st.sampled_from(cvars))
            return f"len({var.name}) > 0"

    # Fallback: any variable predicate
    var = draw(st.sampled_from(env.vars))
    return draw(gen_predicate_for_var(var))


@st.composite
def gen_block_body(
    draw: st.DrawFn, env: Env, indent: str, block_depth: int
) -> list[str]:
    """Generate the body of an if/else block."""
    save = env.save()
    lines: list[str] = []

    num_stmts = draw(st.integers(1, 3))
    for _ in range(num_stmts):
        choice = draw(st.integers(0, 4))
        if choice <= 1:
            code = draw(gen_draw_or_dependent(env))
            lines.append(f"{indent}{code}")
        elif choice <= 3:
            if env.has_vars():
                assertion = draw(gen_rich_assertion(env))
                lines.append(f"{indent}{assertion}")
            else:
                _, code = draw(gen_draw(env))
                lines.append(f"{indent}{code}")
        else:
            if block_depth < 2 and env.has_vars():
                if_lines = draw(gen_if_block(env, indent, block_depth + 1))
                lines.extend(if_lines)
            elif env.has_vars():
                assertion = draw(gen_rich_assertion(env))
                lines.append(f"{indent}{assertion}")
            else:
                _, code = draw(gen_draw(env))
                lines.append(f"{indent}{code}")

    env.restore(save)
    return lines


@st.composite
def gen_if_block(
    draw: st.DrawFn, env: Env, indent: str, block_depth: int = 0
) -> list[str]:
    """Generate an if block (possibly with else)."""
    cond = draw(gen_if_condition(env))
    then_lines = draw(gen_block_body(env, indent + "    ", block_depth))

    lines = [f"{indent}if {cond}:"]
    lines.extend(then_lines)

    if draw(st.booleans()):
        else_lines = draw(gen_block_body(env, indent + "    ", block_depth))
        lines.append(f"{indent}else:")
        lines.extend(else_lines)

    return lines


# ---------------------------------------------------------------------------
# Misc statements (draw, assume, or extra draw)
# ---------------------------------------------------------------------------


@st.composite
def gen_misc_statement(draw: st.DrawFn, env: Env) -> str:
    """Generate a misc statement: draw, dependent draw, or assume."""
    if not env.has_vars():
        _, code = draw(gen_draw(env))
        return code

    choice = draw(st.integers(0, 4))
    if choice <= 2:
        return draw(gen_draw_or_dependent(env))
    elif choice == 3:
        return draw(gen_assume(env))
    else:
        return draw(gen_draw_or_dependent(env))


# ---------------------------------------------------------------------------
# Composite generator generation
# ---------------------------------------------------------------------------


_composite_counter = 0


@st.composite
def gen_composite_function(draw: st.DrawFn) -> tuple[str, str, str]:
    """Generate a @composite function definition.

    Returns (func_name, func_code, output_type) where output_type is
    the type tag of the value the composite produces ("tuple")."""
    global _composite_counter
    _composite_counter += 1
    name = f"_gen{_composite_counter}"

    n_draws = draw(st.integers(2, 4))
    body_lines = []
    return_names = []
    for i in range(n_draws):
        arg = f"_{name}_v{i}"
        _, expr, _, _ = draw(gen_base_generator_expr())
        body_lines.append(f"    {arg} = tc.draw({expr})")
        return_names.append(arg)

    body = "\n".join(body_lines)
    ret = ", ".join(return_names)
    func_code = f"@composite\ndef {name}(tc):\n{body}\n    return ({ret},)\n"
    return (name, func_code, "tuple")


@st.composite
def gen_nested_composite(
    draw: st.DrawFn, tree_entry_names: list[str]
) -> tuple[str, str, str]:
    """Generate a @composite that draws from trees and base generators.

    Returns (func_name, func_code, output_type)."""
    global _composite_counter
    _composite_counter += 1
    name = f"_gen{_composite_counter}"

    n_draws = draw(st.integers(2, 4))
    body_lines = []
    return_names = []
    for i in range(n_draws):
        arg = f"_{name}_v{i}"
        if tree_entry_names and draw(st.integers(0, 2)) == 0:
            tname = draw(st.sampled_from(tree_entry_names))
            body_lines.append(f"    {arg} = tc.draw({tname}())")
        else:
            _, expr, _, _ = draw(gen_base_generator_expr())
            body_lines.append(f"    {arg} = tc.draw({expr})")
        return_names.append(arg)

    body = "\n".join(body_lines)
    ret = ", ".join(return_names)
    func_code = f"@composite\ndef {name}(tc):\n{body}\n    return ({ret},)\n"
    return (name, func_code, "tuple")


# ---------------------------------------------------------------------------
# Recursive tree composite generation
# ---------------------------------------------------------------------------

# Unary operators take one child, binary take two.
_UNARY_OPS = ["neg", "not", "abs", "double"]
_BINARY_OPS = ["add", "sub", "mul", "eq", "lt", "and", "or", "max", "min"]


@st.composite
def gen_recursive_tree_function(draw: st.DrawFn) -> tuple[str, str]:
    """Generate a recursive tree composite function.

    Returns (entry_point_name, full_source_code) where entry_point_name
    is a callable that returns a Generator producing tree values.

    The generated code defines:
    - _tree_leaf(tc): produces a leaf value (integer or boolean)
    - _tree_node(tc, depth): produces a node at the given depth
    - _tree(tc): entry point, draws a depth then calls _tree_node

    Trees are nested tuples: leaves are plain values (int/bool),
    branches are ("op_name", child, ...) tuples.
    """
    global _composite_counter
    _composite_counter += 1
    tag = _composite_counter

    # Pick which operators this tree type uses
    n_unary = draw(st.integers(1, len(_UNARY_OPS)))
    n_binary = draw(st.integers(1, len(_BINARY_OPS)))
    unary_ops = draw(
        st.lists(
            st.sampled_from(_UNARY_OPS), min_size=n_unary, max_size=n_unary, unique=True
        )
    )
    binary_ops = draw(
        st.lists(
            st.sampled_from(_BINARY_OPS),
            min_size=n_binary,
            max_size=n_binary,
            unique=True,
        )
    )
    all_ops = unary_ops + binary_ops
    max_depth = draw(st.integers(2, 5))

    # Build the op dispatch: for each op, generate the branch code
    op_branches = []
    for op in unary_ops:
        op_branches.append(
            f"    if op == {len(op_branches)}:\n"
            f"        child = tc.draw(_tree_node_{tag}(depth - 1))\n"
            f'        return ("{op}", child)'
        )
    for op in binary_ops:
        op_branches.append(
            f"    if op == {len(op_branches)}:\n"
            f"        left = tc.draw(_tree_node_{tag}(depth - 1))\n"
            f"        right = tc.draw(_tree_node_{tag}(depth - 1))\n"
            f'        return ("{op}", left, right)'
        )

    op_dispatch = "\n".join(op_branches)

    leaf_choice = draw(st.sampled_from(["int", "bool", "mixed"]))
    if leaf_choice == "int":
        leaf_expr = "tc.draw(integers(0, 10))"
    elif leaf_choice == "bool":
        leaf_expr = "tc.draw(booleans())"
    else:
        leaf_expr = (
            "tc.draw(integers(0, 10)) if tc.draw(booleans()) else tc.draw(booleans())"
        )

    entry = f"_tree_{tag}"

    code = f"""\
@composite
def _tree_node_{tag}(tc, depth):
    if depth <= 0 or not tc.weighted(0.9):
        return {leaf_expr}
    op = tc.choice({len(all_ops) - 1})
{op_dispatch}

def {entry}():
    return integers(0, {max_depth}).flat_map(lambda d: _tree_node_{tag}(d))
"""
    return (entry, code)


@st.composite
def gen_tree_predicate(draw: st.DrawFn, name: str) -> str:
    """Falsifiable predicates for tree variables."""
    choice = draw(st.integers(0, 8))
    if choice == 0:
        d = draw(st.integers(1, 4))
        return f"tree_depth({name}) < {d}"
    elif choice == 1:
        d = draw(st.integers(2, 6))
        return f"tree_depth({name}) > {d}"
    elif choice == 2:
        n = draw(st.integers(2, 10))
        return f"tree_size({name}) < {n}"
    elif choice == 3:
        n = draw(st.integers(3, 15))
        return f"tree_size({name}) > {n}"
    elif choice == 4:
        n = draw(st.integers(1, 5))
        return f"tree_leaves({name}) < {n}"
    elif choice == 5:
        return f"tree_depth({name}) == 0"
    elif choice == 6:
        return f"isinstance({name}, tuple)"
    elif choice == 7:
        return f"not isinstance({name}, tuple)"
    else:
        n = draw(st.integers(1, 4))
        return f"len(tree_labels({name})) < {n}"


@st.composite
def gen_tree_nodes_predicate(draw: st.DrawFn, name: str) -> str:
    """Falsifiable predicates over the flattened node list of a tree."""
    choice = draw(st.integers(0, 7))
    if choice == 0:
        # All nodes are leaves
        return f"all(not isinstance(n, tuple) for n in {name})"
    elif choice == 1:
        # All nodes are branches
        return f"all(isinstance(n, tuple) for n in {name})"
    elif choice == 2:
        # No node is deeply nested
        d = draw(st.integers(1, 3))
        return f"all(not isinstance(n, tuple) or tree_depth(n) < {d} for n in {name})"
    elif choice == 3:
        # All branch nodes have the same label
        return f"len(set(n[0] for n in {name} if isinstance(n, tuple))) <= 1"
    elif choice == 4:
        # Node count bound
        n = draw(st.integers(2, 8))
        return f"len({name}) < {n}"
    elif choice == 5:
        # Every leaf is zero
        return f"all(n == 0 for n in {name} if not isinstance(n, tuple))"
    elif choice == 6:
        # No two adjacent nodes in the list are both leaves
        return (
            f"all(isinstance({name}[i], tuple) or isinstance({name}[i+1], tuple)"
            f" for i in range(len({name}) - 1))"
        )
    else:
        # At least one branch node exists
        return f"any(isinstance(n, tuple) for n in {name})"


# ---------------------------------------------------------------------------
# Full program generation
# ---------------------------------------------------------------------------


@st.composite
def program(draw: st.DrawFn) -> str:
    """Generate a complete pbtsmith program as Python source code.

    Structure mirrors hegelsmith:
    1. Phase 0: optional @composite helper functions
    2. Phase 1: 1-5 initial draws (some dependent)
    3. Phase 2: 1-3 assertion blocks with optional interleaved draws
    4. Phase 3: optional if block
    5. Phase 4: optional trailing statements (draws/assumes)
    """
    env = Env()
    lines: list[str] = []
    preamble_lines: list[str] = []
    indent = "        "

    # Phase 0: optional composite and recursive tree functions
    composite_names: list[str] = []
    tree_entry_names: list[str] = []
    if draw(st.booleans()):
        n_composites = draw(st.integers(1, 2))
        for _ in range(n_composites):
            name, func_code, out_typ = draw(gen_composite_function())
            preamble_lines.append(func_code)
            composite_names.append(name)
    if draw(st.booleans()):
        n_trees = draw(st.integers(1, 2))
        for _ in range(n_trees):
            entry_name, tree_code = draw(gen_recursive_tree_function())
            preamble_lines.append(tree_code)
            tree_entry_names.append(entry_name)
    # Nested composites that draw from trees (if trees are defined)
    if tree_entry_names and draw(st.booleans()):
        name, func_code, out_typ = draw(gen_nested_composite(tree_entry_names))
        preamble_lines.append(func_code)
        composite_names.append(name)

    # Phase 1: initial draws (sometimes from composites or trees)
    num_draws = draw(st.integers(1, 5))
    for _ in range(num_draws):
        r = draw(st.integers(0, 9))
        if composite_names and r == 0:
            cname = draw(st.sampled_from(composite_names))
            var = env.fresh_var()
            lines.append(f"{indent}{var} = tc.draw({cname}())")
            env.add(var, "tuple")
        elif tree_entry_names and r <= 2:
            tname = draw(st.sampled_from(tree_entry_names))
            var = env.fresh_var()
            tree_draw_kind = draw(st.integers(0, 2))
            if tree_draw_kind == 0:
                # Single tree
                lines.append(f"{indent}{var} = tc.draw({tname}())")
                env.add(var, "tree")
            elif tree_draw_kind == 1:
                # List of trees
                max_sz = draw(st.integers(2, 5))
                lines.append(
                    f"{indent}{var} = tc.draw(lists({tname}(), max_size={max_sz}))"
                )
                env.add(var, "list_tree")
            else:
                # Filtered tree (non-leaf)
                lines.append(
                    f"{indent}{var} = tc.draw("
                    f"{tname}().filter(lambda t: isinstance(t, tuple)))"
                )
                env.add(var, "tree")
            # Expose flattened node list for trees (not lists of trees).
            if env.vars[-1].typ == "tree":
                nodes_var = f"_nodes_{var}"
                lines.append(f"{indent}{nodes_var} = tree_nodes({var})")
                env.add(nodes_var, "list_tree_nodes")
        else:
            code = draw(gen_draw_or_dependent(env))
            lines.append(f"{indent}{code}")

    # Phase 2: assertion blocks
    num_blocks = draw(st.integers(1, 3))
    for i in range(num_blocks):
        # Sometimes interleave a misc statement
        if i > 0 and draw(st.booleans()):
            code = draw(gen_misc_statement(env))
            lines.append(f"{indent}{code}")

        # 1-2 assertions per block
        num_asserts = draw(st.integers(1, 2))
        for _ in range(num_asserts):
            assertion = draw(gen_rich_assertion(env))
            lines.append(f"{indent}{assertion}")

    # Phase 2.5: optional sampling from node lists to make subtree assertions
    node_list_vars = env.of_type("list_tree_nodes")
    if node_list_vars and draw(st.booleans()):
        nlv = draw(st.sampled_from(node_list_vars))
        sampled_var = env.fresh_var()
        lines.append(
            f"{indent}{sampled_var} = tc.draw("
            f"sampled_from({nlv.name}) if {nlv.name} else just(0))"
        )
        env.add(sampled_var, "tree")
        # Assert something about the sampled node
        pred = draw(gen_tree_predicate(sampled_var))
        lines.append(f"{indent}if not ({pred}): raise Failure({pred!r})")

    # Phase 3: optional if block
    if draw(st.booleans()) and env.has_vars():
        if_lines = draw(gen_if_block(env, indent))
        lines.extend(if_lines)

    # Phase 4: optional trailing statements
    num_trailing = draw(st.integers(0, 2))
    for _ in range(num_trailing):
        code = draw(gen_misc_statement(env))
        lines.append(f"{indent}{code}")

    max_examples = draw(st.integers(1, 1000))
    seed = draw(st.integers(0, 2**32 - 1))

    preamble = "\n".join(preamble_lines)
    body = "\n".join(lines)
    result = f"""\
{preamble}
try:
    @run_test(max_examples={max_examples}, database={{}}, quiet=True, random=Random({seed}))
    def _(tc):
{body}
except (Unsatisfiable, Failure):
    pass
    """
    return result


# ---------------------------------------------------------------------------
# Program rendering and execution
# ---------------------------------------------------------------------------

PROGRAM_PREAMBLE = """\
"""

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


pytestmark = [
    pytest.mark.requires("floats"),
    pytest.mark.requires("text"),
    pytest.mark.requires("bytes"),
    pytest.mark.requires("collections"),
    pytest.mark.hypothesis,
]


@given(program())
@settings(
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_pbtsmith_no_internal_errors(pbtkit_program: str) -> None:
    """Generated programs either succeed or fail with Failure,
    never with internal crashes."""
    note(pbtkit_program)
    exec(pbtkit_program, globals())


def test_regression_1():
    try:

        @run_test(max_examples=1, database={}, quiet=True, random=Random(0))
        def _(tc):
            v0 = tc.draw(integers(-1, -1))
            if not (v0 > 0):
                raise Failure("v0 > 0")
    except Failure:
        pass


def test_regression_2():
    try:

        @run_test(max_examples=1, database={}, random=Random(0), quiet=True)
        def _(tc):
            v0 = tc.draw(lists(booleans(), max_size=10))
            v1 = tc.draw(lists(integers(0, 0), max_size=10))
            if not (len(v0) == len(v1)):
                raise Failure("len(v0) == len(v1)")
    except (Unsatisfiable, Failure):
        pass


def test_regression_3():
    """Mutation pass stale index after result shortens during shrinking.

    The mutation pass computed j = i + j_offset based on the result length,
    then called state.test_function() which could shorten state.result.
    The subsequent access to state.result[j] would then crash."""
    try:

        @run_test(max_examples=100, database={}, quiet=True, random=Random(0))
        def _(tc):
            tc.draw(
                one_of(
                    lists(integers(0, 20), max_size=10, unique=True),
                    just(0),
                    just(0),
                    booleans(),
                )
            )
            v1 = tc.draw(
                one_of(
                    lists(integers(0, 20), max_size=10, unique=True),
                    just(1),
                    booleans(),
                    booleans(),
                )
            )
            if not (not v1):
                raise Failure("not v1")
    except (Unsatisfiable, Failure):
        pass


def test_regression_4():
    """Shrink stability: flat_map branch choice + list length failure mode switching.

    Two failure paths exist:
    - Short: v0 non-empty via flat_map True branch (5 choices)
    - Long: v4 has 3+ tuples (13 choices)
    The shrinker must be able to find the short path from either starting point."""
    try:

        @run_test(max_examples=100, database={}, quiet=True, random=Random(0))
        def _(tc):
            v0 = tc.draw(
                booleans().flat_map(
                    lambda x: (
                        text(min_codepoint=32, max_codepoint=126, max_size=5)
                        if x
                        else just("")
                    )
                )
            )
            tc.draw(just(False))
            tc.draw(booleans())
            v3 = tc.draw(floats(allow_nan=False, allow_infinity=False))
            v4 = tc.draw(lists(tuples(integers(0, 0), integers(0, 0)), max_size=10))
            if not (((len(v0) == 0) and (len(v4) < 3)) or (v3 > 0.0)):
                raise Failure("compound condition")
    except (Unsatisfiable, Failure):
        pass
