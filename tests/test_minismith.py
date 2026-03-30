"""Minismith: a random minithesis program generator.

Uses Hypothesis to generate random but valid minithesis test programs
as Python source code, then executes them to verify they either succeed
or fail with TestFailed (never crash internally).

Ported from hegelsmith (which does the same for hegel-rust).

The generated programs exercise minithesis's full API:
  - integers, booleans, floats, text, binary, lists
  - dependent draws (bounds derived from earlier variables)
  - if blocks with nested draws and assertions
  - tc.assume for filtering
  - genuinely falsifiable predicates (not tautologies)
  - multi-variable and compound boolean predicates
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Python type tags and environment tracking
# ---------------------------------------------------------------------------

TYPE_NAMES = ["bool", "int", "float", "str", "bytes", "list_int", "list_bool"]


class VarInfo:
    """A variable in scope."""

    __slots__ = ("name", "typ")

    def __init__(self, name: str, typ: str) -> None:
        self.name = name
        self.typ = typ


class Env:
    """Track variables in scope during program generation."""

    def __init__(self) -> None:
        self.vars: List[VarInfo] = []
        self.next_id = 0

    def fresh_var(self) -> str:
        name = f"v{self.next_id}"
        self.next_id += 1
        return name

    def add(self, name: str, typ: str) -> VarInfo:
        v = VarInfo(name, typ)
        self.vars.append(v)
        return v

    def save(self) -> int:
        return len(self.vars)

    def restore(self, point: int) -> None:
        del self.vars[point:]

    def of_type(self, *typs: str) -> List[VarInfo]:
        return [v for v in self.vars if v.typ in typs]

    def int_vars(self) -> List[VarInfo]:
        return self.of_type("int")

    def bool_vars(self) -> List[VarInfo]:
        return self.of_type("bool")

    def float_vars(self) -> List[VarInfo]:
        return self.of_type("float")

    def str_vars(self) -> List[VarInfo]:
        return self.of_type("str")

    def bytes_vars(self) -> List[VarInfo]:
        return self.of_type("bytes")

    def collection_vars(self) -> List[VarInfo]:
        return self.of_type("list_int", "list_bool", "bytes")

    def has_vars(self) -> bool:
        return len(self.vars) > 0


# ---------------------------------------------------------------------------
# Generator expression rendering
# ---------------------------------------------------------------------------


def gen_expr_code(typ: str, int_lo: int, int_hi: int) -> str:
    """Return source code for a minithesis generator expression."""
    if typ == "bool":
        return "booleans()"
    elif typ == "int":
        return f"integers({int_lo}, {int_hi})"
    elif typ == "float":
        return "floats(allow_nan=False, allow_infinity=False)"
    elif typ == "str":
        return "text(min_codepoint=32, max_codepoint=126, max_size=20)"
    elif typ == "bytes":
        return "binary(max_size=20)"
    elif typ == "list_int":
        return f"lists(integers({int_lo}, {int_hi}), max_size=10)"
    elif typ == "list_bool":
        return "lists(booleans(), max_size=10)"
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
    choice = draw(st.integers(0, 3))
    if choice == 0:
        return f"{name} > 0.0"
    elif choice == 1:
        t = draw(st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False))
        return f"{name} < {t}"
    elif choice == 2:
        return f"abs({name}) < 1.0"
    else:
        return f"{name} >= 0.0"


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
    choice = draw(st.integers(0, 5))
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
def gen_collection_predicate(draw: st.DrawFn, name: str) -> str:
    """Falsifiable predicates for list/bytes variables."""
    choice = draw(st.integers(0, 4))
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
    else:
        t = draw(st.integers(0, 5))
        return f"len({name}) == {t}"


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
    elif var.typ in ("list_int", "list_bool"):
        return draw(gen_collection_predicate(var.name))
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
        choice = draw(st.integers(0, 5))
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
        else:
            return f"{a.name} - {b.name} > 0"
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

    # Fallback: cross-type (int and collection)
    if int_vars and col_vars:
        iv = draw(st.sampled_from(int_vars))
        cv = draw(st.sampled_from(col_vars))
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
    return f"if not ({cond}): raise TestFailed({cond!r})"


# ---------------------------------------------------------------------------
# Draw statements
# ---------------------------------------------------------------------------


@st.composite
def gen_draw(draw: st.DrawFn, env: Env) -> Tuple[str, str]:
    """Generate a simple draw statement. Returns (type, code_line)."""
    typ = draw(st.sampled_from(TYPE_NAMES))
    int_lo = draw(st.integers(-100, 100))
    int_hi = draw(st.integers(int_lo, int_lo + 200))
    var = env.fresh_var()
    expr = gen_expr_code(typ, int_lo, int_hi)
    code = f"{var} = tc.any({expr})"
    env.add(var, typ)
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
        code = f"{var} = tc.any(integers({src}, {src} + {delta}))"
        env.add(var, "int")
    elif choice == 1:
        # Integer with dependent bounds: integers(-abs(src), abs(src))
        code = f"{var} = tc.any(integers(-abs({src}) - 1, abs({src}) + 1))"
        env.add(var, "int")
    elif choice == 2:
        # List with dependent max_size
        code = f"{var} = tc.any(lists(booleans(), max_size=abs({src}) % 10 + 1))"
        env.add(var, "list_bool")
    else:
        # Text with dependent max_size
        code = (
            f"{var} = tc.any(text(min_codepoint=32, max_codepoint=126, "
            f"max_size=abs({src}) % 20 + 1))"
        )
        env.add(var, "str")
    return code


@st.composite
def gen_draw_or_dependent(draw: st.DrawFn, env: Env) -> str:
    """Generate a draw, possibly dependent on an existing variable."""
    int_vars = env.int_vars()
    if int_vars and draw(st.integers(0, 9)) >= 6:
        return draw(gen_dependent_draw(env))
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

    options: List[str] = []
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
) -> List[str]:
    """Generate the body of an if/else block."""
    save = env.save()
    lines: List[str] = []

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
) -> List[str]:
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
# Full program generation
# ---------------------------------------------------------------------------


@st.composite
def program(draw: st.DrawFn) -> str:
    """Generate a complete minismith program as Python source code.

    Structure mirrors hegelsmith:
    1. Phase 1: 1-5 initial draws (some dependent)
    2. Phase 2: 1-3 assertion blocks with optional interleaved draws
    3. Phase 3: optional if block
    4. Phase 4: optional trailing statements (draws/assumes)
    """
    env = Env()
    lines: List[str] = []
    indent = "    "

    # Phase 1: initial draws
    num_draws = draw(st.integers(1, 5))
    for _ in range(num_draws):
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

    # Phase 3: optional if block
    if draw(st.booleans()) and env.has_vars():
        if_lines = draw(gen_if_block(env, indent))
        lines.extend(if_lines)

    # Phase 4: optional trailing statements
    num_trailing = draw(st.integers(0, 2))
    for _ in range(num_trailing):
        code = draw(gen_misc_statement(env))
        lines.append(f"{indent}{code}")

    body = "\n".join(lines)
    return f"def _test_body(tc):\n{body}\n"


# ---------------------------------------------------------------------------
# Program rendering and execution
# ---------------------------------------------------------------------------

PROGRAM_PREAMBLE = """\
from minithesis.core import Unsatisfiable, run_test
from minithesis.generators import (
    binary, booleans, floats, integers, lists, text,
)

class TestFailed(Exception):
    pass

"""


def render_full_test(program_code: str, max_examples: int) -> str:
    """Render a complete executable test script."""
    return f"""\
{PROGRAM_PREAMBLE}
{program_code}

try:
    @run_test(max_examples={max_examples}, database={{}})
    def _(tc):
        try:
            _test_body(tc)
        except TestFailed:
            pass
except Unsatisfiable:
    pass
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


@given(code=program(), max_examples=st.integers(1, 1000))
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_minismith_no_internal_errors(code: str, max_examples: int) -> None:
    """Generated programs either succeed or fail with TestFailed,
    never with internal crashes."""
    full = render_full_test(code, max_examples)
    exec(compile(full, "<minismith>", "exec"), {})
