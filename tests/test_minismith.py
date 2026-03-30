"""Minismith: a random minithesis program generator.

Uses Hypothesis to generate random but valid minithesis test programs,
then verifies they either pass or fail with TestFailed (never crash
internally).

Inspired by hegelsmith (../hegelsmith), which does the same for
hegel-rust.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import pytest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from minithesis.core import (
    Generator,
    StopTest,
    TestCase,
    Unsatisfiable,
    run_test,
)
from minithesis.generators import (
    binary,
    booleans,
    floats,
    integers,
    lists,
    text,
)

# ---------------------------------------------------------------------------
# TestFailed — the only acceptable failure mode
# ---------------------------------------------------------------------------


class TestFailed(Exception):
    """Raised when a generated test program's assertion fails."""


# ---------------------------------------------------------------------------
# Python type model
# ---------------------------------------------------------------------------


class PyType:
    """Represents a Python type for code generation."""

    def __init__(self, name: str):
        self.name = name


BOOL = PyType("bool")
INT = PyType("int")
FLOAT = PyType("float")
STRING = PyType("str")
BYTES = PyType("bytes")
LIST_BOOL = PyType("list_bool")
LIST_INT = PyType("list_int")

LEAF_TYPES = [BOOL, INT, FLOAT, STRING]
ALL_TYPES = [BOOL, INT, FLOAT, STRING, BYTES, LIST_BOOL, LIST_INT]


# ---------------------------------------------------------------------------
# Generator expressions — returns a minithesis Generator for a PyType
# ---------------------------------------------------------------------------


def gen_for_type(typ: PyType, int_lo: int, int_hi: int) -> Generator:
    """Return a minithesis Generator that produces values of the given type."""
    if typ.name == "bool":
        return booleans()
    elif typ.name == "int":
        return integers(int_lo, int_hi)
    elif typ.name == "float":
        return floats(allow_nan=False, allow_infinity=False)
    elif typ.name == "str":
        return text(min_codepoint=32, max_codepoint=126, max_size=20)
    elif typ.name == "bytes":
        return binary(max_size=20)
    elif typ.name == "list_bool":
        return lists(booleans(), max_size=10)
    elif typ.name == "list_int":
        return lists(integers(int_lo, int_hi), max_size=10)
    else:
        raise ValueError(f"Unknown type: {typ.name}")


# ---------------------------------------------------------------------------
# Statement model
# ---------------------------------------------------------------------------


class Statement:
    """A statement in the generated test body."""

    def execute(self, namespace: dict) -> None:
        raise NotImplementedError


class Draw(Statement):
    def __init__(self, var_name: str, gen: Generator):
        self.var_name = var_name
        self.gen = gen

    def execute(self, namespace: dict) -> None:
        tc = namespace["tc"]
        namespace[self.var_name] = tc.any(self.gen)


class Assert(Statement):
    def __init__(self, condition: Callable[[dict], bool], description: str):
        self.condition = condition
        self.description = description

    def execute(self, namespace: dict) -> None:
        if not self.condition(namespace):
            raise TestFailed(self.description)


class Assume(Statement):
    def __init__(self, condition: Callable[[dict], bool]):
        self.condition = condition

    def execute(self, namespace: dict) -> None:
        tc = namespace["tc"]
        if not self.condition(namespace):
            tc.reject()


# ---------------------------------------------------------------------------
# Variable environment
# ---------------------------------------------------------------------------


class VarInfo:
    def __init__(self, name: str, typ: PyType):
        self.name = name
        self.typ = typ


class Env:
    def __init__(self) -> None:
        self.vars: List[VarInfo] = []
        self._next_id = 0

    def fresh_var(self, typ: PyType) -> str:
        name = f"v{self._next_id}"
        self._next_id += 1
        self.vars.append(VarInfo(name, typ))
        return name

    def vars_of_type(self, name: str) -> List[VarInfo]:
        return [v for v in self.vars if v.typ.name == name]


# ---------------------------------------------------------------------------
# Assertion strategies
# ---------------------------------------------------------------------------


def int_assertion(v: VarInfo, kind: int, other: Optional[VarInfo]) -> Statement:
    """Generate an assertion about an integer variable."""
    if kind == 0:
        return Assert(lambda ns, n=v.name: ns[n] == ns[n], f"{v.name} == {v.name}")
    elif kind == 1 and other is not None:
        return Assert(
            lambda ns, a=v.name, b=other.name: ns[a] + ns[b] == ns[b] + ns[a],
            f"{v.name} + {other.name} == {other.name} + {v.name}",
        )
    elif kind == 2:
        return Assert(
            lambda ns, n=v.name: isinstance(ns[n], int),
            f"isinstance({v.name}, int)",
        )
    else:
        return Assert(
            lambda ns, n=v.name: ns[n] - ns[n] == 0, f"{v.name} - {v.name} == 0"
        )


def bool_assertion(v: VarInfo, kind: int) -> Statement:
    if kind == 0:
        return Assert(
            lambda ns, n=v.name: ns[n] or not ns[n], f"{v.name} or not {v.name}"
        )
    else:
        return Assert(
            lambda ns, n=v.name: ns[n] == (not not ns[n]),
            f"{v.name} == (not not {v.name})",
        )


def str_assertion(v: VarInfo, kind: int) -> Statement:
    if kind == 0:
        return Assert(lambda ns, n=v.name: len(ns[n]) >= 0, f"len({v.name}) >= 0")
    elif kind == 1:
        return Assert(lambda ns, n=v.name: ns[n] == ns[n], f"{v.name} == {v.name}")
    else:
        return Assert(
            lambda ns, n=v.name: isinstance(ns[n], str), f"isinstance({v.name}, str)"
        )


def list_assertion(v: VarInfo, kind: int) -> Statement:
    if kind == 0:
        return Assert(lambda ns, n=v.name: len(ns[n]) >= 0, f"len({v.name}) >= 0")
    else:
        return Assert(
            lambda ns, n=v.name: ns[n] == list(ns[n]),
            f"{v.name} == list({v.name})",
        )


def fallback_assertion(v: VarInfo) -> Statement:
    return Assert(
        lambda ns, n=v.name: ns[n] is not None or ns[n] is None,
        f"{v.name} is not None or {v.name} is None",
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies for program generation
# ---------------------------------------------------------------------------


@st.composite
def draw_statement(draw, env: Env) -> Statement:
    """Generate a Draw statement for a fresh variable."""
    typ = draw(st.sampled_from(ALL_TYPES))
    int_lo = draw(st.integers(-1000, 1000))
    int_hi = draw(st.integers(int_lo, int_lo + 2000))
    name = env.fresh_var(typ)
    return Draw(name, gen_for_type(typ, int_lo, int_hi))


@st.composite
def assertion_statement(draw, env: Env) -> Optional[Statement]:
    """Generate an assertion about an in-scope variable."""
    if not env.vars:
        return None

    v = draw(st.sampled_from(env.vars))

    int_vars = env.vars_of_type("int")
    bool_vars = env.vars_of_type("bool")
    str_vars = env.vars_of_type("str")
    list_vars = [vi for vi in env.vars if vi.typ.name.startswith("list_")]

    if v.typ.name == "int" and int_vars:
        other = draw(st.sampled_from(int_vars))
        kind = draw(st.integers(0, 3))
        return int_assertion(v, kind, other if other is not v else None)
    elif v.typ.name == "bool" and bool_vars:
        return bool_assertion(v, draw(st.integers(0, 1)))
    elif v.typ.name == "str" and str_vars:
        return str_assertion(v, draw(st.integers(0, 2)))
    elif v.typ.name.startswith("list_") and list_vars:
        return list_assertion(v, draw(st.integers(0, 1)))
    else:
        return fallback_assertion(v)


@st.composite
def program(draw) -> List[Statement]:
    """Generate a complete minismith test program."""
    env = Env()
    stmts: List[Statement] = []

    # Phase 1: draw some variables (1-4)
    n_draws = draw(st.integers(1, 4))
    for _ in range(n_draws):
        stmts.append(draw(draw_statement(env)))

    # Phase 2: assertions (1-3)
    n_asserts = draw(st.integers(1, 3))
    for _ in range(n_asserts):
        a = draw(assertion_statement(env))
        if a is not None:
            stmts.append(a)

    # Phase 3: optionally draw more and assert again
    if draw(st.booleans()):
        stmts.append(draw(draw_statement(env)))
        a = draw(assertion_statement(env))
        if a is not None:
            stmts.append(a)

    return stmts


def run_program(tc: TestCase, stmts: List[Statement]) -> None:
    """Execute a generated program. Raises TestFailed on assertion
    failure, or propagates other exceptions on internal errors."""
    namespace: dict = {"tc": tc}
    for stmt in stmts:
        stmt.execute(namespace)


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


@given(stmts=program())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_minismith_no_internal_errors(stmts):
    """Generated programs either pass or fail with TestFailed,
    never with internal crashes."""
    try:

        @run_test(max_examples=10, database={})
        def _(tc):
            try:
                run_program(tc, stmts)
            except TestFailed:
                pass

    except Unsatisfiable:
        # Some generated programs may be unsatisfiable — that's fine.
        pass
