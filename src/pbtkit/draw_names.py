"""Named draws for pbtkit.

When imported, this module automatically rewrites every test function run via
``run_test()``: each ``x = tc.draw(gen)`` assignment is transformed to
``x = tc.draw_named(gen, "x", repeatable)`` using a libcst-based CST rewrite,
so failing test output shows ``x = 42`` instead of the generic ``draw_1 = 42``.

``TestCase.draw_named(generator, name, repeatable)`` is also available directly:
prints ``name = value`` (non-repeatable) or ``name_1 = value``, ``name_2 = value``
(repeatable) instead of the generic ``draw_N = value`` label.

Usage::

    import pbtkit.draw_names  # enables auto-rewriting for all run_test() tests
    from pbtkit import run_test
    from pbtkit.generators import integers

    @run_test()
    def test_something(tc):
        x = tc.draw(integers(0, 100))
        assert x == 0

When the test fails, the output will be ``x = 42`` rather than ``draw_1 = 42``.
"""

from __future__ import annotations

import inspect
import textwrap
import types
from collections.abc import Callable
from typing import TypeVar, cast

import libcst as cst

from pbtkit.core import Generator, PbtkitState, TestCase, setup_hook

T = TypeVar("T")

# ---------------------------------------------------------------------------
# draw_named implementation
# ---------------------------------------------------------------------------


def _allocate_name(tc: TestCase, name: str, repeatable: bool) -> str:
    """Validate flag consistency, then return the display name for this draw.

    Panics (AssertionError) if:
    - ``name`` was previously used with the opposite ``repeatable`` flag.
    - ``repeatable=False`` and ``name`` was already used.

    For repeatable draws, returns ``name_N`` where N is the smallest positive
    integer whose formatted name is not already taken.
    """
    prev = tc._named_draw_flags.get(name)
    assert prev is None or prev == repeatable, (
        f"Name {name!r} used with inconsistent repeatable flags "
        f"(was {prev}, now {repeatable})"
    )
    tc._named_draw_flags[name] = repeatable

    if not repeatable:
        assert name not in tc._named_draw_used, (
            f"Non-repeatable name {name!r} used more than once"
        )
        tc._named_draw_used.add(name)
        return name

    # Repeatable: find the smallest N >= 1 not already taken.
    n = 1
    while f"{name}_{n}" in tc._named_draw_used:
        n += 1
    display = f"{name}_{n}"
    tc._named_draw_used.add(display)
    return display


def _draw_named(
    self: TestCase, generator: Generator[T], name: str, repeatable: bool
) -> T:
    """Draw from *generator*, printing the result as ``name = value`` on failure."""
    self.depth += 1
    try:
        result = generator.produce(self)
    finally:
        self.depth -= 1

    # Validate and (conditionally) print at top level only.
    if self.depth == 0:
        display = _allocate_name(self, name, repeatable)
        if self.print_results:
            print(f"{display} = {result!r}")

    return result  # type: ignore[return-value]


TestCase.draw_named = _draw_named  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# libcst rewriter — pass 1 (collect) and pass 2 (transform)
# ---------------------------------------------------------------------------


class _DrawNameCollector(cst.CSTTransformer):
    """Pass 1 (collecting, non-transforming): find all ``x = tc.draw(gen)``
    assignments in the function body and determine whether each name is
    repeatable.

    A name is repeatable if:
    - It appears inside a nested scope (for/while/if/with/try/nested fn/class), OR
    - It appears more than once (in any location).

    All ``leave_*`` methods return the unchanged node — this pass does not
    transform the tree.
    """

    def __init__(self, tc_param: str) -> None:
        super().__init__()
        self.tc_param = tc_param
        self._depth = 0  # 0 = function body level; 1+ = inside control flow
        # True until the outermost FunctionDef has been entered; we do not
        # count the test function itself as a nesting scope.
        self._at_top_level_func = True
        # name → repeatable flag (True if nested or seen more than once)
        self.names: dict[str, bool] = {}

    def _is_draw_call(self, node: cst.BaseExpression) -> bool:
        """Return True if *node* is ``{tc_param}.draw(...)``."""
        return (
            isinstance(node, cst.Call)
            and isinstance(node.func, cst.Attribute)
            and isinstance(node.func.value, cst.Name)
            and node.func.value.value == self.tc_param
            and node.func.attr.value == "draw"
        )

    def _record(self, name: str) -> None:
        """Record an occurrence of *name* at the current nesting depth."""
        nested = self._depth > 0
        if name in self.names:
            self.names[name] = self.names[name] or nested
        else:
            self.names[name] = nested

    def visit_Assign(self, node: cst.Assign) -> bool:
        """Detect the ``x = tc.draw(gen)`` pattern."""
        if len(node.targets) == 1:
            target = node.targets[0].target
            if isinstance(target, cst.Name) and self._is_draw_call(node.value):
                self._record(target.value)
        return True  # continue visiting children

    # --- nesting depth tracking (visit_* increments, leave_* decrements) ---

    def visit_For(self, node: cst.For) -> bool:
        self._depth += 1
        return True

    def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
        self._depth -= 1
        return original_node

    def visit_While(self, node: cst.While) -> bool:
        self._depth += 1
        return True

    def leave_While(
        self, original_node: cst.While, updated_node: cst.While
    ) -> cst.While:
        self._depth -= 1
        return original_node

    def visit_If(self, node: cst.If) -> bool:
        self._depth += 1
        return True

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        self._depth -= 1
        return original_node

    def visit_With(self, node: cst.With) -> bool:
        self._depth += 1
        return True

    def leave_With(self, original_node: cst.With, updated_node: cst.With) -> cst.With:
        self._depth -= 1
        return original_node

    def visit_Try(self, node: cst.Try) -> bool:
        self._depth += 1
        return True

    def leave_Try(self, original_node: cst.Try, updated_node: cst.Try) -> cst.Try:
        self._depth -= 1
        return original_node

    def visit_TryStar(self, node: cst.TryStar) -> bool:
        self._depth += 1
        return True

    def leave_TryStar(
        self, original_node: cst.TryStar, updated_node: cst.TryStar
    ) -> cst.TryStar:
        self._depth -= 1
        return original_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self._at_top_level_func:
            self._at_top_level_func = False
            return True  # don't count the test function itself as a nesting scope
        self._depth += 1
        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if self._depth > 0:
            self._depth -= 1
        return original_node

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._depth += 1
        return True

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self._depth -= 1
        return original_node


class _DrawRewriter(cst.CSTTransformer):
    """Pass 2: rewrite ``x = tc.draw(gen)`` → ``x = tc.draw_named(gen, "x", bool)``
    and strip decorators from the outermost function definition."""

    def __init__(self, tc_param: str, names: dict[str, bool]) -> None:
        super().__init__()
        self.tc_param = tc_param
        self.names = names
        # Tracks nesting depth so we strip decorators from the outermost
        # function only.  visit_FunctionDef increments before the body is
        # walked; leave_FunctionDef checks depth then decrements.
        self._funcdef_depth = 0

    def _is_draw_call(self, node: cst.BaseExpression) -> bool:
        return (
            isinstance(node, cst.Call)
            and isinstance(node.func, cst.Attribute)
            and isinstance(node.func.value, cst.Name)
            and node.func.value.value == self.tc_param
            and node.func.attr.value == "draw"
        )

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._funcdef_depth += 1
        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        """Strip decorators from the outermost function definition only."""
        depth = self._funcdef_depth
        self._funcdef_depth -= 1
        if depth == 1:
            return updated_node.with_changes(decorators=[])
        return updated_node

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.Assign:
        """Rewrite ``x = tc.draw(gen)`` to ``x = tc.draw_named(gen, "x", bool)``."""
        if len(updated_node.targets) != 1:
            return updated_node
        target = updated_node.targets[0].target
        if not isinstance(target, cst.Name):
            return updated_node
        if not self._is_draw_call(updated_node.value):
            return updated_node

        var_name = target.value
        if var_name not in self.names:
            assert False, f"BUG: {var_name!r} matched rewriter but not collector"

        repeatable = self.names[var_name]
        old_call = updated_node.value
        assert isinstance(old_call, cst.Call)

        # Build new func attribute: tc.draw_named
        new_func = old_call.func.with_changes(attr=cst.Name("draw_named"))

        # Existing argument (the generator): strip trailing comma.
        existing_args = list(old_call.args)
        if existing_args:
            existing_args[-1] = existing_args[-1].with_changes(
                comma=cst.MaybeSentinel.DEFAULT
            )

        name_arg = cst.Arg(
            value=cst.SimpleString(f'"{var_name}"'),
            comma=cst.MaybeSentinel.DEFAULT,
        )
        repeatable_arg = cst.Arg(
            value=cst.Name("True" if repeatable else "False"),
        )

        new_call = old_call.with_changes(
            func=new_func,
            args=[*existing_args, name_arg, repeatable_arg],
        )
        return updated_node.with_changes(value=new_call)


def rewrite_test_function(func: Callable) -> Callable:
    """Rewrite *func* so that ``x = tc.draw(gen)`` becomes
    ``x = tc.draw_named(gen, "x", repeatable)``.

    Returns the rewritten callable, or the original *func* if anything fails.
    """
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)

        tree = cst.parse_module(source)

        # Determine the TestCase parameter name from the code object.
        # co_varnames[0] is always the first positional parameter.
        tc_param = func.__code__.co_varnames[0]

        # Pass 1: collect draw-assignment names and their repeatability.
        collector = _DrawNameCollector(tc_param)
        tree.visit(collector)

        # Nothing to rewrite — return original unchanged.
        if not collector.names:
            return func

        # Pass 2: rewrite the CST.
        rewriter = _DrawRewriter(tc_param, collector.names)
        new_tree = tree.visit(rewriter)
        new_source = new_tree.code

        if func.__closure__:
            # When exec'd at module level, free variables would be compiled as
            # globals, not free variables.  Wrap the source in an outer function
            # that takes those names as parameters so Python compiles the inner
            # function with the correct co_freevars in the code object.
            freevars = func.__code__.co_freevars
            params = ", ".join(freevars)
            outer_source = (
                f"def _make_closure_({params}):\n"
                + "\n".join(f"    {line}" for line in new_source.splitlines())
                + f"\n    return {func.__name__}\n"
            )
            outer_ns: dict[str, object] = {}
            exec(outer_source, func.__globals__, outer_ns)  # noqa: S102
            cell_values = [c.cell_contents for c in func.__closure__]
            make = cast(Callable[..., Callable], outer_ns["_make_closure_"])
            intermediate = make(*cell_values)
            # Rebind the rewritten code object to the *original* closure cells so
            # that mutations via `nonlocal` are reflected in the enclosing scope.
            new_func: Callable = types.FunctionType(
                intermediate.__code__,
                func.__globals__,
                func.__name__,
                func.__defaults__,
                func.__closure__,
            )
        else:
            # Exec the rewritten source in the function's global namespace.
            ns: dict[str, object] = {}
            exec(new_source, func.__globals__, ns)  # noqa: S102
            new_func = cast(Callable, ns[func.__name__])

        # Preserve keyword-only defaults if any.
        if func.__kwdefaults__:
            new_func.__kwdefaults__ = func.__kwdefaults__  # type: ignore[attr-defined]

        return new_func
    except Exception:
        return func


# ---------------------------------------------------------------------------
# Setup hook: automatically rewrite all test functions.
# ---------------------------------------------------------------------------


@setup_hook
def _draw_names_hook(state: PbtkitState) -> None:
    """Rewrite ``x = tc.draw(gen)`` → ``x = tc.draw_named(gen, "x", ...)`` for
    every test function, installing the rewritten version on the state for both
    execution and the final failing replay.  Active for all tests once this
    module is imported."""
    orig = state._original_test
    if orig is None:
        return
    rewritten = rewrite_test_function(orig)
    if rewritten is orig:
        # Rewriting was a no-op or failed — leave the state untouched.
        return
    state.replace_test_function(rewritten)
    state._print_function = rewritten
