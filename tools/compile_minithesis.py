#!/usr/bin/env python3
"""Compile the minithesis package into a single minithesis.py file.

Uses libcst for structural analysis (finding monkey-patches, imports, stubs)
and text-based manipulation for assembly. The result is formatted with ruff.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import libcst as cst

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "minithesis"
BUILD = ROOT / "build"

EXTENSIONS = ["floats", "bytes", "text", "collections", "targeting"]

HEADER = """\
# Compiled minithesis — generated from the modular source.
# Do not edit by hand.
#
# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
"""


# ---------------------------------------------------------------------------
# libcst analysis
# ---------------------------------------------------------------------------


def find_monkey_patches(tree: cst.Module) -> dict[str, str]:
    """Find TestCase.method = func assignments. Returns {method: func}."""
    patches: dict[str, str] = {}
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for s in stmt.body:
            if not isinstance(s, cst.Assign):
                continue
            for target in s.targets:
                t = target.target
                if (
                    isinstance(t, cst.Attribute)
                    and isinstance(t.value, cst.Name)
                    and t.value.value == "TestCase"
                    and isinstance(s.value, cst.Name)
                ):
                    patches[t.attr.value] = s.value.value
    return patches


def collect_imports(trees: dict[str, cst.Module]) -> str:
    """Merge all non-minithesis imports from all modules."""
    from_imports: dict[str, set[str]] = defaultdict(set)
    regular: set[str] = set()

    for tree in trees.values():
        for stmt in tree.body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                continue
            for s in stmt.body:
                if isinstance(s, cst.ImportFrom):
                    mod_name = _dotted(s.module)
                    if mod_name.startswith("minithesis") or mod_name == "__future__":
                        continue
                    if isinstance(s.names, cst.ImportStar):
                        continue
                    for alias in s.names:
                        from_imports[mod_name].add(alias.name.value)
                elif isinstance(s, cst.Import):
                    if isinstance(s.names, cst.ImportStar):
                        continue
                    for alias in s.names:
                        name = _dotted(alias.name)
                        if name.startswith("minithesis"):
                            continue
                        regular.add(name)

    lines = ["from __future__ import annotations", ""]
    for name in sorted(regular):
        lines.append(f"import {name}")
    if regular:
        lines.append("")
    for mod in sorted(from_imports):
        names = sorted(from_imports[mod])
        lines.append(f"from {mod} import {', '.join(names)}")
    return "\n".join(lines)


def _dotted(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_dotted(node.value)}.{node.attr.value}"
    return ""


# ---------------------------------------------------------------------------
# Text-based helpers
# ---------------------------------------------------------------------------


def _find_body_start(lines: list[str], def_line: int) -> int:
    """Find the first line of the function body (after the signature).
    Handles multi-line signatures by tracking parentheses."""
    depth = 0
    for i in range(def_line, len(lines)):
        for ch in lines[i]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        if depth == 0 and ":" in lines[i]:
            return i + 1
    return def_line + 1


def _block_end(lines: list[str], start: int) -> int:
    """Find end of a block (function/class) starting at lines[start].
    The start line should be the def/class line (with the colon).
    Returns exclusive end index."""
    # First, find the actual body start (skip multi-line signatures)
    body_start = _find_body_start(lines, start)
    if body_start >= len(lines):
        return body_start
    # The body is indented deeper than the def line
    indent = len(lines[start]) - len(lines[start].lstrip())
    last = body_start - 1
    for i in range(body_start, len(lines)):
        if not lines[i].strip():
            continue
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent <= indent:
            return last + 1
        last = i
    return last + 1


def extract_function(source: str, func_name: str) -> str:
    """Extract a module-level function's source text by name."""
    lines = source.splitlines(keepends=True)
    pattern = re.compile(rf"^def {re.escape(func_name)}\(")
    start = None
    for i, line in enumerate(lines):
        if pattern.match(line):
            start = i
            break
    assert start is not None, f"Function {func_name} not found"
    end = _block_end(lines, start)
    # Trim trailing blank lines
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "".join(lines[start:end])


def func_to_method(text: str, old_name: str, new_name: str) -> str:
    """Convert a standalone function to a class method."""
    # Rename
    text = text.replace(f"def {old_name}(", f"def {new_name}(", 1)
    # Strip self annotation: "self: TestCase" -> "self"
    text = re.sub(r"\bself\s*:\s*TestCase\b", "self", text)
    # Indent by 4 spaces (module level -> class method level)
    lines = text.splitlines(keepends=True)
    return "".join("    " + line if line.strip() else line for line in lines)


def _skip_docstring(lines: list[str], start: int) -> int:
    """Skip a triple-quoted docstring. Returns next index."""
    line = lines[start].strip()
    for q in ('"""', "'''"):
        if q in line:
            if line.count(q) >= 2:
                return start + 1
            for i in range(start + 1, len(lines)):
                if q in lines[i]:
                    return i + 1
    return start + 1


def _is_import(line: str) -> bool:
    return line.startswith("import ") or line.startswith("from ")


def _skip_import(lines: list[str], start: int) -> int:
    """Skip an import (possibly multi-line with parens)."""
    if "(" in lines[start] and ")" not in lines[start]:
        for i in range(start + 1, len(lines)):
            if ")" in lines[i]:
                return i + 1
    return start + 1


def _is_divider(line: str) -> bool:
    """Check for section divider like '# -----...'"""
    s = line.strip()
    return s.startswith("# ") and len(s) > 10 and s[2:].strip("-") == ""


def _skip_header(lines: list[str]) -> int:
    """Skip leading comment lines and blank lines. Returns first code line."""
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("#")):
        i += 1
    return i


# ---------------------------------------------------------------------------
# Module processing
# ---------------------------------------------------------------------------


def process_core(
    source: str, method_texts: dict[str, str], stubs_to_remove: set[str]
) -> str:
    """Process core.py: remove imports/header, replace or remove stubs."""
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    i = _skip_header(lines)

    while i < len(lines):
        line = lines[i]

        # Skip imports
        if _is_import(line):
            i = _skip_import(lines, i)
            continue

        # Skip the stub-related comment block
        if "draw_float, draw_bytes, draw_string are added" in line:
            while i < len(lines) and lines[i].strip().startswith("#"):
                i += 1
            continue

        # Check for stub method to replace or remove
        match = re.match(r"^    def (\w+)\(", line)
        if match and match.group(1) in stubs_to_remove:
            name = match.group(1)
            end = _block_end(lines, i)
            if name in method_texts:
                out.append(method_texts[name])
                if not method_texts[name].endswith("\n"):
                    out.append("\n")
            i = end
            continue

        out.append(line)
        i += 1

    return "".join(out)


def process_extension(source: str, moved_funcs: set[str]) -> str:
    """Process an extension module: remove imports, docstring, section dividers,
    monkey-patch assignments, and functions that were moved to TestCase."""
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    i = _skip_header(lines)

    # Skip docstring if present
    if i < len(lines) and ('"""' in lines[i] or "'''" in lines[i]):
        i = _skip_docstring(lines, i)

    while i < len(lines):
        line = lines[i]

        # Skip imports
        if _is_import(line):
            i = _skip_import(lines, i)
            continue

        # Skip section dividers (3-line blocks: divider, title, divider)
        if _is_divider(line):
            i += 1
            if i < len(lines) and lines[i].strip().startswith("#") and not _is_divider(lines[i]):
                i += 1
            if i < len(lines) and _is_divider(lines[i]):
                i += 1
            continue

        # Skip monkey-patch assignment: TestCase.X = Y
        if re.match(r"^TestCase\.\w+\s*=", line):
            i += 1
            continue

        # Skip single comment line immediately before a monkey-patch
        if (
            line.strip().startswith("#")
            and i + 1 < len(lines)
            and re.match(r"^TestCase\.\w+\s*=", lines[i + 1])
        ):
            i += 1
            continue

        # Skip moved functions (that became methods on TestCase)
        func_match = re.match(r"^def (\w+)\(", line)
        if func_match and func_match.group(1) in moved_funcs:
            i = _block_end(lines, i)
            continue

        out.append(line)
        i += 1

    return "".join(out)


def process_generators(source: str) -> str:
    """Process generators.py: remove imports and docstring."""
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    i = _skip_header(lines)

    # Skip docstring
    if i < len(lines) and ('"""' in lines[i] or "'''" in lines[i]):
        i = _skip_docstring(lines, i)

    while i < len(lines):
        line = lines[i]
        if _is_import(line):
            i = _skip_import(lines, i)
            continue
        out.append(line)
        i += 1

    return "".join(out)


# ---------------------------------------------------------------------------
# Main compilation
# ---------------------------------------------------------------------------


def extract_docstring(source: str) -> str:
    """Extract the module docstring from source text, or return empty string."""
    lines = source.splitlines(keepends=True)
    i = _skip_header(lines)
    if i >= len(lines):
        return ""
    line = lines[i].strip()
    for q in ('"""', "'''"):
        if line.startswith(q):
            if line.count(q) >= 2:
                return lines[i]
            end = _skip_docstring(lines, i)
            return "".join(lines[i:end])
    return ""


def compile_minithesis(disabled: frozenset[str] = frozenset()) -> str:
    """Compile the minithesis package into a single source string.

    Extensions listed in ``disabled`` are excluded from the compiled
    output — their TestCase stubs are removed entirely.
    """
    extensions = [e for e in EXTENSIONS if e not in disabled]

    # Read and parse all modules
    sources: dict[str, str] = {}
    trees: dict[str, cst.Module] = {}
    for name in ["core", "__init__"] + extensions + ["generators"]:
        src = (SRC / f"{name}.py").read_text()
        sources[name] = src
        trees[name] = cst.parse_module(src)

    # Also parse disabled extensions (just to find their monkey-patch names)
    all_stub_names: set[str] = set()
    for ext in EXTENSIONS:
        if ext in disabled:
            src = (SRC / f"{ext}.py").read_text()
            tree = cst.parse_module(src)
        else:
            tree = trees[ext]
        all_stub_names.update(find_monkey_patches(tree))

    # Extract the package docstring from __init__.py
    module_docstring = extract_docstring(sources["__init__"])

    # Find monkey-patches across enabled extensions
    all_patches: dict[str, tuple[str, str]] = {}  # method -> (ext, func)
    for ext in extensions:
        for method, func in find_monkey_patches(trees[ext]).items():
            all_patches[method] = (ext, func)

    # Convert implementation functions to method text
    method_texts: dict[str, str] = {}
    for method, (ext, func) in all_patches.items():
        func_text = extract_function(sources[ext], func)
        method_texts[method] = func_to_method(func_text, func, method)

    # Merge imports from all modules (exclude __init__ — it has no unique stdlib imports)
    import_trees = {k: v for k, v in trees.items() if k != "__init__"}
    import_text = collect_imports(import_trees)

    # Process core (replace enabled stubs with real methods, remove disabled stubs)
    core_body = process_core(sources["core"], method_texts, all_stub_names)

    # Process extensions (remove imports, monkey-patches, moved functions)
    ext_bodies: list[str] = []
    for ext in extensions:
        moved = {func for _, (en, func) in all_patches.items() if en == ext}
        body = process_extension(sources[ext], moved)
        if body.strip():
            ext_bodies.append(body)

    # Process generators (remove imports)
    gen_body = process_generators(sources["generators"])

    # Assemble everything
    parts = [HEADER, module_docstring, "\n", import_text, "\n\n", core_body]
    for body in ext_bodies:
        parts.append("\n")
        parts.append(body)
    parts.append("\n")
    parts.append(gen_body)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Test package generation
# ---------------------------------------------------------------------------


def _generate_init_py(disabled: frozenset[str]) -> str:
    """Generate the __init__.py for the compiled test package."""
    enabled = [
        e
        for e in ["floats", "bytes", "text", "collections", "targeting", "generators"]
        if e not in disabled
    ]
    disabled_list = sorted(disabled)

    lines = [
        "from __future__ import annotations",
        "",
        "import sys",
        "import types",
        "",
        "import minithesis.core as _core",
        "",
        "from minithesis.core import (  # noqa: PLC0415",
        "    Database,",
        "    DirectoryDB,",
        "    Generator,",
        "    TestCase,",
        "    Unsatisfiable,",
        "    run_test,",
        ")",
        "",
        "import minithesis as _pkg  # noqa: PLC0415",
        "",
        "# Alias enabled submodules to the compiled core.",
    ]
    for name in enabled:
        lines.append(f'sys.modules["minithesis.{name}"] = _core')
        lines.append(f'setattr(_pkg, "{name}", _core)')

    if disabled_list:
        lines.append("")
        lines.append("")
        lines.append("# Disabled modules get a dummy that returns placeholder symbols.")
        lines.append("class _DisabledSymbol:")
        lines.append("    def __init__(self, mod, name):")
        lines.append("        self._mod, self._name = mod, name")
        lines.append("    def __call__(self, *a, **kw):")
        lines.append(
            "        raise NotImplementedError("
            'f"{self._mod}.{self._name} is not available")'
        )
        lines.append("    def __repr__(self):")
        lines.append('        return f"<disabled: minithesis.{self._mod}.{self._name}>"')
        lines.append("")
        lines.append("class _DisabledModule(types.ModuleType):")
        lines.append("    def __init__(self, mod, full):")
        lines.append("        super().__init__(full)")
        lines.append("        self._module_name = mod")
        lines.append("        self.DISABLED = True")
        lines.append("    def __getattr__(self, name):")
        lines.append('        if name.startswith("__") and name.endswith("__"):')
        lines.append("            raise AttributeError(name)")
        lines.append("        return _DisabledSymbol(self._module_name, name)")
        lines.append("")
        for name in disabled_list:
            lines.append(
                f'sys.modules["minithesis.{name}"] = _DisabledModule("{name}", "minithesis.{name}")'
            )

    lines.extend([
        "",
        "__all__ = [",
        '    "Database",',
        '    "DirectoryDB",',
        '    "Generator",',
        '    "TestCase",',
        '    "Unsatisfiable",',
        '    "run_test",',
        "]",
        "",
    ])

    return "\n".join(lines)


def write_test_package(
    compiled_source: str, pkg_dir: Path, disabled: frozenset[str]
) -> None:
    """Write a test package that uses the compiled source as its core."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "core.py").write_text(compiled_source)
    (pkg_dir / "__init__.py").write_text(_generate_init_py(disabled))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Compile minithesis into a single file")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=BUILD / "minithesis.py",
        help="Output path for the compiled file",
    )
    p.add_argument(
        "--pkg",
        type=Path,
        default=BUILD / "pkg" / "minithesis",
        help="Output directory for the test package",
    )
    p.add_argument(
        "--disable",
        type=str,
        default="",
        help="Comma-separated list of extensions to exclude (e.g. floats,text)",
    )
    args = p.parse_args()

    disabled = frozenset(m for m in args.disable.split(",") if m)
    unknown = disabled - set(EXTENSIONS)
    if unknown:
        for name in sorted(unknown):
            matches = difflib.get_close_matches(name, EXTENSIONS, n=1, cutoff=0.5)
            msg = f"Unknown extension: {name!r}"
            if matches:
                msg += f" (did you mean {matches[0]!r}?)"
            print(msg, file=sys.stderr)
        print(f"Valid extensions: {', '.join(EXTENSIONS)}", file=sys.stderr)
        sys.exit(1)
    result = compile_minithesis(disabled=disabled)

    # Write standalone compiled file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)

    # Write test package
    write_test_package(result, args.pkg, disabled)

    # Format with ruff
    targets = [str(args.output), str(args.pkg / "core.py")]
    subprocess.run(["uv", "run", "ruff", "format", *targets], check=True, cwd=ROOT)
    subprocess.run(
        ["uv", "run", "ruff", "check", "--fix", *targets], check=True, cwd=ROOT
    )

    print(f"Compiled to {args.output}")
    print(f"Test package at {args.pkg}")


if __name__ == "__main__":
    main()
