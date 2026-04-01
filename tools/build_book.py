"""Build the mdbook for pbtkit.

Compiles each feature variant using compile_pbtkit.py, strips the
generator library, and assembles markdown pages with the code embedded.

Usage:
    uv run python tools/build_book.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK_SRC = ROOT / "book" / "src"

# Each entry: (feature_flag, page_filename, title)
# feature_flag is passed to --features (empty string means --minimal).
PAGES: list[tuple[str, str, str]] = [
    ("", "core", "Core"),
    ("database", "database", "Database"),
    ("caching", "caching", "Caching"),
    ("targeting", "targeting", "Targeting"),
    ("collections", "collections", "Collections"),
    ("floats", "floats", "Floats"),
    ("bytes", "bytes", "Bytes"),
    ("text", "text", "Text"),
    (
        "shrinking.advanced_integer_passes",
        "shrink_integers",
        "Shrinking: Advanced Integer Passes",
    ),
    ("shrinking.bind_deletion", "shrink_bind", "Shrinking: Bind Deletion"),
    (
        "shrinking.duplication_passes",
        "shrink_duplication",
        "Shrinking: Duplication Passes",
    ),
    ("shrinking.sorting", "shrink_sorting", "Shrinking: Sorting"),
    ("shrinking.index_passes", "shrink_index", "Shrinking: Index Passes"),
    ("shrinking.mutation", "shrink_mutation", "Shrinking: Mutation"),
    ("draw_names", "draw_names", "Draw Names"),
]


def compile_variant(feature: str) -> str:
    """Compile a pbtkit variant and return the source code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "pbtkit.py"
        pkg = Path(tmpdir) / "pkg"
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "compile_pbtkit.py"),
            "-o",
            str(out),
            "--pkg",
            str(pkg),
        ]
        if feature:
            cmd += ["--features", feature]
        else:
            cmd.append("--minimal")
        subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
        return out.read_text()


def strip_generators(source: str) -> str:
    """Remove the generator library section from compiled output.

    Generator functions (integers, just, nothing, etc.) are defined at the
    end of the compiled file.  We look for the first top-level ``def`` or
    ``class`` that defines a Generator-returning function after the main
    infrastructure, and strip everything from there to the end.

    The heuristic: find the first ``def`` whose body contains
    ``Generator[`` that appears after PbtkitState — these are the
    user-facing generator functions.
    """
    lines = source.splitlines(keepends=True)

    # Find the end of the PbtkitState class and infrastructure.
    # Generator functions are the last section in the compiled output.
    # We look for the first top-level def that returns a Generator.
    generator_start = None
    for i, line in enumerate(lines):
        # Generator functions are top-level defs returning Generator[...]
        if line.startswith("def ") and "Generator[" in line:
            generator_start = i
            break

    if generator_start is None:
        return source

    # Walk back to include any preceding blank lines / comments
    while generator_start > 0 and lines[generator_start - 1].strip() == "":
        generator_start -= 1

    return "".join(lines[:generator_start])


def page_content_path(page_name: str) -> Path:
    """Return the path to a page's human-written content file."""
    return BOOK_SRC / "content" / f"{page_name}.md"


def build_page(feature: str, page_name: str, title: str) -> str:
    """Build the markdown content for one page."""
    code = strip_generators(compile_variant(feature))

    parts = [f"# {title}\n"]

    # Include human-written content if it exists.
    content_file = page_content_path(page_name)
    if content_file.exists():
        parts.append(content_file.read_text())
    else:
        parts.append(
            f'<div class="ai">\n\n'
            f"*This page does not yet have human-written content. "
            f"The summary below was written by an LLM.*\n\n"
            f"</div>\n"
        )

    parts.append(f"\n## Code\n\n```python\n{code}```\n")

    # Include human-written design notes if they exist.
    design_file = BOOK_SRC / "content" / f"{page_name}_design.md"
    if design_file.exists():
        parts.append("\n## Design Considerations\n\n")
        parts.append(design_file.read_text())

    return "\n".join(parts)


def build_summary() -> str:
    """Build SUMMARY.md."""
    lines = ["# Summary\n", ""]
    for _, page_name, title in PAGES:
        lines.append(f"- [{title}](./{page_name}.md)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    # Create content directory for human-written prose
    (BOOK_SRC / "content").mkdir(exist_ok=True)

    # Generate each page
    for feature, page_name, title in PAGES:
        print(f"Building {page_name}...", flush=True)
        md = build_page(feature, page_name, title)
        (BOOK_SRC / f"{page_name}.md").write_text(md)

    # Write SUMMARY.md
    (BOOK_SRC / "SUMMARY.md").write_text(build_summary())

    # Remove the stub chapter_1.md if it exists
    stub = BOOK_SRC / "chapter_1.md"
    if stub.exists():
        stub.unlink()

    print("Done. Run 'mdbook build book/' or 'mdbook serve book/' to view.")


if __name__ == "__main__":
    main()
