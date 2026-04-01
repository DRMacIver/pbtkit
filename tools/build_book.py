"""Build the mdbook for pbtkit.

Reads the individual source module files and assembles markdown pages
with the code embedded, alongside human/AI-written prose from the
content directory.

Usage:
    uv run python tools/build_book.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "pbtkit"
BOOK_SRC = ROOT / "book" / "src"

# Each entry: (source_files, page_filename, title)
# source_files is a list of paths relative to src/pbtkit/.
PAGES: list[tuple[list[str], str, str]] = [
    (["core.py"], "core", "Core"),
    (["database.py"], "database", "Database"),
    (["caching.py"], "caching", "Caching"),
    (["targeting.py"], "targeting", "Targeting"),
    (["collections.py"], "collections", "Collections"),
    (["floats.py"], "floats", "Floats"),
    (["bytes.py"], "bytes", "Bytes"),
    (["text.py"], "text", "Text"),
    (
        ["shrinking/advanced_integer_passes.py"],
        "shrink_integers",
        "Shrinking: Advanced Integer Passes",
    ),
    (["shrinking/bind_deletion.py"], "shrink_bind", "Shrinking: Bind Deletion"),
    (
        ["shrinking/duplication_passes.py"],
        "shrink_duplication",
        "Shrinking: Duplication Passes",
    ),
    (
        ["shrinking/sorting.py"],
        "shrink_sorting",
        "Shrinking: Sorting",
    ),
    (
        ["shrinking/sequence.py", "shrinking/sequence_redistribution.py"],
        "shrink_sequence",
        "Shrinking: Sequence Passes",
    ),
    (["shrinking/index_passes.py"], "shrink_index", "Shrinking: Index Passes"),
    (["shrinking/mutation.py"], "shrink_mutation", "Shrinking: Mutation"),
    (
        ["shrinking/advanced_bytes_passes.py"],
        "shrink_bytes",
        "Shrinking: Advanced Bytes Passes",
    ),
    (
        ["shrinking/advanced_string_passes.py"],
        "shrink_strings",
        "Shrinking: Advanced String Passes",
    ),
    (["draw_names.py"], "draw_names", "Draw Names"),
]


def build_page(source_files: list[str], page_name: str, title: str) -> str:
    """Build the markdown content for one page."""
    parts = [f"# {title}\n"]

    # Include human-written content if it exists.
    content_file = BOOK_SRC / "content" / f"{page_name}.md"
    if content_file.exists():
        parts.append(content_file.read_text())
    else:
        parts.append(
            '<div class="ai">\n\n'
            "*This page does not yet have human-written content. "
            "The summary below was written by an LLM.*\n\n"
            "</div>\n"
        )

    # Embed each source file.
    for rel_path in source_files:
        source = (SRC / rel_path).read_text()
        label = f"src/pbtkit/{rel_path}"
        parts.append(f"\n## `{label}`\n\n```python\n{source}```\n")

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
    for source_files, page_name, title in PAGES:
        print(f"Building {page_name}...", flush=True)
        md = build_page(source_files, page_name, title)
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
