#!/usr/bin/env python3
"""Run coverage analysis for each feature compiled in isolation.

For each extension, compiles with only that feature, runs tests with
coverage, and reports uncovered lines in the compiled core.py.
"""

from __future__ import annotations

import os
import subprocess

from compile_minithesis import EXTENSIONS, ROOT, BUILD, expand_disabled


def run_feature_coverage(ext: str) -> str | None:
    """Compile with only this feature and run coverage. Returns missing lines or None on failure."""
    # Compile
    result = subprocess.run(
        ["uv", "run", "python", "tools/compile_minithesis.py", f"--features={ext}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    # Compute disabled set
    disabled = expand_disabled(frozenset(set(EXTENSIONS) - {ext}))
    env = {**os.environ, "MINITHESIS_DISABLED": ",".join(sorted(disabled))}

    # Run tests with coverage
    subprocess.run(
        [
            "uv", "run", "python", "-m", "coverage", "run",
            f"--source={BUILD / 'pkg' / 'minithesis'}",
            "--branch",
            "-m", "pytest", "tests/",
            "-m", "not hypothesis",
            f"--override-ini=pythonpath={BUILD / 'pkg'}",
            "-q",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    # Get coverage report
    pkg_core = str(BUILD / "pkg" / "minithesis" / "core.py")
    result = subprocess.run(
        [
            "uv", "run", "coverage", "report",
            "--show-missing",
            f"--include={pkg_core}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    # Extract missing lines from the report
    for line in result.stdout.splitlines():
        if "core.py" in line and "TOTAL" not in line:
            # Format: Name  Stmts  Miss  Branch  BrPart  Cover  Missing
            # Find the percentage and everything after it
            parts = line.split()
            # Find the cover% column (ends with %)
            for i, p in enumerate(parts):
                if p.endswith("%"):
                    cover = p
                    missing = " ".join(parts[i + 1:])
                    return f"{cover} missing: {missing}" if missing else cover
    return "no data"


def main() -> None:
    for ext in EXTENSIONS:
        result = run_feature_coverage(ext)
        if result is None:
            print(f"{ext}: COMPILE FAILED")
        else:
            print(f"{ext}: {result}")


if __name__ == "__main__":
    main()
