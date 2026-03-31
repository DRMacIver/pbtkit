#!/usr/bin/env python3
"""Test each feature in isolation.

For each extension, compiles minithesis with only that feature enabled,
then runs the test suite against the compiled output.
"""

from __future__ import annotations

import os
import subprocess
import sys

from compile_minithesis import EXTENSIONS, ROOT, expand_disabled


def main() -> None:
    failed: list[str] = []
    for ext in EXTENSIONS:
        print(f"\n{'='*60}")
        print(f"Feature: {ext}")
        print(f"{'='*60}")

        # Compile with only this feature
        result = subprocess.run(
            ["uv", "run", "python", "tools/compile_minithesis.py", f"--features={ext}"],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"COMPILE FAILED for {ext}")
            failed.append(ext)
            continue

        # Compute disabled set for MINITHESIS_DISABLED env var
        disabled = expand_disabled(frozenset(set(EXTENSIONS) - {ext}))
        env = {**os.environ, "MINITHESIS_DISABLED": ",".join(sorted(disabled))}

        result = subprocess.run(
            [
                "uv", "run", "pytest", "tests/",
                "-m", "not hypothesis",
                "--override-ini=pythonpath=build/pkg",
                "--verbose",
                "-x",
            ],
            cwd=ROOT,
            env=env,
        )
        if result.returncode != 0:
            print(f"TESTS FAILED for {ext}")
            failed.append(ext)
            continue

        print(f"PASSED: {ext}")

    if failed:
        print(f"\n{'='*60}")
        print(f"FAILED features: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("All features passed!")


if __name__ == "__main__":
    main()
