#!/usr/bin/env python3
"""Test all compilation variants: full, each-disabled, each-enabled, minimal.

For each variant:
1. Compile with the appropriate flags
2. Typecheck the compiled code with pyright
3. Run tests with branch coverage
4. Require 100% coverage of the compiled core.py
"""

from __future__ import annotations

import os
import subprocess
import sys

from compile_minithesis import EXTENSIONS, ROOT, BUILD, expand_disabled


def _compile(args: list[str]) -> bool:
    result = subprocess.run(
        ["uv", "run", "python", "tools/compile_minithesis.py"] + args,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  COMPILE FAILED\n{result.stderr}")
        return False
    return True


def _typecheck() -> bool:
    pkg = BUILD / "pkg"
    result = subprocess.run(
        ["uv", "run", "pyright", str(pkg / "minithesis" / "core.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if "error" in line.lower() and "0 error" not in line:
            print(f"  TYPECHECK FAILED\n{result.stdout}")
            return False
    return True


def _test_with_coverage(disabled: frozenset[str]) -> bool:
    pkg = BUILD / "pkg"
    env = {**os.environ}
    if disabled:
        env["MINITHESIS_DISABLED"] = ",".join(sorted(disabled))

    try:
        result = subprocess.run(
            [
                "uv", "run", "python", "-m", "coverage", "run",
                f"--source={pkg / 'minithesis'}",
                "--branch",
                "-m", "pytest", "tests/",
                "-m", "not hypothesis",
                f"--override-ini=pythonpath={pkg}",
                "-q",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT (>30s)")
        return False
    test_output = result.stdout + result.stderr
    if result.returncode != 0:
        for line in test_output.splitlines():
            if "FAILED" in line or "Error" in line:
                print(f"  {line}")
        if "passed" not in test_output:
            print(f"  Tests crashed")
        return False

    result = subprocess.run(
        [
            "uv", "run", "coverage", "report",
            "--show-missing",
            f"--include={pkg / 'minithesis' / 'core.py'}",
            "--fail-under=100",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        for line in result.stdout.splitlines():
            if "core.py" in line and "TOTAL" not in line:
                print(f"  {line.strip()}")
        return False
    return True


def run_variant(label: str, compile_args: list[str], disabled: frozenset[str]) -> bool:
    if not _compile(compile_args):
        return False
    if not _typecheck():
        return False
    if not _test_with_coverage(disabled):
        return False
    return True


def main() -> None:
    survey = "--survey" in sys.argv
    count = 0
    failed: list[str] = []

    def check(label: str, compile_args: list[str], disabled: frozenset[str]) -> None:
        nonlocal count
        print(label, flush=True)
        if not run_variant(label, compile_args, disabled):
            if survey:
                failed.append(label)
                return
            print(f"\nFAILED: {label}")
            sys.exit(1)
        print("  OK")
        count += 1

    check("full", [], frozenset())

    for ext in EXTENSIONS:
        disabled = expand_disabled(frozenset([ext]))
        check(f"disable={ext}", [f"--disable={ext}"], disabled)

    for ext in EXTENSIONS:
        disabled = expand_disabled(frozenset(set(EXTENSIONS) - {ext}))
        check(f"feature={ext}", [f"--features={ext}"], disabled)

    disabled = expand_disabled(frozenset(EXTENSIONS))
    check("minimal", ["--minimal"], disabled)

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nAll {count} variants passed!")


if __name__ == "__main__":
    main()
