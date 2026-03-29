# Minithesis Development Guide

## Project structure

```
src/minithesis/
  core.py            — standalone core (integers + booleans only)
  __init__.py        — adds float/bytes/string types, re-exports public API
  generators.py      — user-facing generator functions
tests/
  conftest.py        — database isolation fixture
  test_core.py       — core engine tests
  test_generators.py — generator function tests
  test_floats.py     — float tests
  test_text.py       — text/string tests
  test_bytes.py      — bytes tests
  test_hypothesis.py — Hypothesis meta-test
```

## Commands

* `just test` — run tests with 100% branch coverage requirement
* `just typecheck` — run pyright (must be 0 errors)
* `just format` — run ruff format + lint

All three must pass before committing.

## Code conventions

* **100% branch coverage is mandatory.** Do not lower the threshold. If a branch appears uncoverable, either write a test that covers it, convert unreachable branches to asserts, or restructure the code.
* **No `type: ignore` comments.** Fix the types properly (e.g. add stub methods, use proper signatures).
* **No ruff suppression comments or per-file ignores** unless absolutely unavoidable.
* **No `import X as X`** for re-exports. Use `__all__` in `__init__.py` to declare the public API.
* **No wildcard imports.**
* **No unused variable suppression** (`_var` prefix). Drop the assignment, or use `let _ = ...` if the expression must stay.
* **`__all__` only in `__init__.py`**, not in submodules.

## Public API vs internals

* **Public API** (in `__all__`): `run_test`, `TestCase`, `Generator`, `Unsatisfiable`, `Database`, `DirectoryDB`.
* Everything else is internal. Tests should import internals from `minithesis.core` directly, not via the package.
* Do not add internal names to `__all__` for testing convenience.

## Architecture

* **Shrink passes** are registered via the `@shrink_pass` decorator into the global `SHRINK_PASSES` list. Each pass takes a `TestingState` and uses `state.consider()` / `state.replace()`.
* **Serialization** uses a registry (`register_serializer`). Tags are in the `SerializationTag` enum.
* **Type extensions** (float, bytes, string) monkey-patch draw methods onto `TestCase` and register their serializers and shrink passes when `__init__.py` is imported.
* **`minithesis.py`** must stand alone with no dependencies on the type extensions.

## Testing

* Tests that need to monkeypatch module constants (like `BUFFER_SIZE`, `_DEFAULT_DATABASE_PATH`) should target `minithesis.core` (the module where the constant is defined and read).
* `NAN_DRAW_PROBABILITY` is defined in `__init__.py` and can be monkeypatched via `import minithesis; monkeypatch.setattr(minithesis, ...)`.
* The database isolation fixture in `conftest.py` ensures tests don't share state via `.minithesis-cache`. Each test gets a fresh tmp directory.
* Coverage must be deterministically 100%. If a path depends on randomness, boost the probability via monkeypatch rather than increasing `max_examples`.

## Commit discipline

* Work in small commits. Tests must pass at each commit.
* When the user asks to amend a specific commit, make sure to target the right one. Keep the latest commit as a low-stakes one (like .gitignore) so that amending defaults to the right place.
* Do not rewrite history across many commits unless specifically asked. Prefer fixup commits.
* Commit messages should be concise and descriptive. End with `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`.
* Do not include CLAUDE.md changes in code commits. CLAUDE.md updates should be in their own separate commits.

## Style

* Prefer clarity over cleverness.
* Comments are important — keep them accurate as code changes.
* When conditions are checked outside a function (e.g. bounded vs unbounded), define separate functions rather than checking inside one function body.
* Don't preserve backwards compatibility for its own sake. We have no external consumers.
