extensions := `uv run python -c "from tools.compile_minithesis import EXTENSIONS; print(' '.join(EXTENSIONS))"`

test:
    uv run python -m coverage run --source=src/minithesis --branch -m pytest tests/ --ff --maxfail=1 -m 'not hypothesis' --durations=100 --verbose
    uv run coverage report --show-missing --fail-under=100

test-core:
    for ext in {{extensions}}; do \
        MINITHESIS_DISABLED=$ext uv run pytest tests/ -m 'not hypothesis' --verbose || exit 1; \
    done

compile:
    uv run python tools/compile_minithesis.py

test-compiled: compile
    uv run pytest tests/ -m 'not hypothesis' --override-ini='pythonpath=build/pkg' --verbose
    for ext in {{extensions}}; do \
        uv run python tools/compile_minithesis.py --disable=$ext && \
        MINITHESIS_DISABLED=$ext uv run pytest tests/ -m 'not hypothesis' --override-ini='pythonpath=build/pkg' --verbose || exit 1; \
    done

typecheck:
    uv run pyright src/

format:
    uv run ruff format src/ tests/
    uv run ruff check --fix src/ tests/
