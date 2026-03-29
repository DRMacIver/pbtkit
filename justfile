test:
    uv run python -m coverage run --source=src/minithesis --branch -m pytest tests/ --ff --maxfail=1 -m 'not hypothesis' --durations=100 --verbose
    uv run coverage report --show-missing --fail-under=100

test-core:
    MINITHESIS_DISABLED=floats uv run pytest tests/ -m 'not hypothesis' --verbose
    MINITHESIS_DISABLED=bytes uv run pytest tests/ -m 'not hypothesis' --verbose
    MINITHESIS_DISABLED=text uv run pytest tests/ -m 'not hypothesis' --verbose
    MINITHESIS_DISABLED=collections uv run pytest tests/ -m 'not hypothesis' --verbose

typecheck:
    uv run pyright src/

format:
    uv run ruff format src/ tests/
    uv run ruff check --fix src/ tests/
