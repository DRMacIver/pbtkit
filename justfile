test:
    uv run python -m coverage run --include=minithesis.py --branch -m pytest test_minithesis.py --ff --maxfail=1 -m 'not hypothesis' --durations=100 --verbose
    uv run coverage report --show-missing --fail-under=100

typecheck:
    uv run pyright minithesis.py

format:
    uv run ruff format *.py
    uv run ruff check --fix *.py
