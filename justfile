extensions := `uv run python -c "from tools.compile_pbtkit import EXTENSIONS; print(' '.join(EXTENSIONS))"`

test:
    uv run python -m coverage run --source=src/pbtkit --branch -m pytest tests/ --ff --maxfail=1 -m 'not hypothesis' --durations=100 --verbose
    PBTKIT_DISABLED=edge_case_boosting uv run python -m coverage run --append --source=src/pbtkit --branch -m pytest tests/ -m 'not hypothesis' -q --no-header
    uv run coverage report --show-missing --fail-under=100

test-core:
    for ext in {{extensions}}; do \
        echo "--- Disabling: $ext ---" && \
        PBTKIT_DISABLED=$ext uv run pytest tests/ -m 'not hypothesis' --verbose || exit 1; \
    done

compile:
    uv run python tools/compile_pbtkit.py

test-features:
    uv run python tools/test_features.py

test-compiled:
    uv run python -u tools/test_compiled.py

typecheck:
    uv run pyright src/

format:
    uv run ruff format src/ tests/
    uv run ruff check --fix src/ tests/

check: typecheck test
