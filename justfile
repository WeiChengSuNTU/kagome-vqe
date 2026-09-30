# Common tasks. Install `just` (https://just.systems) or copy the commands by hand.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Create the virtualenv and install everything (runtime + dev + notebooks).
setup:
    uv sync --all-groups
    uv run pre-commit install

# Lint and format check.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint findings and reformat.
fmt:
    uv run ruff check . --fix
    uv run ruff format .

# Static typing.
typecheck:
    uv run mypy

# Fast test suite (unit + cheap integration).
test *args:
    uv run pytest -n auto {{args}}

# Everything except tests that need IBM credentials.
test-all:
    uv run pytest -n auto -m "not ibm" -o addopts="--strict-markers -ra"

# Lint, types and fast tests: what CI runs on every push.
check: lint typecheck test

# Exact diagonalisation of the kagome cell.
exact:
    uv run kagome-vqe exact

# The paper's four combinations on the exact simulator.
bench *args:
    uv run kagome-vqe benchmark configs/benchmark.toml {{args}}

# The same on a noisy simulation of ibmq_guadalupe (takes ~1 h).
bench-fake *args:
    uv run kagome-vqe benchmark configs/benchmark_fake_guadalupe.toml {{args}}

# Plot one or more results directories.
plot +dirs:
    uv run kagome-vqe plot {{dirs}} --out convergence.png

# Remove caches and build artefacts (keeps results/).
clean:
    rm -rf .ruff_cache .mypy_cache .pytest_cache .hypothesis .coverage coverage.xml dist build
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
