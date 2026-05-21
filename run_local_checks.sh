#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

echo "Running local validation for core-llm-bridge..."

pdm run ruff check src tests
pdm run black --check src examples tests
pdm run mypy src tests
pdm run pytest -q

echo "All local checks passed!"
