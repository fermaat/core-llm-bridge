.PHONY: help install install-dev test test-unit test-integration coverage lint format mypy clean

help:
	@echo "llm-bridge Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install              Install base dependencies"
	@echo "  make install-dev          Install all dependencies including dev"
	@echo ""
	@echo "Testing:"
	@echo "  make test                 Run all tests"
	@echo "  make test-unit            Run unit tests only"
	@echo "  make test-integration     Run integration tests (requires Ollama)"
	@echo "  make coverage             Generate coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint                 Run all linters"
	@echo "  make format               Format code with black"
	@echo "  make mypy                 Run type checker"
	@echo "  make clean                Remove build artifacts and cache"

install:
	pdm install

install-dev:
	pdm install -d

test:
	pdm run pytest

test-unit:
	pdm run pytest -m unit

test-integration:
	pdm run pytest -m integration

coverage:
	pdm run pytest --cov --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint: format mypy
	pdm run ruff check src tests

format:
	pdm run black src tests

mypy:
	pdm run mypy src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned up build artifacts and cache"
