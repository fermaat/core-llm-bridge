"""
Pytest configuration and fixtures.

Provides common fixtures and configuration for all tests in the project.
"""

import os
from pathlib import Path

import pytest

# Set test environment
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_LEVEL", "DEBUG")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Provide the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Provide the test data directory."""
    data_dir = project_root / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def mock_ollama_response() -> dict:
    """Provide a mock Ollama API response for testing."""
    return {
        "model": "llama2",
        "created_at": "2024-01-01T00:00:00.000000000Z",
        "response": "This is a test response from the mock Ollama provider.",
        "done": True,
        "context": [101, 102, 103],
        "total_duration": 1000000000,
        "load_duration": 100000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 500000000,
        "eval_count": 5,
        "eval_duration": 400000000,
    }


@pytest.fixture
def ollama_available() -> bool:
    """Check if Ollama is available locally."""
    import httpx

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
def skip_if_no_ollama(ollama_available: bool) -> None:
    """Skip test if Ollama is not available."""
    if not ollama_available:
        pytest.skip("Ollama is not available")
