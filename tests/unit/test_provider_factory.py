"""Unit tests for provider factory utilities."""

import pytest

from core_llm_bridge.core.base import BaseLLMProvider
from core_llm_bridge.exceptions import ProviderNotAvailableError
from core_llm_bridge.providers import create_provider


def test_create_ollama_provider() -> None:
    provider = create_provider("ollama", model="test-model")
    assert isinstance(provider, BaseLLMProvider)
    assert provider.model == "test-model"


def test_create_unknown_provider_raises() -> None:
    with pytest.raises(ProviderNotAvailableError):
        create_provider("unknown_provider")
