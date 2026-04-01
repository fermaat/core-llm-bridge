"""Unit tests for Ollama provider error handling."""

import httpx
import pytest

from core_llm_bridge.core.models import ConversationBuffer
from core_llm_bridge.exceptions import OllamaConnectionError, OllamaTimeoutError
from core_llm_bridge.providers.ollama import OllamaProvider


def test_validate_connection_raises_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(model="test-model")

    def fake_get(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(provider.client, "get", fake_get)

    with pytest.raises(OllamaTimeoutError):
        provider.validate_connection(raise_on_error=True)


def test_generate_raises_connection_error_when_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(model="test-model")

    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("failed to connect")

    monkeypatch.setattr(provider.client, "get", fake_get)

    with pytest.raises(OllamaConnectionError):
        provider.generate("Hello", ConversationBuffer())
