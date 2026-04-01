"""Provider factory utilities for core_llm_bridge."""

from typing import Any

from core_llm_bridge.core.base import BaseLLMProvider
from core_llm_bridge.exceptions import ProviderNotAvailableError

from .ollama import OllamaProvider


def create_provider(name: str, **kwargs: Any) -> BaseLLMProvider:
    """Create a provider instance by name.

    Args:
        name: Provider name (case-insensitive)
        **kwargs: Provider-specific arguments

    Returns:
        BaseLLMProvider instance

    Raises:
        ProviderNotAvailableError: If the provider is not supported or unavailable.
    """
    normalized = name.strip().lower()

    if normalized == "ollama":
        return OllamaProvider(**kwargs)

    raise ProviderNotAvailableError(
        f"Unknown provider '{name}'. Supported providers: ollama"
    )
