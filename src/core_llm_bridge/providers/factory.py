"""Provider factory utilities for core_llm_bridge."""

from typing import Any

from core_llm_bridge.core.base import BaseLLMProvider
from core_llm_bridge.exceptions import ProviderNotAvailableError

from .ollama import OllamaProvider

_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
}


def register_provider(name: str, provider_cls: type[BaseLLMProvider]) -> None:
    """Register a new provider class for factory creation."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Provider name must not be empty")
    _PROVIDERS[normalized] = provider_cls


def get_supported_providers() -> list[str]:
    """Return the list of supported provider names."""
    return sorted(_PROVIDERS.keys())


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
    if normalized in _PROVIDERS:
        return _PROVIDERS[normalized](**kwargs)

    supported = ", ".join(get_supported_providers())
    raise ProviderNotAvailableError(f"Unknown provider '{name}'. Supported providers: {supported}")
