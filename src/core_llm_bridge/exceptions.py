"""
Custom exceptions for llm-bridge.
"""


class LLMBridgeError(Exception):
    """Base exception for all llm-bridge errors."""

    pass


class ProviderError(LLMBridgeError):
    """Base exception for provider-related errors."""

    pass


class ProviderNotAvailableError(ProviderError):
    """Exception raised when the requested provider is unavailable."""

    pass


class OllamaError(ProviderError):
    """Exception raised for Ollama-specific errors."""

    pass


class OllamaConnectionError(OllamaError):
    """Exception raised when unable to connect to Ollama."""

    pass


class OllamaTimeoutError(OllamaError):
    """Exception raised when Ollama request times out."""

    pass


class OllamaModelNotFoundError(OllamaError):
    """Exception raised when requested model is not available in Ollama."""

    pass


class ConfigurationError(LLMBridgeError):
    """Exception raised for configuration-related errors."""

    pass


class TokenLimitError(LLMBridgeError):
    """Exception raised when token limit is exceeded."""

    pass


class InvalidMessageError(LLMBridgeError):
    """Exception raised for invalid message format."""

    pass


__all__ = [
    "LLMBridgeError",
    "ProviderError",
    "ProviderNotAvailableError",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaModelNotFoundError",
    "ConfigurationError",
    "TokenLimitError",
    "InvalidMessageError",
]
