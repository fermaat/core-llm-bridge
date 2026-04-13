"""
Custom exceptions for llm-bridge.
"""


class LLMBridgeError(Exception):
    """Base exception for all llm-bridge errors."""

    pass


class ProviderError(LLMBridgeError):
    """Base exception for provider-related errors."""

    pass


class LLMProviderError(ProviderError):
    """Exception raised when the LLM provider is unavailable or unhealthy."""

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


class AnthropicError(ProviderError):
    """Exception raised for Anthropic-specific errors."""

    pass


class AnthropicConnectionError(AnthropicError):
    """Exception raised when unable to connect to the Anthropic API."""

    pass


class AnthropicRateLimitError(AnthropicError):
    """Exception raised when the Anthropic API rate limit is exceeded."""

    pass


class AnthropicAuthError(AnthropicError):
    """Exception raised when the Anthropic API key is invalid or missing."""

    pass


class OpenAIError(ProviderError):
    """Exception raised for OpenAI-specific errors."""

    pass


class OpenAIConnectionError(OpenAIError):
    """Exception raised when unable to connect to the OpenAI API."""

    pass


class OpenAIRateLimitError(OpenAIError):
    """Exception raised when the OpenAI API rate limit is exceeded."""

    pass


class OpenAIAuthError(OpenAIError):
    """Exception raised when the OpenAI API key is invalid or missing."""

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
    "LLMProviderError",
    "ProviderNotAvailableError",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaModelNotFoundError",
    "AnthropicError",
    "AnthropicConnectionError",
    "AnthropicRateLimitError",
    "AnthropicAuthError",
    "OpenAIError",
    "OpenAIConnectionError",
    "OpenAIRateLimitError",
    "OpenAIAuthError",
    "ConfigurationError",
    "TokenLimitError",
    "InvalidMessageError",
]
