"""
Abstract base class and protocol for LLM providers.

Defines the interface that all provider implementations must follow.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Any

from .models import BridgeResponse, ConversationBuffer, LLMConfig


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Defines the interface that all LLM provider implementations must follow.
    Providers handle the actual communication with specific LLMs or services.

    Example:
        >>> class MyProvider(BaseLLMProvider):
        ...     def generate(self, prompt: str, config: LLMConfig) -> str:
        ...         # Implement actual LLM call
        ...         pass
        ...
        ...     async def generate_async(self, prompt: str, config: LLMConfig) -> str:
        ...         # Implement async version
        ...         pass
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        """
        Initialize the provider.

        Args:
            model: The model identifier to use
            **kwargs: Additional provider-specific configuration
        """
        self.model = model
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            history: The conversation history
            config: Optional LLM configuration

        Returns:
            BridgeResponse with the generated text and metadata

        Raises:
            ProviderError: If the generation fails
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> Generator[BridgeResponse, None, None]:
        """
        Generate a response from the LLM with streaming.

        Yields tokens/chunks as they become available.

        Args:
            prompt: The user prompt
            history: The conversation history
            config: Optional LLM configuration

        Yields:
            BridgeResponse objects with partial text

        Raises:
            ProviderError: If the generation fails
        """
        pass

    async def generate_async(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Async version of generate.

        Default implementation calls the sync version.
        Can be overridden for true async support.

        Args:
            prompt: The user prompt
            history: The conversation history
            config: Optional LLM configuration

        Returns:
            BridgeResponse with the generated text and metadata
        """
        return self.generate(prompt, history, config)

    async def generate_stream_async(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> AsyncGenerator[BridgeResponse, None]:
        """
        Async version of generate_stream.

        Default implementation wraps sync streaming.
        Can be overridden for true async support.

        Args:
            prompt: The user prompt
            history: The conversation history
            config: Optional LLM configuration

        Yields:
            BridgeResponse objects with partial text
        """
        for response in self.generate_stream(prompt, history, config):
            yield response

    def validate_connection(self, raise_on_error: bool = False) -> bool:
        """
        Validate that the provider can be reached.

        Args:
            raise_on_error: If True, raise provider-specific errors instead of returning False.

        Should attempt a simple health check or connection test.

        Returns:
            True if provider is reachable, False otherwise

        Example:
            >>> provider = OllamaProvider(model="llama2")
            >>> if provider.validate_connection():
            ...     print("Connected!")
        """
        return True

    def health_check(self, raise_on_error: bool = False) -> bool:
        """
        Check that the provider is healthy and ready to serve requests.

        Args:
            raise_on_error: If True, raise provider-specific errors on failure.

        Returns:
            True if the provider is healthy, False otherwise
        """
        return self.validate_connection(raise_on_error=raise_on_error)

    def get_model_info(self) -> dict[str, Any] | None:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information or None if not available
        """
        return None

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return f"{self.__class__.__name__}(model={self.model})"


class ToolProvider(ABC):
    """
    Optional mixin for providers that support tool/function calling.

    Implement this to add tool/function calling support to a provider.
    """

    @abstractmethod
    def register_tool(self, name: str, description: str, schema: dict[str, Any]) -> None:
        """
        Register a tool that the LLM can call.

        Args:
            name: Name of the tool/function
            description: Description of what the tool does
            schema: JSON schema of the tool's parameters
        """
        pass

    @abstractmethod
    def get_registered_tools(self) -> list[dict[str, Any]]:
        """
        Get list of all registered tools.

        Returns:
            List of tool definitions
        """
        pass


__all__ = [
    "BaseLLMProvider",
    "ToolProvider",
]
