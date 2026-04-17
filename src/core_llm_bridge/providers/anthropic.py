"""
Anthropic LLM provider implementation.

Provides integration with the Anthropic API (Claude models).

Requires:
    - ANTHROPIC_API_KEY set in .env or environment
    - Model name (default: claude-sonnet-4-6)
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any

import anthropic

from core_llm_bridge.exceptions import (
    AnthropicAPIError,
    AnthropicAuthError,
    AnthropicConnectionError,
    AnthropicRateLimitError,
)
from core_utils.logger import logger

from ..core.base import BaseLLMProvider
from ..core.models import BridgeResponse, ConversationBuffer, LLMConfig, MessageRole


class AnthropicProvider(BaseLLMProvider):
    """
    LLM provider for the Anthropic API (Claude models).

    Handles the Anthropic-specific message format where the system prompt
    is passed as a top-level parameter rather than as a message in the array.

    Prerequisites:
        - ANTHROPIC_API_KEY set in .env or environment
        - Model available (e.g. claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5-20251001)

    Example:
        >>> provider = AnthropicProvider(model="claude-sonnet-4-6")
        >>> if not provider.validate_connection():
        ...     print("Check your ANTHROPIC_API_KEY!")
        >>> response = provider.generate("Hello", ConversationBuffer())
        >>> print(response.text)

    Attributes:
        model: Anthropic model identifier
        api_key: Anthropic API key
        timeout: Request timeout in seconds
        client: Synchronous Anthropic client
        async_client: Asynchronous Anthropic client
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            model: Model identifier (e.g. "claude-sonnet-4-6"). Required.
            api_key: Anthropic API key. Required.
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to parent.

        Raises:
            AnthropicAuthError: If api_key is empty.
        """
        super().__init__(model=model, **kwargs)

        if not api_key:
            raise AnthropicAuthError("Anthropic API key is required.")

        self.api_key = api_key
        self.timeout = timeout

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=float(self.timeout),
        )
        self.async_client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            timeout=float(self.timeout),
        )

        logger.debug(f"AnthropicProvider initialized: {self.model}")

    def validate_connection(self, raise_on_error: bool = False) -> bool:
        """
        Validate that the Anthropic API is reachable.

        Sends a minimal request to verify key validity and connectivity.

        Args:
            raise_on_error: If True, raise on failure instead of returning False.

        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            self.client.models.retrieve(self.model)
            logger.debug(f"Anthropic connection validated for model '{self.model}'.")
            return True
        except anthropic.AuthenticationError as exc:
            msg = "Invalid Anthropic API key."
            if raise_on_error:
                raise AnthropicAuthError(msg) from exc
            logger.error(msg)
            return False
        except anthropic.APIConnectionError as exc:
            msg = "Cannot connect to the Anthropic API."
            if raise_on_error:
                raise AnthropicConnectionError(msg) from exc
            logger.error(msg)
            return False
        except Exception as exc:
            msg = f"Error validating Anthropic connection: {exc}"
            if raise_on_error:
                raise AnthropicConnectionError(msg) from exc
            logger.error(msg)
            return False

    def _build_messages_and_system(
        self, history: ConversationBuffer
    ) -> tuple[list[dict[str, str]], str | None]:
        """
        Convert ConversationBuffer to Anthropic message format.

        Anthropic requires the system prompt as a separate top-level parameter.
        Non-system messages are passed in the messages array with alternating
        user/assistant roles.

        Returns:
            Tuple of (messages list, system prompt or None).
        """
        system_prompt = history.system_prompt or None
        messages = []

        for msg in history.messages:
            if msg.role == MessageRole.SYSTEM:
                # Inline system messages override the buffer-level system prompt
                system_prompt = msg.content
                continue
            messages.append({"role": msg.role.value, "content": msg.content})

        return messages, system_prompt

    def _build_kwargs(self, config: LLMConfig | None) -> dict[str, Any]:
        """Build extra keyword arguments from LLMConfig."""
        kwargs: dict[str, Any] = {}
        if config:
            kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.max_tokens is not None:
                kwargs["max_tokens"] = config.max_tokens
            if config.stop_sequences:
                kwargs["stop_sequences"] = config.stop_sequences
        # Anthropic requires max_tokens; use a sensible default if not set
        kwargs.setdefault("max_tokens", 4096)
        return kwargs

    def _to_bridge_response(self, message: anthropic.types.Message) -> BridgeResponse:
        """Convert an Anthropic Message to BridgeResponse."""
        text = "".join(block.text for block in message.content if hasattr(block, "text"))
        tokens_used = (message.usage.input_tokens or 0) + (message.usage.output_tokens or 0)
        finish_reason = message.stop_reason or "stop"
        return BridgeResponse(
            text=text,
            finish_reason=finish_reason,
            tokens_used=tokens_used,
            metadata={
                "model": message.model,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "stop_reason": message.stop_reason,
            },
        )

    def generate(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Generate a response from the Anthropic API.

        Args:
            prompt: User prompt (already added to history by BridgeEngine).
            history: Full conversation history including the latest user message.
            config: Optional LLM configuration.

        Returns:
            BridgeResponse with the generated text.

        Raises:
            AnthropicAuthError: If the API key is invalid.
            AnthropicRateLimitError: If the rate limit is exceeded.
            AnthropicConnectionError: If the API cannot be reached.
        """
        messages, system_prompt = self._build_messages_and_system(history)
        kwargs = self._build_kwargs(config)

        logger.debug(f"Sending request to Anthropic: {self.model}")

        try:
            call_kwargs = {**({"system": system_prompt} if system_prompt else {}), **kwargs}
            message = self.client.messages.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                **call_kwargs,
            )
            response = self._to_bridge_response(message)
            logger.debug(f"Anthropic response received: {response.tokens_used} tokens")
            return response

        except anthropic.AuthenticationError as exc:
            raise AnthropicAuthError("Invalid Anthropic API key.") from exc
        except anthropic.RateLimitError as exc:
            raise AnthropicRateLimitError("Anthropic API rate limit exceeded.") from exc
        except anthropic.APIConnectionError as exc:
            raise AnthropicConnectionError("Cannot connect to the Anthropic API.") from exc
        except anthropic.APIStatusError as exc:
            raise AnthropicAPIError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc

    def generate_stream(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> Generator[BridgeResponse, None, None]:
        """
        Generate a streaming response from the Anthropic API.

        Yields text chunks as they arrive, followed by a final chunk with
        finish_reason="stop" and token usage.

        Args:
            prompt: User prompt.
            history: Full conversation history.
            config: Optional LLM configuration.

        Yields:
            BridgeResponse objects with partial text.

        Raises:
            AnthropicAuthError: If the API key is invalid.
            AnthropicRateLimitError: If the rate limit is exceeded.
            AnthropicConnectionError: If the API cannot be reached.
        """
        messages, system_prompt = self._build_messages_and_system(history)
        kwargs = self._build_kwargs(config)

        logger.debug(f"Sending streaming request to Anthropic: {self.model}")

        try:
            call_kwargs = {**({"system": system_prompt} if system_prompt else {}), **kwargs}
            with self.client.messages.stream(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                **call_kwargs,
            ) as stream:
                for text_chunk in stream.text_stream:
                    yield BridgeResponse(text=text_chunk, finish_reason="incomplete")

                # Final chunk with usage stats
                final_message = stream.get_final_message()
                tokens_used = (final_message.usage.input_tokens or 0) + (
                    final_message.usage.output_tokens or 0
                )
                yield BridgeResponse(
                    text="",
                    finish_reason=final_message.stop_reason or "stop",
                    tokens_used=tokens_used,
                    metadata={
                        "model": final_message.model,
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                    },
                )

        except anthropic.AuthenticationError as exc:
            raise AnthropicAuthError("Invalid Anthropic API key.") from exc
        except anthropic.RateLimitError as exc:
            raise AnthropicRateLimitError("Anthropic API rate limit exceeded.") from exc
        except anthropic.APIConnectionError as exc:
            raise AnthropicConnectionError("Cannot connect to the Anthropic API.") from exc
        except anthropic.APIStatusError as exc:
            raise AnthropicAPIError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc

    async def generate_async(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Async version of generate using AsyncAnthropic client.

        Args:
            prompt: User prompt.
            history: Full conversation history.
            config: Optional LLM configuration.

        Returns:
            BridgeResponse with the generated text.
        """
        messages, system_prompt = self._build_messages_and_system(history)
        kwargs = self._build_kwargs(config)

        logger.debug(f"Sending async request to Anthropic: {self.model}")

        try:
            call_kwargs = {**({"system": system_prompt} if system_prompt else {}), **kwargs}
            message = await self.async_client.messages.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                **call_kwargs,
            )
            return self._to_bridge_response(message)

        except anthropic.AuthenticationError as exc:
            raise AnthropicAuthError("Invalid Anthropic API key.") from exc
        except anthropic.RateLimitError as exc:
            raise AnthropicRateLimitError("Anthropic API rate limit exceeded.") from exc
        except anthropic.APIConnectionError as exc:
            raise AnthropicConnectionError("Cannot connect to the Anthropic API.") from exc
        except anthropic.APIStatusError as exc:
            raise AnthropicAPIError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc

    async def generate_stream_async(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> AsyncGenerator[BridgeResponse, None]:
        """
        Async streaming version using AsyncAnthropic client.

        Args:
            prompt: User prompt.
            history: Full conversation history.
            config: Optional LLM configuration.

        Yields:
            BridgeResponse objects with partial text.
        """
        messages, system_prompt = self._build_messages_and_system(history)
        kwargs = self._build_kwargs(config)

        logger.debug(f"Sending async streaming request to Anthropic: {self.model}")

        try:
            call_kwargs = {**({"system": system_prompt} if system_prompt else {}), **kwargs}
            async with self.async_client.messages.stream(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                **call_kwargs,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    yield BridgeResponse(text=text_chunk, finish_reason="incomplete")

                final_message = await stream.get_final_message()
                tokens_used = (final_message.usage.input_tokens or 0) + (
                    final_message.usage.output_tokens or 0
                )
                yield BridgeResponse(
                    text="",
                    finish_reason=final_message.stop_reason or "stop",
                    tokens_used=tokens_used,
                    metadata={
                        "model": final_message.model,
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                    },
                )

        except anthropic.AuthenticationError as exc:
            raise AnthropicAuthError("Invalid Anthropic API key.") from exc
        except anthropic.RateLimitError as exc:
            raise AnthropicRateLimitError("Anthropic API rate limit exceeded.") from exc
        except anthropic.APIConnectionError as exc:
            raise AnthropicConnectionError("Cannot connect to the Anthropic API.") from exc
        except anthropic.APIStatusError as exc:
            raise AnthropicAPIError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model})"


__all__ = ["AnthropicProvider"]
