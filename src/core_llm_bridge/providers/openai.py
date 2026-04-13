"""
OpenAI LLM provider implementation.

Provides integration with the OpenAI API (GPT models).
Also compatible with any OpenAI-compatible endpoint (e.g. local vLLM, LM Studio)
by setting OPENAI_BASE_URL.

Requires:
    - OPENAI_API_KEY set in .env or environment
    - Model name (default: gpt-4o)
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any

import openai

from core_llm_bridge.config import logger, settings
from core_llm_bridge.exceptions import (
    OpenAIAuthError,
    OpenAIConnectionError,
    OpenAIRateLimitError,
)

from ..core.base import BaseLLMProvider
from ..core.models import BridgeResponse, ConversationBuffer, LLMConfig


class OpenAIProvider(BaseLLMProvider):
    """
    LLM provider for the OpenAI API (GPT models).

    The system prompt is included as the first message with role="system",
    matching the OpenAI chat completions format.

    Also works with OpenAI-compatible endpoints (vLLM, LM Studio, Azure OpenAI, etc.)
    by setting OPENAI_BASE_URL and OPENAI_API_KEY accordingly.

    Prerequisites:
        - OPENAI_API_KEY set in .env or environment
        - Model available (e.g. gpt-4o, gpt-4o-mini, gpt-4-turbo)

    Example:
        >>> provider = OpenAIProvider(model="gpt-4o")
        >>> response = provider.generate("Hello", ConversationBuffer())
        >>> print(response.text)

    Attributes:
        model: OpenAI model identifier
        api_key: OpenAI API key
        base_url: API base URL (empty = default OpenAI endpoint)
        timeout: Request timeout in seconds
        client: Synchronous OpenAI client
        async_client: Asynchronous OpenAI client
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            model: Model identifier. Defaults to OPENAI_DEFAULT_MODEL from config.
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY from config.
            base_url: API base URL. Defaults to OPENAI_BASE_URL from config (empty = OpenAI default).
            timeout: Request timeout in seconds. Defaults to OPENAI_TIMEOUT from config.
            **kwargs: Additional arguments passed to parent.

        Raises:
            OpenAIAuthError: If no API key is provided or found in config.
        """
        if model is None:
            model = settings.openai_default_model

        super().__init__(model=model, **kwargs)

        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise OpenAIAuthError(
                "OpenAI API key is required. Set OPENAI_API_KEY in .env "
                "or pass api_key parameter."
            )

        self.base_url = base_url or settings.openai_base_url or None
        self.timeout = timeout or settings.openai_timeout

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": float(self.timeout),
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = openai.AsyncOpenAI(**client_kwargs)

        logger.debug(f"OpenAIProvider initialized: {self.model}")

    def validate_connection(self, raise_on_error: bool = False) -> bool:
        """
        Validate that the OpenAI API is reachable.

        Args:
            raise_on_error: If True, raise on failure instead of returning False.

        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            self.client.models.retrieve(self.model)
            logger.debug(f"OpenAI connection validated for model '{self.model}'.")
            return True
        except openai.AuthenticationError as exc:
            msg = "Invalid OpenAI API key."
            if raise_on_error:
                raise OpenAIAuthError(msg) from exc
            logger.error(msg)
            return False
        except openai.APIConnectionError as exc:
            msg = f"Cannot connect to the OpenAI API at {self.base_url or 'default endpoint'}."
            if raise_on_error:
                raise OpenAIConnectionError(msg) from exc
            logger.error(msg)
            return False
        except Exception as exc:
            msg = f"Error validating OpenAI connection: {exc}"
            if raise_on_error:
                raise OpenAIConnectionError(msg) from exc
            logger.error(msg)
            return False

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
                kwargs["stop"] = config.stop_sequences
        return kwargs

    def _to_bridge_response(
        self, completion: openai.types.chat.ChatCompletion
    ) -> BridgeResponse:
        """Convert an OpenAI ChatCompletion to BridgeResponse."""
        choice = completion.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason or "stop"
        tokens_used = None
        if completion.usage:
            tokens_used = completion.usage.prompt_tokens + completion.usage.completion_tokens
        return BridgeResponse(
            text=text,
            finish_reason=finish_reason,
            tokens_used=tokens_used,
            metadata={
                "model": completion.model,
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "completion_tokens": (
                    completion.usage.completion_tokens if completion.usage else None
                ),
                "finish_reason": finish_reason,
            },
        )

    def generate(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Generate a response from the OpenAI API.

        Args:
            prompt: User prompt (already added to history by BridgeEngine).
            history: Full conversation history including the latest user message.
            config: Optional LLM configuration.

        Returns:
            BridgeResponse with the generated text.

        Raises:
            OpenAIAuthError: If the API key is invalid.
            OpenAIRateLimitError: If the rate limit is exceeded.
            OpenAIConnectionError: If the API cannot be reached.
        """
        messages = history.get_messages_for_api()
        kwargs = self._build_kwargs(config)

        logger.debug(f"Sending request to OpenAI: {self.model}")

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                stream=False,
                **kwargs,
            )
            response = self._to_bridge_response(completion)
            logger.debug(f"OpenAI response received: {response.tokens_used} tokens")
            return response

        except openai.AuthenticationError as exc:
            raise OpenAIAuthError("Invalid OpenAI API key.") from exc
        except openai.RateLimitError as exc:
            raise OpenAIRateLimitError("OpenAI API rate limit exceeded.") from exc
        except openai.APIConnectionError as exc:
            raise OpenAIConnectionError("Cannot connect to the OpenAI API.") from exc
        except openai.APIStatusError as exc:
            raise OpenAIConnectionError(
                f"OpenAI API error {exc.status_code}: {exc.message}"
            ) from exc

    def generate_stream(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> Generator[BridgeResponse, None, None]:
        """
        Generate a streaming response from the OpenAI API.

        Yields text chunks as they arrive. The last chunk has finish_reason set
        and may include token usage if the API returns it.

        Args:
            prompt: User prompt.
            history: Full conversation history.
            config: Optional LLM configuration.

        Yields:
            BridgeResponse objects with partial text.

        Raises:
            OpenAIAuthError: If the API key is invalid.
            OpenAIRateLimitError: If the rate limit is exceeded.
            OpenAIConnectionError: If the API cannot be reached.
        """
        messages = history.get_messages_for_api()
        kwargs = self._build_kwargs(config)

        logger.debug(f"Sending streaming request to OpenAI: {self.model}")

        try:
            with self.client.chat.completions.stream(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield BridgeResponse(text=delta.content, finish_reason="incomplete")

                # Final chunk with usage
                final = stream.get_final_completion()
                tokens_used = None
                if final.usage:
                    tokens_used = (
                        final.usage.prompt_tokens + final.usage.completion_tokens
                    )
                yield BridgeResponse(
                    text="",
                    finish_reason=final.choices[0].finish_reason or "stop",
                    tokens_used=tokens_used,
                    metadata={
                        "model": final.model,
                        "prompt_tokens": final.usage.prompt_tokens if final.usage else None,
                        "completion_tokens": (
                            final.usage.completion_tokens if final.usage else None
                        ),
                    },
                )

        except openai.AuthenticationError as exc:
            raise OpenAIAuthError("Invalid OpenAI API key.") from exc
        except openai.RateLimitError as exc:
            raise OpenAIRateLimitError("OpenAI API rate limit exceeded.") from exc
        except openai.APIConnectionError as exc:
            raise OpenAIConnectionError("Cannot connect to the OpenAI API.") from exc
        except openai.APIStatusError as exc:
            raise OpenAIConnectionError(
                f"OpenAI API error {exc.status_code}: {exc.message}"
            ) from exc

    async def generate_async(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Async version of generate using AsyncOpenAI client.

        Args:
            prompt: User prompt.
            history: Full conversation history.
            config: Optional LLM configuration.

        Returns:
            BridgeResponse with the generated text.
        """
        messages = history.get_messages_for_api()
        kwargs = self._build_kwargs(config)

        logger.debug(f"Sending async request to OpenAI: {self.model}")

        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                stream=False,
                **kwargs,
            )
            return self._to_bridge_response(completion)

        except openai.AuthenticationError as exc:
            raise OpenAIAuthError("Invalid OpenAI API key.") from exc
        except openai.RateLimitError as exc:
            raise OpenAIRateLimitError("OpenAI API rate limit exceeded.") from exc
        except openai.APIConnectionError as exc:
            raise OpenAIConnectionError("Cannot connect to the OpenAI API.") from exc
        except openai.APIStatusError as exc:
            raise OpenAIConnectionError(
                f"OpenAI API error {exc.status_code}: {exc.message}"
            ) from exc

    async def generate_stream_async(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> AsyncGenerator[BridgeResponse, None]:
        """
        Async streaming version using AsyncOpenAI client.

        Args:
            prompt: User prompt.
            history: Full conversation history.
            config: Optional LLM configuration.

        Yields:
            BridgeResponse objects with partial text.
        """
        messages = history.get_messages_for_api()
        kwargs = self._build_kwargs(config)

        logger.debug(f"Sending async streaming request to OpenAI: {self.model}")

        try:
            async with self.async_client.chat.completions.stream(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield BridgeResponse(text=delta.content, finish_reason="incomplete")

                final = await stream.get_final_completion()
                tokens_used = None
                if final.usage:
                    tokens_used = (
                        final.usage.prompt_tokens + final.usage.completion_tokens
                    )
                yield BridgeResponse(
                    text="",
                    finish_reason=final.choices[0].finish_reason or "stop",
                    tokens_used=tokens_used,
                    metadata={
                        "model": final.model,
                        "prompt_tokens": final.usage.prompt_tokens if final.usage else None,
                        "completion_tokens": (
                            final.usage.completion_tokens if final.usage else None
                        ),
                    },
                )

        except openai.AuthenticationError as exc:
            raise OpenAIAuthError("Invalid OpenAI API key.") from exc
        except openai.RateLimitError as exc:
            raise OpenAIRateLimitError("OpenAI API rate limit exceeded.") from exc
        except openai.APIConnectionError as exc:
            raise OpenAIConnectionError("Cannot connect to the OpenAI API.") from exc
        except openai.APIStatusError as exc:
            raise OpenAIConnectionError(
                f"OpenAI API error {exc.status_code}: {exc.message}"
            ) from exc

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model})"


__all__ = ["OpenAIProvider"]
