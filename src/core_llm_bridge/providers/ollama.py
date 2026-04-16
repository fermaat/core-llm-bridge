"""
Ollama LLM provider implementation.

Provides integration with Ollama for local LLM inference.

Requires:
    - Ollama running locally (ollama serve)
    - Model downloaded (ollama pull llama2)
    - OLLAMA_BASE_URL configured in .env
"""

from collections.abc import AsyncGenerator, Generator
from contextlib import suppress
from typing import Any

import httpx

from core_utils.logger import logger
from core_llm_bridge.exceptions import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)

from ..core.base import BaseLLMProvider
from ..core.models import BridgeResponse, ConversationBuffer, LLMConfig


class OllamaProvider(BaseLLMProvider):
    """
    LLM provider for Ollama.

    Ollama is a local LLM inference engine that allows running models like
    Llama 2, Mistral, and others on your machine without relying on cloud APIs.

    Prerequisites:
        - Ollama installed: https://ollama.ai
        - Ollama running: ollama serve
        - Model downloaded: ollama pull llama2
        - OLLAMA_BASE_URL configured in .env (default: http://localhost:11434)

    Example:
        >>> provider = OllamaProvider(model="llama2")
        >>> if not provider.validate_connection():
        ...     print("Ollama is not running!")
        >>> response = provider.generate("Hello", ConversationBuffer())
        >>> print(response.text)

    Attributes:
        model: Name of the model to use (e.g., "llama2", "mistral")
        base_url: URL where Ollama is running
        timeout: Request timeout in seconds
        client: HTTP client for requests
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "llama2"). Required.
            base_url: Ollama service URL.
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(model=model, **kwargs)

        self.base_url = base_url
        self.timeout = timeout

        # Initialize HTTP client
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.debug(f"OllamaProvider initialized: {self.model} at {self.base_url}")

    def validate_connection(self, raise_on_error: bool = False) -> bool:
        """
        Validate that Ollama is reachable and has the model.

        Args:
            raise_on_error: If True, raise a provider-specific exception on failure.

        Returns:
            True if connection is valid and model exists, False otherwise

        Example:
            >>> provider = OllamaProvider(model="llama2")
            >>> if provider.validate_connection():
            ...     print("Ready!")
            >>> else:
            ...     print("Ollama not running or model not found")
        """
        try:
            response = self.client.get("/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            model_found = any(self.model in name for name in model_names)

            if not model_found:
                message = (
                    f"Model '{self.model}' not found in Ollama. " f"Available models: {model_names}"
                )
                if raise_on_error:
                    raise OllamaModelNotFoundError(message)
                logger.error(message)
                return False

            logger.debug(f"Connected to Ollama. Model '{self.model}' found.")
            return True

        except httpx.TimeoutException as exc:
            message = f"Timeout connecting to Ollama at {self.base_url}"
            if raise_on_error:
                raise OllamaTimeoutError(message) from exc
            logger.error(message)
            return False
        except httpx.RequestError as exc:
            message = (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
            if raise_on_error:
                raise OllamaConnectionError(message) from exc
            logger.error(message)
            return False
        except Exception as exc:
            message = f"Error validating Ollama connection: {exc}"
            if raise_on_error:
                raise OllamaConnectionError(message) from exc
            logger.error(message)
            return False

    def get_model_info(self) -> dict[str, Any] | None:
        """
        Get information about the model.

        Returns:
            Dictionary with model information or None if error
        """
        try:
            response = self.client.get("/api/show", params={"name": self.model})
            if response.status_code == 200:
                model_info = response.json()
                if isinstance(model_info, dict):
                    return model_info
                logger.warning("Model info response is not a dict")
            return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None

    def generate(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Generate a response from Ollama.

        Args:
            prompt: User prompt
            history: Conversation history
            config: Optional LLM configuration

        Returns:
            BridgeResponse with the generated text

        Raises:
            OllamaConnectionError: If cannot connect to Ollama
            OllamaModelNotFoundError: If model not found
            OllamaTimeoutError: If request times out
        """
        self.validate_connection(raise_on_error=True)

        # Get messages in API format
        messages = history.get_messages_for_api()

        # Prepare request
        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        # Add LLM config if provided
        if config:
            if config.temperature is not None:
                request_data["temperature"] = config.temperature
            if config.top_p is not None:
                request_data["top_p"] = config.top_p

        logger.debug(f"Sending request to Ollama: {self.model}")

        try:
            response = self.client.post("/api/chat", json=request_data)

            if response.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{self.model}' not found. " f"Download it: ollama pull {self.model}"
                )

            if response.status_code != 200:
                raise OllamaConnectionError(
                    f"Ollama error: {response.status_code} - {response.text}"
                )

            data = response.json()

            # Extract response
            generated_text = data.get("message", {}).get("content", "")

            # Extract token count if available
            tokens_used = None
            if "eval_count" in data:
                tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

            bridge_response = BridgeResponse(
                text=generated_text,
                finish_reason=data.get("done", True) and "stop" or "incomplete",
                tokens_used=tokens_used,
                raw_response=data,
                metadata={
                    "model": self.model,
                    "eval_count": data.get("eval_count"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_duration": data.get("eval_duration"),
                },
            )

            logger.debug(f"Response received from Ollama: {len(generated_text)} chars")
            return bridge_response

        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Request to Ollama timed out (>{self.timeout}s). "
                "Model might be too large or system too slow."
            ) from e
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure it's running: ollama serve"
            ) from e

    def generate_stream(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> Generator[BridgeResponse, None, None]:
        """
        Generate a response from Ollama with streaming.

        Yields tokens as they become available.

        Args:
            prompt: User prompt
            history: Conversation history
            config: Optional LLM configuration

        Yields:
            BridgeResponse objects with partial text

        Raises:
            OllamaConnectionError: If cannot connect to Ollama
            OllamaModelNotFoundError: If model not found
        """
        self.validate_connection(raise_on_error=True)

        # Get messages in API format
        messages = history.get_messages_for_api()

        # Prepare request
        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        # Add LLM config if provided
        if config:
            if config.temperature is not None:
                request_data["temperature"] = config.temperature
            if config.top_p is not None:
                request_data["top_p"] = config.top_p

        logger.debug(f"Sending streaming request to Ollama: {self.model}")

        try:
            with self.client.stream("POST", "/api/chat", json=request_data) as response:
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{self.model}' not found. " f"Download it: ollama pull {self.model}"
                    )

                if response.status_code != 200:
                    raise OllamaConnectionError(f"Ollama error: {response.status_code}")

                # Stream responses as they come
                full_response = ""
                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = response.json() if hasattr(response, "json") else {}
                        # Parse line as JSON (Ollama sends newline-delimited JSON)
                        import json

                        data = json.loads(line)
                        chunk_text = data.get("message", {}).get("content", "")

                        if chunk_text:
                            full_response += chunk_text
                            yield BridgeResponse(
                                text=chunk_text,
                                finish_reason="incomplete",
                            )

                        # Check if this is the last chunk
                        if data.get("done", False):
                            tokens_used = data.get("eval_count", 0) + data.get(
                                "prompt_eval_count", 0
                            )
                            yield BridgeResponse(
                                text="",
                                finish_reason="stop",
                                tokens_used=tokens_used,
                                metadata={
                                    "model": self.model,
                                    "eval_count": data.get("eval_count"),
                                    "prompt_eval_count": data.get("prompt_eval_count"),
                                },
                            )

                    except Exception as e:
                        logger.debug(f"Error parsing streaming response: {e}")
                        continue

        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Streaming request to Ollama timed out (>{self.timeout}s)"
            ) from e
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure it's running: ollama serve"
            ) from e

    async def generate_async(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Async version of generate (uses sync implementation for now).

        In production, could use httpx.AsyncClient for true async.
        """
        return self.generate(prompt, history, config)

    async def generate_stream_async(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> AsyncGenerator[BridgeResponse, None]:
        """
        Async version of generate_stream (uses sync implementation for now).

        In production, could use httpx.AsyncClient for true async streaming.
        """
        for response in self.generate_stream(prompt, history, config):
            yield response

    def __del__(self) -> None:
        """Clean up HTTP client."""
        with suppress(Exception):
            self.client.close()


__all__ = ["OllamaProvider"]
