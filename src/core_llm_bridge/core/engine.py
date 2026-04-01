"""
Main orchestration engine for llm-bridge.

The BridgeEngine is the primary class users interact with.
It manages providers, conversation history, and coordinates all interactions.
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any

from core_llm_bridge.config import logger

from .base import BaseLLMProvider
from .models import BridgeResponse, ConversationBuffer, LLMConfig, Message, ToolCall


class BridgeEngine:
    """
    Main orchestration engine for llm-bridge.

    Handles:
    - Conversation history management
    - Provider abstraction and switching
    - Tool/function registration and calling
    - Token management
    - Request retry logic

    Example:
        >>> from core_llm_bridge import BridgeEngine
        >>> from core_llm_bridge.providers import OllamaProvider
        >>>
        >>> provider = OllamaProvider(model="llama2")
        >>> engine = BridgeEngine(provider=provider)
        >>> response = engine.chat("Hello!")
        >>> print(response.text)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        system_prompt: str | None = None,
        max_history_length: int = 10,
        history_prune_step: int = 2,
    ) -> None:
        """
        Initialize the BridgeEngine.

        Args:
            provider: The LLM provider to use
            system_prompt: Optional system prompt to use for all requests
            max_history_length: Maximum number of messages to keep in history

        Raises:
            ValueError: If provider is not a BaseLLMProvider instance
        """
        if not isinstance(provider, BaseLLMProvider):
            raise ValueError("provider must be an instance of BaseLLMProvider")

        self.provider = provider
        self.system_prompt = system_prompt
        self.max_history_length = max_history_length

        # Initialize conversation buffer
        self.history = ConversationBuffer(system_prompt=system_prompt)
        self.history_prune_step = history_prune_step
        self.internal_state: str | None = None

        # Tool registry
        self._tools: dict[str, Any] = {}

        logger.info(f"BridgeEngine initialized with provider: {provider}")

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt.

        Args:
            prompt: The new system prompt
        """
        self.system_prompt = prompt
        self.history.system_prompt = prompt
        logger.debug(f"System prompt updated: {prompt[:50]}...")

    def register_tool(self, func: Any) -> None:
        """
        Register a tool/function that the LLM can call.

        Args:
            func: Callable function to register

        Example:
            >>> def add(a: int, b: int) -> int:
            ...     return a + b
            >>> engine.register_tool(add)
        """
        if not callable(func):
            raise ValueError(f"{func} is not callable")

        name = func.__name__
        self._tools[name] = func
        logger.debug(f"Tool registered: {name}")

    def get_tools(self) -> dict[str, Any]:
        """
        Get all registered tools.

        Returns:
            Dictionary mapping tool names to callables
        """
        return self._tools.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history.clear()
        self.internal_state = None
        logger.debug("Conversation history cleared")

    def _update_internal_state(self, removed_messages: list[Message]) -> None:
        """Update an internal state summary from removed messages."""
        summary = " | ".join(
            f"{message.role.value}: {message.content}" for message in removed_messages
        )
        if self.internal_state:
            self.internal_state += f"\n{summary}"
        else:
            self.internal_state = summary

    def prune_history(self) -> None:
        """Prune old messages from history and update internal state."""
        if len(self.history) < self.max_history_length:
            return

        removed_messages: list[Message] = []
        remove_count = max(
            self.history_prune_step,
            len(self.history) - self.max_history_length + 2,
        )

        for _ in range(remove_count):
            if self.history.messages:
                removed_messages.append(self.history.messages.pop(0))

        if removed_messages:
            self._update_internal_state(removed_messages)
            logger.debug(
                f"History pruned by removing {len(removed_messages)} old messages"
            )

    def chat(
        self,
        user_input: str,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Send a message and get a response.

        Automatically manages conversation history.

        Args:
            user_input: The user's message
            config: Optional LLM configuration

        Returns:
            BridgeResponse with the assistant's reply

        Example:
            >>> response = engine.chat("Tell me a joke")
            >>> print(response.text)
        """
        logger.debug(f"Chat: {user_input[:50]}...")

        # Check context window
        if len(self.history) >= self.max_history_length:
            self.prune_history()

        # Add user message to history
        self.history.add_user_message(user_input)

        # Get response from provider
        response = self.provider.generate(
            prompt=user_input,
            history=self.history,
            config=config,
        )

        # Add assistant response to history
        self.history.add_assistant_message(response.text)

        # Handle tool calls if any
        if response.tool_calls:
            self._handle_tool_calls(response.tool_calls)

        logger.debug(f"Response received: {response.finish_reason}")
        return response

    def chat_stream(
        self,
        user_input: str,
        config: LLMConfig | None = None,
    ) -> Generator[BridgeResponse, None, None]:
        """
        Send a message and stream the response.

        Args:
            user_input: The user's message
            config: Optional LLM configuration

        Yields:
            BridgeResponse objects with streaming tokens

        Example:
            >>> for chunk in engine.chat_stream("Tell me a story"):
            ...     print(chunk.text, end="", flush=True)
        """
        logger.debug(f"Chat stream: {user_input[:50]}...")

        # Check context window
        if len(self.history) >= self.max_history_length:
            self.prune_history()

        # Add user message to history
        self.history.add_user_message(user_input)

        # Stream response from provider
        full_text = ""
        for partial_response in self.provider.generate_stream(
            prompt=user_input,
            history=self.history,
            config=config,
        ):
            full_text += partial_response.text
            yield partial_response

        # Add full response to history
        if full_text:
            self.history.add_assistant_message(full_text)

        logger.debug("Stream completed")

    async def chat_async(
        self,
        user_input: str,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """
        Async version of chat.

        Args:
            user_input: The user's message
            config: Optional LLM configuration

        Returns:
            BridgeResponse with the assistant's reply
        """
        logger.debug(f"Async chat: {user_input[:50]}...")

        # Check context window
        if len(self.history) >= self.max_history_length:
            self.prune_history()

        # Add user message to history
        self.history.add_user_message(user_input)

        # Get response from provider
        response = await self.provider.generate_async(
            prompt=user_input,
            history=self.history,
            config=config,
        )

        # Add assistant response to history
        self.history.add_assistant_message(response.text)

        return response

    async def chat_stream_async(
        self,
        user_input: str,
        config: LLMConfig | None = None,
    ) -> AsyncGenerator[BridgeResponse, None]:
        """
        Async version of chat_stream.

        Args:
            user_input: The user's message
            config: Optional LLM configuration

        Yields:
            BridgeResponse objects with streaming tokens
        """
        logger.debug(f"Async chat stream: {user_input[:50]}...")

        # Check context window
        if len(self.history) >= self.max_history_length:
            self.prune_history()

        # Add user message to history
        self.history.add_user_message(user_input)

        # Stream response from provider
        full_text = ""
        async for partial_response in self.provider.generate_stream_async(
            prompt=user_input,
            history=self.history,
            config=config,
        ):
            full_text += partial_response.text
            yield partial_response

        # Add full response to history
        if full_text:
            self.history.add_assistant_message(full_text)

    def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """
        Handle tool calls from the LLM response.

        Args:
            tool_calls: List of tool calls to execute
        """
        for tool_call in tool_calls:
            if tool_call.function_name not in self._tools:
                logger.warning(f"Tool not found: {tool_call.function_name}")
                continue

            try:
                tool_func = self._tools[tool_call.function_name]
                result = tool_func(**tool_call.arguments)
                tool_call.result = str(result)
                logger.debug(f"Tool executed: {tool_call.function_name}")
            except Exception as e:
                logger.error(f"Error executing tool {tool_call.function_name}: {e}")
                tool_call.result = f"Error: {str(e)}"

    def get_conversation_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current conversation.

        Returns:
            Dictionary with conversation stats
        """
        return {
            "total_messages": len(self.history),
            "provider": str(self.provider),
            "has_system_prompt": self.system_prompt is not None,
            "registered_tools": list(self._tools.keys()),
            "internal_state": self.internal_state,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BridgeEngine(provider={self.provider}, messages={len(self.history)})"


__all__ = ["BridgeEngine"]
