"""
Pydantic models for llm-bridge.

Defines all data structures used across the application.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Enum for message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        role: The role of the message sender (user, assistant, system, tool)
        content: The text content of the message
        timestamp: When the message was created
        metadata: Additional metadata about the message
    """

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[dict[str, Any]] = None

    model_config = {"use_enum_values": False}

    def __str__(self) -> str:
        """Return a string representation of the message."""
        return f"[{self.role.value.upper()}] {self.content[:100]}..."

    def to_dict_for_api(self) -> dict[str, str]:
        """
        Convert message to format expected by LLM APIs.

        Returns:
            Dictionary with 'role' and 'content' keys
        """
        return {
            "role": self.role.value,
            "content": self.content,
        }


class ToolCall(BaseModel):
    """
    Represents a tool/function call made by the LLM.

    Attributes:
        id: Unique identifier for the tool call
        function_name: Name of the function to call
        arguments: Dictionary of arguments to pass to the function
        result: Result of the function call (populated after execution)
    """

    id: str
    function_name: str
    arguments: dict[str, Any]
    result: Optional[str] = None


class BridgeResponse(BaseModel):
    """
    Response from the LLM through BridgeEngine.

    Attributes:
        text: The main text response from the LLM
        raw_response: Raw response from the provider (may be used for debugging)
        tool_calls: List of tool calls made by the LLM
        tokens_used: Number of tokens consumed for this response
        finish_reason: Why the generation stopped (stop, tool_use, length, etc)
        metadata: Additional metadata about the response
    """

    text: str
    raw_response: Optional[dict[str, Any]] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    finish_reason: str = "stop"  # stop, tool_use, length, error
    metadata: Optional[dict[str, Any]] = None

    def __str__(self) -> str:
        """Return a string representation of the response."""
        return f"Response: {self.text[:100]}...\n(finish_reason={self.finish_reason})"


class ConversationBuffer(BaseModel):
    """
    Manages a buffer of messages in a conversation.

    Keeps track of conversation history with support for:
    - Token counting
    - Context window limits
    - Automatic message pruning
    - Summary generation (future)

    Attributes:
        messages: List of messages in the conversation
        system_prompt: System prompt to prepend to all requests
        max_tokens: Maximum tokens allowed in the buffer
        current_tokens: Current token count in the buffer
    """

    messages: list[Message] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    max_tokens: int = 4096
    current_tokens: int = 0

    def add_message(self, role: MessageRole, content: str) -> None:
        """
        Add a message to the buffer.

        Args:
            role: The role of the message sender
            content: The content of the message
        """
        message = Message(role=role, content=content)
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the buffer."""
        self.add_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the buffer."""
        self.add_message(MessageRole.ASSISTANT, content)

    def add_system_message(self, content: str) -> None:
        """Add a system message to the buffer."""
        self.add_message(MessageRole.SYSTEM, content)

    def get_messages_for_api(self) -> list[dict[str, str]]:
        """
        Get messages formatted for LLM API calls.

        Returns:
            List of message dictionaries with 'role' and 'content' keys

        Example:
            >>> buffer = ConversationBuffer()
            >>> buffer.add_user_message("Hello!")
            >>> buffer.get_messages_for_api()
            [{'role': 'user', 'content': 'Hello!'}]
        """
        messages = []

        # Add system prompt if present
        if self.system_prompt:
            messages.append(
                {
                    "role": MessageRole.SYSTEM.value,
                    "content": self.system_prompt,
                }
            )

        # Add all messages
        for message in self.messages:
            messages.append(message.to_dict_for_api())

        return messages

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        self.messages.clear()
        self.current_tokens = 0

    def prune_old_messages(self, keep_last_n: int = 10) -> None:
        """
        Remove old messages, keeping only the last N messages.

        Useful when context window is getting full.

        Args:
            keep_last_n: Number of recent messages to keep
        """
        if len(self.messages) > keep_last_n:
            self.messages = self.messages[-keep_last_n:]

    def __len__(self) -> int:
        """Return the number of messages in the buffer."""
        return len(self.messages)

    def __repr__(self) -> str:
        """Return a string representation of the buffer."""
        return f"ConversationBuffer({len(self.messages)} messages, {self.current_tokens} tokens)"


class LLMConfig(BaseModel):
    """
    Configuration for LLM requests.

    Attributes:
        temperature: Controls randomness (0-1, lower = more deterministic)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        stop_sequences: Sequences that stop generation
        system_prompt: System prompt for this request
    """

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    stop_sequences: list[str] = Field(default_factory=list)
    system_prompt: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1024,
                    "stop_sequences": [],
                }
            ]
        }
    }


__all__ = [
    "MessageRole",
    "Message",
    "ToolCall",
    "BridgeResponse",
    "ConversationBuffer",
    "LLMConfig",
]
