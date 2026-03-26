"""Core abstractions and main engine for llm-bridge."""

from .base import BaseLLMProvider, ToolProvider
from .engine import BridgeEngine
from .models import (
    BridgeResponse,
    ConversationBuffer,
    LLMConfig,
    Message,
    MessageRole,
    ToolCall,
)

__all__ = [
    # Models
    "Message",
    "MessageRole",
    "ConversationBuffer",
    "BridgeResponse",
    "ToolCall",
    "LLMConfig",
    # Abstractions
    "BaseLLMProvider",
    "ToolProvider",
    # Engine
    "BridgeEngine",
]
