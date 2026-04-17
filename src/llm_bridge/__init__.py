"""Alias package for llm-bridge.

This module re-exports the core llm-bridge API from core_llm_bridge,
allowing consumers to use the shorter import path:

    from llm_bridge import BridgeEngine, OllamaProvider

"""

from core_llm_bridge import (
    BaseLLMProvider,
    BridgeEngine,
    BridgeResponse,
    ConversationBuffer,
    LLMConfig,
    Message,
    MessageRole,
    OllamaProvider,
    Settings,
    ToolCall,
    configure_logger,
    create_provider,
    logger,
)

__all__ = [
    "BridgeEngine",
    "BaseLLMProvider",
    "BridgeResponse",
    "ConversationBuffer",
    "LLMConfig",
    "Message",
    "MessageRole",
    "ToolCall",
    "create_provider",
    "OllamaProvider",
    "Settings",
    "configure_logger",
    "logger",
]
