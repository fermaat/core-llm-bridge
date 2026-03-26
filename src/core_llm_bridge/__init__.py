"""
llm-bridge: Unified interface for local and cloud-based LLMs.

This library provides a standardized way to interact with various LLM providers
(currently Ollama, with OpenAI and Anthropic support planned), handling conversation
memory, streaming, and tool integration seamlessly.

Quick Start:
    >>> from core_llm_bridge import BridgeEngine
    >>> from core_llm_bridge.providers import OllamaProvider
    >>>
    >>> provider = OllamaProvider(model="llama2")
    >>> bridge = BridgeEngine(provider=provider)
    >>> response = bridge.chat("Hello!")
    >>> print(response.text)

Main Classes:
    - BridgeEngine: Main orchestration engine
    - Message, ConversationBuffer: Conversation management
    - BaseLLMProvider: Abstract base for providers
    - PromptManager, TokenCounter: Utilities

Configuration:
    - Settings are loaded from .env files
    - Use logger from config module
    - See llm_bridge.config for customization

Example .env:
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_DEFAULT_MODEL=llama2
    LOG_LEVEL=INFO
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Core imports
from .core import (
    BridgeEngine,
    BridgeResponse,
    ConversationBuffer,
    LLMConfig,
    Message,
    MessageRole,
    BaseLLMProvider,
    ToolCall,
)

# Configuration
from .config import settings, logger, configure_logger

# Utilities
from .utils import (
    TokenCounter,
    PromptManager,
    PromptTemplate,
    create_prompt_manager,
)

# Exceptions
from .exceptions import (
    LLMBridgeError,
    ProviderError,
    OllamaError,
    ConfigurationError,
    TokenLimitError,
)

__all__ = [
    # Core
    "BridgeEngine",
    "Message",
    "MessageRole",
    "ConversationBuffer",
    "BridgeResponse",
    "ToolCall",
    "LLMConfig",
    "BaseLLMProvider",
    # Config
    "settings",
    "logger",
    "configure_logger",
    # Utils
    "TokenCounter",
    "PromptTemplate",
    "PromptManager",
    "create_prompt_manager",
    # Exceptions
    "LLMBridgeError",
    "ProviderError",
    "OllamaError",
    "ConfigurationError",
    "TokenLimitError",
]

