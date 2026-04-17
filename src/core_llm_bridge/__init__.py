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

__version__ = "1.3.0"
__author__ = "Fernando Velasco"
__license__ = "MIT"

# Configuration — Settings class and logger (no singleton instantiated here)
from .config import Settings, configure_logger, logger
from .core import (
    BaseLLMProvider,
    BridgeEngine,
    BridgeResponse,
    ConversationBuffer,
    LLMConfig,
    Message,
    MessageRole,
    ToolCall,
)

# Exceptions
from .exceptions import (
    ConfigurationError,
    LLMBridgeError,
    OllamaError,
    ProviderError,
    TokenLimitError,
)
from .providers import AnthropicProvider, OllamaProvider, OpenAIProvider, create_provider

# Cost tracking
from .cost_tracker import CostEntry, CostTracker, ModelPricing, cost_tracker

# Utilities
from .utils import (
    PromptManager,
    PromptTemplate,
    TokenCounter,
    create_prompt_manager,
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
    "OllamaProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "create_provider",
    # Config
    "Settings",
    "logger",
    "configure_logger",
    # Cost tracking
    "cost_tracker",
    "CostTracker",
    "CostEntry",
    "ModelPricing",
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
