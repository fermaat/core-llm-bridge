"""
Configuration schema for core-llm-bridge.

Settings is provided as a class for consumers that want a structured way to
declare provider configuration. It is NOT instantiated here — the consuming
application is responsible for instantiation and for configuring the logger.

Usage in a consumer application:
    from core_llm_bridge.config import Settings
    from core_utils.logger import configure_logger

    class AppSettings(Settings):
        model_config = {"env_file": [".env"], ...}

    settings = AppSettings()
    configure_logger(settings)
"""

from core_utils.logger import configure_logger, logger
from core_utils.settings import CoreSettings


class Settings(CoreSettings):
    """
    Configuration schema for core-llm-bridge providers.

    Defines the fields each provider needs. Does not load any files.
    Subclass and add env_file to model_config in your application.
    """

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300
    ollama_default_model: str = ""

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_default_model: str = "claude-sonnet-4-6"
    anthropic_timeout: int = 300

    # OpenAI
    openai_api_key: str = ""
    openai_default_model: str = "gpt-4o"
    openai_base_url: str = ""
    openai_timeout: int = 300

    max_context_tokens: int = 4096
    token_safety_margin: int = 100
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


__all__ = ["Settings", "configure_logger", "logger"]
