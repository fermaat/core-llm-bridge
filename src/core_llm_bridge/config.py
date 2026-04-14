"""
Configuration management for core-llm-bridge.

Settings are loaded from .env / .env.local at the project root.
Logger is configured via core-utils' configure_logger.

Usage:
    from core_llm_bridge.config import settings, logger
"""

from pathlib import Path

from core_utils.logger import configure_logger, logger
from core_utils.settings import CoreSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(CoreSettings):
    """
    Application settings for core-llm-bridge.

    Inherits environment/logging fields from CoreSettings.
    Adds provider-specific configuration on top.
    """

    model_config = {
        "case_sensitive": False,
        "extra": "allow",
        "env_file": [
            str(PROJECT_ROOT / ".env"),
            str(PROJECT_ROOT / ".env.local"),
        ],
        "env_file_encoding": "utf-8",
    }

    # ========== Ollama Provider ==========
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300
    ollama_default_model: str = "gemma3:4b"

    # ========== Anthropic Provider ==========
    anthropic_api_key: str = ""
    anthropic_default_model: str = "claude-sonnet-4-6"
    anthropic_timeout: int = 300

    # ========== OpenAI Provider ==========
    openai_api_key: str = ""
    openai_default_model: str = "gpt-4o"
    openai_base_url: str = ""
    openai_timeout: int = 300

    # ========== LLM Settings ==========
    max_context_tokens: int = 4096
    token_safety_margin: int = 100

    # ========== Request Configuration ==========
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT


settings = Settings()
configure_logger(settings, log_file=str(settings.logs_dir / "llm-bridge.log"))

__all__ = [
    "settings",
    "logger",
    "configure_logger",
    "PROJECT_ROOT",
]
