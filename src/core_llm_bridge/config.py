"""
Configuration management for llm-bridge.

This module handles environment variables and application settings using pydantic-settings.
All settings are loaded from .env files or environment variables.

Usage:
    from core_llm_bridge.config import settings, logger

    logger.info(f"Using Ollama at: {settings.ollama_base_url}")
    print(f"Log level: {settings.log_level}")
"""

import sys
from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings

# Project root: src/llm_bridge/config.py -> 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Automatically loads from .env files in this order:
    1. .env (or .env.local for local overrides)

    Configuration:
        case_sensitive: False - Environment variables are case-insensitive
        extra: allow - Ignore extra environment variables
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

    # ========== Environment & Logging ==========
    environment: str = "development"  # development, staging, production
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_storage_folder: str = "logs"
    log_console_output: bool = True

    # ========== Ollama Provider Configuration ==========
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300  # seconds
    ollama_default_model: str = "gemma3:4b"  # "llama2"

    # ========== Anthropic Provider Configuration ==========
    anthropic_api_key: str = ""
    anthropic_default_model: str = "claude-sonnet-4-6"
    anthropic_timeout: int = 300  # seconds

    # ========== OpenAI Provider Configuration ==========
    openai_api_key: str = ""
    openai_default_model: str = "gpt-4o"
    openai_base_url: str = ""  # leave empty for default OpenAI endpoint
    openai_timeout: int = 300  # seconds

    # ========== LLM Settings ==========
    max_context_tokens: int = 4096
    token_safety_margin: int = 100  # Reserve tokens for output

    # ========== Request Configuration ==========
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return PROJECT_ROOT

    @property
    def logs_dir(self) -> Path:
        """Get the logs directory (creates it if needed)."""
        logs_path = PROJECT_ROOT / self.log_storage_folder
        logs_path.mkdir(parents=True, exist_ok=True)
        return logs_path


def configure_logger(
    level: str | None = None,
    enable_console: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configure the application logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to settings.log_level
        enable_console: Whether to enable console output
        log_file: Path to log file. Defaults to logs/llm-bridge.log

    Example:
        >>> from core_llm_bridge.config import configure_logger
        >>> configure_logger(level="DEBUG", enable_console=True)
    """
    # Use settings default if not provided
    if level is None:
        level = settings.log_level
    if log_file is None:
        log_file = str(settings.logs_dir / "llm-bridge.log")

    # Remove default logger
    logger.remove()

    # Add file logger
    logger.add(
        log_file,
        level=level,
        format=(
            "<level>{time:YYYY-MM-DD HH:mm:ss}</level> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
    )

    # Add console logger if enabled
    if enable_console:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "{message}"
            ),
        )


# ========== Initialize at module import ==========
settings = Settings()
configure_logger()

__all__ = [
    "settings",
    "logger",
    "configure_logger",
    "PROJECT_ROOT",
]
