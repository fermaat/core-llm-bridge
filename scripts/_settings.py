"""Shared settings and logger setup for scripts in this directory."""

from pathlib import Path

from core_utils.logger import configure_logger
from loguru import logger

from core_llm_bridge.config import Settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppSettings(Settings):
    model_config = {
        "case_sensitive": False,
        "extra": "allow",
        "env_file": [str(_PROJECT_ROOT / ".env"), str(_PROJECT_ROOT / ".env.local")],
        "env_file_encoding": "utf-8",
    }


settings = AppSettings()
configure_logger(settings, log_file=str(settings.logs_dir / "llm-bridge.log"))

__all__ = ["settings", "logger"]
