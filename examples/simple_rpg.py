"""Simple RPG example using core-llm-bridge.

This example runs a minimal text-based RPG loop using the BridgeEngine and
OllamaProvider. It is intentionally simple and uses plain console input/output.

Usage:
    python examples/simple_rpg.py
"""

import sys
from pathlib import Path

# Add src to path to import core_llm_bridge
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core_llm_bridge import BridgeEngine
from core_llm_bridge.config import logger
from core_llm_bridge.exceptions import LLMProviderError
from core_llm_bridge.providers import OllamaProvider


def main() -> None:
    logger.info("Starting simple RPG example...")

    provider = OllamaProvider()
    try:
        if not provider.health_check():
            logger.error("Ollama is not available. Make sure ollama serve is running.")
            return
    except LLMProviderError as exc:
        logger.error(f"Provider health check failed: {exc}")
        return

    bridge = BridgeEngine(
        provider=provider,
        system_prompt=(
            "You are a fantasy role-playing game master. "
            "Describe scenes vividly and respond to player actions."  # noqa: E501
        ),
    )

    logger.info("RPG session ready. Type 'exit' to quit.")
    while True:
        player_input = input("What do you do? ").strip()
        if player_input.lower() in {"exit", "quit"}:
            logger.info("Ending RPG session.")
            break

        if not player_input:
            continue

        try:
            response = bridge.chat(player_input)
            print(f"\n{response.text}\n")
        except LLMProviderError as exc:
            logger.error(f"Provider error during chat: {exc}")
            break
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}")
            break


if __name__ == "__main__":
    main()
