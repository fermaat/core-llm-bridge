"""
Simple chat example with core-llm-bridge.

This example demonstrates basic usage of the BridgeEngine with OllamaProvider.

Prerequisites:
    - Ollama installed and running (ollama serve)
    - Model pulled (ollama pull llama2)
    - .env file configured with OLLAMA_BASE_URL and OLLAMA_DEFAULT_MODEL

Usage:
    python examples/simple_chat.py
"""

import sys
from pathlib import Path

# Add src to path to import core_llm_bridge
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core_llm_bridge import BridgeEngine
from core_llm_bridge.config import logger
from core_llm_bridge.providers import OllamaProvider


def main() -> None:
    """Run a simple chat session with the LLM."""
    try:
        logger.info("Starting simple chat example...")

        # Initialize provider
        provider = OllamaProvider(model="llama2")

        # Validate connection
        if not provider.validate_connection():
            logger.error("Cannot connect to Ollama. " "Make sure it's running: ollama serve")
            return

        # Create engine
        bridge = BridgeEngine(provider=provider, system_prompt="You are a helpful assistant.")

        # Chat loop
        logger.info("Chat initialized. Type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "exit":
                logger.info("Goodbye!")
                break

            if not user_input:
                continue

            logger.debug(f"Sending: {user_input}")
            response = bridge.chat(user_input)
            print(f"\nAssistant: {response.text}\n")

    except Exception as e:
        logger.error(f"Error in simple chat: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
