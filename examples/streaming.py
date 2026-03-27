"""
Streaming example with core-llm-bridge.

This example demonstrates streaming responses from the LLM.
Responses are yielded token-by-token as they become available.

Prerequisites:
    - Ollama installed and running
    - Model pulled
    - .env file configured

Usage:
    python examples/streaming.py
"""

import sys
from pathlib import Path

# Add src to path to import core_llm_bridge
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core_llm_bridge import BridgeEngine
from core_llm_bridge.config import logger
from core_llm_bridge.providers import OllamaProvider


def main() -> None:
    """Stream a response from the LLM."""
    try:
        logger.info("Starting streaming example...")

        # Initialize provider
        provider = OllamaProvider(model="llama2")

        # Validate connection
        if not provider.validate_connection():
            logger.error(
                "Cannot connect to Ollama. "
                "Make sure it's running: ollama serve"
            )
            return

        # Create engine
        bridge = BridgeEngine(provider=provider)

        # Streaming request
        prompt = "Write a short poem about programming."
        logger.info(f"Prompt: {prompt}")

        print("Response (streaming):\n")
        for chunk in bridge.chat_stream(prompt):
            # Print each token as it arrives
            print(chunk.text, end="", flush=True)
        print("\n")

    except Exception as e:
        logger.error(f"Error in streaming example: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
