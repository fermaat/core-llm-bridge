"""
Tool/function calling example with core-llm-bridge.

This example demonstrates how to use the tool calling feature
to let the LLM invoke external functions.

Note: Ollama's tool calling support varies by model.
This example shows the API even if the model doesn't support it yet.

Prerequisites:
    - Ollama installed and running
    - Model pulled (preferably a model that supports tool calling)
    - .env file configured

Usage:
    python examples/with_tools.py
"""

import sys
from pathlib import Path

# Add src to path to import core_llm_bridge
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core_llm_bridge import BridgeEngine
from core_llm_bridge.config import logger
from core_llm_bridge.providers import OllamaProvider


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def main() -> None:
    """Run a chat session with tool calling."""
    try:
        logger.info("Starting tool calling example...")

        # Initialize provider
        provider = OllamaProvider(model="llama2")

        # Validate connection
        if not provider.validate_connection():
            logger.error("Cannot connect to Ollama. " "Make sure it's running: ollama serve")
            return

        # Create engine
        bridge = BridgeEngine(provider=provider)

        # Register tools (functions the LLM can call)
        bridge.register_tool(add)
        bridge.register_tool(subtract)
        bridge.register_tool(multiply)

        logger.info("Registered tools: add, subtract, multiply")

        # Ask a question that might trigger tool use
        prompt = "What is 15 + 7? And what is 25 - 8? And what is 6 * 4?"
        logger.info(f"Prompt: {prompt}")

        response = bridge.chat(prompt)
        print(f"\nResponse: {response.text}\n")

        # Check if any tools were called
        if response.tool_calls:
            print("Tools called:")
            for call in response.tool_calls:
                print(f"  - {call.function_name}({call.arguments}) = {call.result}")
        else:
            print("(No tool calls in this response)")

    except Exception as e:
        logger.error(f"Error in tool calling example: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
