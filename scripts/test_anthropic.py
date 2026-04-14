"""
Quick smoke test for AnthropicProvider.

Usage:
    pdm run python scripts/test_anthropic.py

Requires ANTHROPIC_API_KEY set in .env or environment.
Get your key at: https://console.anthropic.com/settings/keys
"""

from core_llm_bridge import BridgeEngine
from core_llm_bridge.config import logger, settings
from core_llm_bridge.providers import AnthropicProvider


def main() -> None:
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set.")
        print("Add it to your .env file:")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        return

    logger.info(f"Model: {settings.anthropic_default_model}")

    provider = AnthropicProvider()
    engine = BridgeEngine(
        provider=provider,
        system_prompt="You are a concise assistant. Answer in one sentence.",
    )

    print("\n--- Sync chat ---")
    response = engine.chat("What is the capital of France?")
    print(f"Response: {response.text}")
    print(f"Finish reason: {response.finish_reason}")
    if response.tokens_used:
        print(f"Tokens used: {response.tokens_used}")

    print("\n--- Streaming chat ---")
    print("Response: ", end="", flush=True)
    for chunk in engine.chat_stream("Count from 1 to 5."):
        print(chunk.text, end="", flush=True)
    print()

    print("\n--- Multi-turn conversation ---")
    engine2 = BridgeEngine(provider=AnthropicProvider())
    r1 = engine2.chat("My name is Fernando.")
    print(f"Turn 1: {r1.text}")
    r2 = engine2.chat("What is my name?")
    print(f"Turn 2: {r2.text}")

    print("\nAll Anthropic tests passed.")


if __name__ == "__main__":
    main()
