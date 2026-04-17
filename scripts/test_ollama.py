"""
Quick smoke test for OllamaProvider.

Usage:
    pdm run python scripts/test_ollama.py

Requires Ollama running locally (default: http://localhost:11434).
The model set in OLLAMA_DEFAULT_MODEL must be pulled.
"""

from _settings import logger, settings

from core_llm_bridge import BridgeEngine
from core_llm_bridge.providers import OllamaProvider


def main() -> None:
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Model: {settings.ollama_default_model}")

    provider = OllamaProvider(
        model=settings.ollama_default_model,
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout,
    )

    print("\n--- Health check ---")
    healthy = provider.health_check()
    print(f"Ollama healthy: {healthy}")
    if not healthy:
        print("Ollama is not reachable. Is it running?")
        return

    engine = BridgeEngine(
        provider=provider,
        system_prompt="You are a concise assistant. Answer in one sentence.",
    )

    print("\n--- Sync chat ---")
    response = engine.chat("What is the capital of France?")
    print(f"Response: {response.text}")
    print(f"Finish reason: {response.finish_reason}")

    print("\n--- Streaming chat ---")
    print("Response: ", end="", flush=True)
    for chunk in engine.chat_stream("Count from 1 to 5."):
        print(chunk.text, end="", flush=True)
    print()

    print("\n--- Multi-turn conversation ---")
    engine2 = BridgeEngine(provider=OllamaProvider(
        model=settings.ollama_default_model,
        base_url=settings.ollama_base_url,
    ))
    r1 = engine2.chat("My name is Fernando.")
    print(f"Turn 1: {r1.text}")
    r2 = engine2.chat("What is my name?")
    print(f"Turn 2: {r2.text}")

    print("\nAll Ollama tests passed.")


if __name__ == "__main__":
    main()
