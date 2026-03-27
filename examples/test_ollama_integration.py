#!/usr/bin/env python
"""
Test OllamaProvider with a real Ollama instance.

Requirements:
- Ollama running: ollama serve
- Model downloaded: ollama pull gemma3:4b (or your preferred model)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core_llm_bridge.core import BridgeEngine, LLMConfig
from core_llm_bridge.providers import OllamaProvider
from core_llm_bridge.utils import create_prompt_manager


def test_ollama_connection():
    """Test basic Ollama connection."""
    print("=" * 60)
    print("Test 1: Ollama Connection")
    print("=" * 60)

    try:
        provider = OllamaProvider(model="gemma3:4b")
        is_connected = provider.validate_connection()

        if is_connected:
            print("✅ Connection successful!")
            model_info = provider.get_model_info()
            if model_info:
                print(f"   Model: {model_info.get('name', 'unknown')}")
        else:
            print("❌ Connection failed")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


def test_simple_generation():
    """Test simple text generation."""
    print("\n" + "=" * 60)
    print("Test 2: Simple Generation")
    print("=" * 60)

    try:
        provider = OllamaProvider(model="gemma3:4b")

        from core_llm_bridge.core import ConversationBuffer, Message, MessageRole

        history = ConversationBuffer()
        history.add_user_message("Say 'Hello from Ollama!' in one sentence.")

        print("\n🤖 Generating response...")
        response = provider.generate(prompt="test", history=history)

        print(f"\n✅ Response received:")
        print(f"   Text: {response.text[:200]}")
        print(f"   Tokens used: {response.tokens_used}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


def test_streaming():
    """Test streaming generation."""
    print("\n" + "=" * 60)
    print("Test 3: Streaming Generation")
    print("=" * 60)

    try:
        provider = OllamaProvider(model="gemma3:4b")

        from core_llm_bridge.core import ConversationBuffer

        history = ConversationBuffer()
        history.add_user_message("Write a short 3-line poem about AI.")

        print("\n🤖 Streaming response:")
        print("-" * 40)

        for response in provider.generate_stream(prompt="test", history=history):
            print(response.text, end="", flush=True)

        print("\n" + "-" * 40)
        print("✅ Streaming completed")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


def test_with_bridge_engine():
    """Test using OllamaProvider with BridgeEngine."""
    print("\n" + "=" * 60)
    print("Test 4: BridgeEngine with OllamaProvider")
    print("=" * 60)

    try:
        provider = OllamaProvider(model="gemma3:4b")
        config = LLMConfig(temperature=0.7, max_tokens=500)

        engine = BridgeEngine(
            provider=provider,
            system_prompt="You are a helpful AI assistant. Be concise and clear.",
        )

        print("\n💬 Chat interaction:")
        print("-" * 40)

        # First turn
        response1 = engine.chat("What is Python?", config=config)
        print(f"User: What is Python?")
        print(f"AI: {response1.text[:150]}...\n")

        # Second turn (with history)
        response2 = engine.chat("Why is it popular?", config=config)
        print(f"User: Why is it popular?")
        print(f"AI: {response2.text[:150]}...\n")

        print("-" * 40)
        print("✅ Chat completed with history management")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


def test_with_prompts():
    """Test using prompts from YAML."""
    print("\n" + "=" * 60)
    print("Test 5: With YAML Prompts")
    print("=" * 60)

    try:
        prompt_manager = create_prompt_manager()
        provider = OllamaProvider(model="gemma3:4b")
        config = LLMConfig(temperature=0.5, max_tokens=300)

        # Get a prompt
        system_prompt = prompt_manager.render(
            "code_assistant",
            query="General assistance"
        )

        engine = BridgeEngine(
            provider=provider,
            system_prompt=system_prompt,
        )

        print(f"\n📝 Using 'code_assistant' prompt")
        print("-" * 40)

        response = engine.chat("Write a simple function to add two numbers", config=config)
        print(f"User: Write a simple function to add two numbers")
        print(f"\nAI:\n{response.text[:400]}")
        print("\n" + "-" * 40)
        print("✅ Prompt-based generation completed")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  Testing core-llm-bridge with Ollama               ║")
    print("╚" + "=" * 58 + "╝")

    tests = [
        ("Connection", test_ollama_connection),
        ("Generation", test_simple_generation),
        ("Streaming", test_streaming),
        ("BridgeEngine", test_with_bridge_engine),
        ("Prompts", test_with_prompts),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nPassed {passed}/{total} tests")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
