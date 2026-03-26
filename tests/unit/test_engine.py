"""Unit tests for core engine and abstractions."""

import pytest

from core_llm_bridge import BridgeEngine, ConversationBuffer, LLMConfig
from core_llm_bridge.core import BaseLLMProvider, BridgeResponse
from core_llm_bridge.exceptions import LLMBridgeError


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""

    def generate(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ) -> BridgeResponse:
        """Return a mock response."""
        return BridgeResponse(
            text=f"Mock response to: {prompt}",
            finish_reason="stop",
        )

    def generate_stream(
        self,
        prompt: str,
        history: ConversationBuffer,
        config: LLMConfig | None = None,
    ):
        """Yield mock response in chunks."""
        response_text = f"Mock response to: {prompt}"
        for chunk in response_text.split():
            yield BridgeResponse(text=chunk + " ")


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Test that BaseLLMProvider cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLLMProvider(model="test")  # type: ignore

    def test_mock_provider_creation(self) -> None:
        """Test creating a mock provider."""
        provider = MockProvider(model="mock")
        assert provider.model == "mock"
        assert "mock" in repr(provider).lower()


class TestBridgeEngine:
    """Tests for BridgeEngine class."""

    @pytest.fixture
    def mock_provider(self) -> MockProvider:
        """Create a mock provider for testing."""
        return MockProvider(model="test-model")

    @pytest.fixture
    def engine(self, mock_provider: MockProvider) -> BridgeEngine:
        """Create a BridgeEngine instance for testing."""
        return BridgeEngine(provider=mock_provider)

    def test_create_engine(self, engine: BridgeEngine) -> None:
        """Test creating a BridgeEngine."""
        assert engine is not None
        assert engine.provider is not None

    def test_invalid_provider(self) -> None:
        """Test creating engine with invalid provider."""
        with pytest.raises(ValueError):
            BridgeEngine(provider="not-a-provider")  # type: ignore

    def test_set_system_prompt(self, engine: BridgeEngine) -> None:
        """Test setting system prompt."""
        system_prompt = "You are helpful"
        engine.set_system_prompt(system_prompt)
        assert engine.system_prompt == system_prompt

    def test_register_tool(self, engine: BridgeEngine) -> None:
        """Test registering a tool."""

        def add(a: int, b: int) -> int:
            return a + b

        engine.register_tool(add)
        tools = engine.get_tools()
        assert "add" in tools
        assert tools["add"] is add

    def test_register_non_callable(self, engine: BridgeEngine) -> None:
        """Test registering non-callable raises error."""
        with pytest.raises(ValueError):
            engine.register_tool("not-callable")  # type: ignore

    def test_chat(self, engine: BridgeEngine) -> None:
        """Test basic chat."""
        response = engine.chat("Hello")
        assert response is not None
        assert response.text is not None
        assert "Hello" in response.text

    def test_chat_adds_to_history(self, engine: BridgeEngine) -> None:
        """Test that chat adds messages to history."""
        initial_count = len(engine.history)
        engine.chat("Test message")
        # Should have at least added user message and assistant response
        assert len(engine.history) >= initial_count + 2

    def test_chat_stream(self, engine: BridgeEngine) -> None:
        """Test streaming chat."""
        chunks = list(engine.chat_stream("Hello"))
        assert len(chunks) > 0
        # All chunks should be BridgeResponse
        for chunk in chunks:
            assert isinstance(chunk, BridgeResponse)

    def test_clear_history(self, engine: BridgeEngine) -> None:
        """Test clearing history."""
        engine.chat("Test")
        assert len(engine.history) > 0
        engine.clear_history()
        assert len(engine.history) == 0

    def test_prune_history(self, engine: BridgeEngine) -> None:
        """Test pruning history."""
        for i in range(15):
            engine.chat(f"Message {i}")

        engine.prune_history()
        # After pruning, should have limited messages
        assert len(engine.history) <= engine.max_history_length

    def test_get_conversation_summary(self, engine: BridgeEngine) -> None:
        """Test getting conversation summary."""
        engine.chat("Test")
        summary = engine.get_conversation_summary()

        assert "total_messages" in summary
        assert "provider" in summary
        assert "has_system_prompt" in summary
        assert "registered_tools" in summary

    def test_engine_repr(self, engine: BridgeEngine) -> None:
        """Test engine string representation."""
        repr_str = repr(engine)
        assert "BridgeEngine" in repr_str
