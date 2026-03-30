"""Unit tests for core models."""

from datetime import datetime

from core_llm_bridge.core import (
    BridgeResponse,
    ConversationBuffer,
    LLMConfig,
    Message,
    MessageRole,
    ToolCall,
)


class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)

    def test_create_assistant_message(self) -> None:
        """Test creating an assistant message."""
        msg = Message(role=MessageRole.ASSISTANT, content="Hi there!")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"

    def test_message_to_dict_for_api(self) -> None:
        """Test converting message to API format."""
        msg = Message(role=MessageRole.USER, content="Test")
        api_dict = msg.to_dict_for_api()
        assert api_dict == {"role": "user", "content": "Test"}

    def test_message_string_representation(self) -> None:
        """Test string representation of message."""
        msg = Message(role=MessageRole.USER, content="Hello world")
        assert "[USER]" in str(msg)
        assert "Hello world" in str(msg)


class TestConversationBuffer:
    """Tests for ConversationBuffer model."""

    def test_create_empty_buffer(self) -> None:
        """Test creating an empty conversation buffer."""
        buffer = ConversationBuffer()
        assert len(buffer) == 0
        assert buffer.messages == []

    def test_add_message(self) -> None:
        """Test adding messages to buffer."""
        buffer = ConversationBuffer()
        buffer.add_user_message("Hello")
        buffer.add_assistant_message("Hi!")

        assert len(buffer) == 2
        assert buffer.messages[0].role == MessageRole.USER
        assert buffer.messages[1].role == MessageRole.ASSISTANT

    def test_add_system_message(self) -> None:
        """Test adding system message."""
        buffer = ConversationBuffer(system_prompt="You are helpful")
        assert buffer.system_prompt == "You are helpful"

    def test_get_messages_for_api_without_system(self) -> None:
        """Test getting messages in API format without system prompt."""
        buffer = ConversationBuffer()
        buffer.add_user_message("Hello")
        buffer.add_assistant_message("Hi!")

        api_messages = buffer.get_messages_for_api()
        assert len(api_messages) == 2
        assert api_messages[0]["role"] == "user"
        assert api_messages[1]["role"] == "assistant"

    def test_get_messages_for_api_with_system(self) -> None:
        """Test getting messages in API format with system prompt."""
        buffer = ConversationBuffer(system_prompt="Be helpful")
        buffer.add_user_message("Hello")

        api_messages = buffer.get_messages_for_api()
        assert len(api_messages) == 2
        assert api_messages[0]["role"] == "system"
        assert api_messages[0]["content"] == "Be helpful"
        assert api_messages[1]["role"] == "user"

    def test_clear_buffer(self) -> None:
        """Test clearing buffer."""
        buffer = ConversationBuffer()
        buffer.add_user_message("Hello")
        buffer.add_assistant_message("Hi!")
        assert len(buffer) == 2

        buffer.clear()
        assert len(buffer) == 0

    def test_prune_old_messages(self) -> None:
        """Test pruning old messages."""
        buffer = ConversationBuffer()
        for i in range(15):
            buffer.add_user_message(f"Message {i}")

        assert len(buffer) == 15
        buffer.prune_old_messages(keep_last_n=5)
        assert len(buffer) == 5

    def test_buffer_representation(self) -> None:
        """Test string representation."""
        buffer = ConversationBuffer()
        buffer.add_user_message("Test")
        repr_str = repr(buffer)
        assert "ConversationBuffer" in repr_str
        assert "1 messages" in repr_str


class TestBridgeResponse:
    """Tests for BridgeResponse model."""

    def test_create_simple_response(self) -> None:
        """Test creating a simple response."""
        response = BridgeResponse(text="Hello!")
        assert response.text == "Hello!"
        assert response.finish_reason == "stop"
        assert len(response.tool_calls) == 0

    def test_response_with_tool_calls(self) -> None:
        """Test response with tool calls."""
        tool_call = ToolCall(
            id="tool1",
            function_name="add",
            arguments={"a": 1, "b": 2},
        )
        response = BridgeResponse(text="I'll add those", tool_calls=[tool_call])
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function_name == "add"

    def test_response_string_representation(self) -> None:
        """Test string representation."""
        response = BridgeResponse(text="Hello world!", finish_reason="stop")
        repr_str = str(response)
        assert "Hello world!" in repr_str
        assert "stop" in repr_str


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self) -> None:
        """Test creating a tool call."""
        tool_call = ToolCall(
            id="tool1",
            function_name="calculate",
            arguments={"operation": "add", "values": [1, 2, 3]},
        )
        assert tool_call.id == "tool1"
        assert tool_call.function_name == "calculate"
        assert tool_call.arguments["operation"] == "add"
        assert tool_call.result is None

    def test_tool_call_with_result(self) -> None:
        """Test tool call with result."""
        tool_call = ToolCall(
            id="tool1",
            function_name="add",
            arguments={"a": 1, "b": 2},
            result="3",
        )
        assert tool_call.result == "3"


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = LLMConfig()
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens is None
        assert config.stop_sequences == []

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = LLMConfig(
            temperature=0.9,
            top_p=0.95,
            max_tokens=512,
            stop_sequences=["END"],
        )
        assert config.temperature == 0.9
        assert config.max_tokens == 512
        assert "END" in config.stop_sequences
