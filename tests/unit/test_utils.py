"""Unit tests for utilities."""

import tempfile
from pathlib import Path

import pytest

from core_llm_bridge.utils import PromptManager, PromptTemplate, TokenCounter, create_prompt_manager


class TestTokenCounter:
    """Tests for TokenCounter utility."""

    def test_estimate_tokens_empty_string(self) -> None:
        """Test estimating tokens for empty string."""
        tokens = TokenCounter.estimate_tokens("")
        assert tokens == 0

    def test_estimate_tokens_single_word(self) -> None:
        """Test estimating tokens for single word."""
        tokens = TokenCounter.estimate_tokens("hello")
        assert tokens >= 1

    def test_estimate_tokens_multiple_words(self) -> None:
        """Test estimating tokens for multiple words."""
        tokens = TokenCounter.estimate_tokens("hello world test example")
        assert tokens >= 4

    def test_estimate_tokens_with_model(self) -> None:
        """Test token estimation with different models."""
        text = "This is a test message"
        tokens_default = TokenCounter.estimate_tokens(text, "default")
        tokens_llama = TokenCounter.estimate_tokens(text, "llama")
        tokens_gpt = TokenCounter.estimate_tokens(text, "gpt")

        # All should be positive
        assert tokens_default > 0
        assert tokens_llama > 0
        assert tokens_gpt > 0

    def test_count_messages_tokens(self) -> None:
        """Test counting tokens in messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        tokens = TokenCounter.count_messages_tokens(messages)
        assert tokens > 0

    def test_will_fit_in_context(self) -> None:
        """Test checking if text fits in context."""
        short_text = "Hello"
        long_text = "word " * 1000

        assert TokenCounter.will_fit_in_context(short_text, max_tokens=100)
        assert not TokenCounter.will_fit_in_context(long_text, max_tokens=100)

    def test_truncate_to_fit(self) -> None:
        """Test truncating text to fit."""
        long_text = "word " * 100
        truncated = TokenCounter.truncate_to_fit(long_text, max_tokens=50)

        # Truncated should be shorter and fit in context
        assert len(truncated) <= len(long_text)
        assert TokenCounter.will_fit_in_context(truncated, max_tokens=50, safety_margin=10)


class TestPromptTemplate:
    """Tests for PromptTemplate utility."""

    def test_create_template(self) -> None:
        """Test creating a prompt template."""
        template = PromptTemplate("You are a $role")
        assert "role" in template.get_variables()

    def test_render_template(self) -> None:
        """Test rendering a template."""
        template = PromptTemplate("You are a $role in $domain")
        result = template.render(role="teacher", domain="math")
        assert result == "You are a teacher in math"

    def test_render_missing_variable(self) -> None:
        """Test rendering with missing variable."""
        template = PromptTemplate("You are a $role")
        with pytest.raises(KeyError):
            template.render(something="else")

    def test_get_variables(self) -> None:
        """Test getting template variables."""
        template = PromptTemplate("Hello $name, you are $age years old")
        vars = template.get_variables()
        assert "name" in vars
        assert "age" in vars
        assert len(vars) == 2


class TestPromptManager:
    """Tests for PromptManager utility."""

    def test_register_template(self) -> None:
        """Test registering a template."""
        manager = PromptManager()
        manager.register("greeting", "Hello $name!")
        templates = manager.list_templates()
        assert "greeting" in templates

    def test_register_duplicate(self) -> None:
        """Test registering duplicate template."""
        manager = PromptManager()
        manager.register("greeting", "Hello!")
        with pytest.raises(ValueError):
            manager.register("greeting", "Hi!")

    def test_get_template(self) -> None:
        """Test getting a template."""
        manager = PromptManager()
        manager.register("test", "Test template")
        template = manager.get("test")
        assert template is not None
        assert "test" in template.template_str

    def test_render_template(self) -> None:
        """Test rendering through manager."""
        manager = PromptManager()
        manager.register("greeting", "Hello $name!")
        result = manager.render("greeting", name="Alice")
        assert result == "Hello Alice!"

    def test_render_nonexistent(self) -> None:
        """Test rendering nonexistent template."""
        manager = PromptManager()
        with pytest.raises(ValueError):
            manager.render("nonexistent")

    def test_unregister_template(self) -> None:
        """Test unregistering a template."""
        manager = PromptManager()
        manager.register("test", "Test")
        assert manager.unregister("test") is True
        assert manager.unregister("test") is False

    def test_list_templates(self) -> None:
        """Test listing templates."""
        manager = PromptManager()
        manager.register("greeting", "Hello")
        manager.register("farewell", "Goodbye")
        templates = manager.list_templates()
        assert len(templates) == 2
        assert "greeting" in templates
        assert "farewell" in templates

    def test_load_from_yaml(self) -> None:
        """Test loading a template from YAML file."""
        manager = PromptManager()

        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_prompt\n")
            f.write("template: |\n")
            f.write("  You are a $role\n")
            f.write("description: Test prompt\n")
            yaml_path = f.name

        try:
            manager.load_from_yaml(yaml_path)
            assert "test_prompt" in manager.list_templates()

            # Test rendering
            result = manager.render("test_prompt", role="teacher")
            assert "teacher" in result
        finally:
            Path(yaml_path).unlink()

    def test_load_from_directory(self) -> None:
        """Test loading templates from a directory."""
        manager = PromptManager()

        # Create a temporary directory with YAML files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create first YAML file
            yaml1 = tmpdir_path / "prompt1.yaml"
            yaml1.write_text("name: prompt1\ntemplate: Hello $name\n")

            # Create second YAML file
            yaml2 = tmpdir_path / "prompt2.yaml"
            yaml2.write_text("name: prompt2\ntemplate: Goodbye $name\n")

            # Load from directory
            loaded = manager.load_from_directory(tmpdir_path)
            assert loaded == 2
            assert "prompt1" in manager.list_templates()
            assert "prompt2" in manager.list_templates()


class TestCreatePromptManager:
    """Tests for create_prompt_manager factory function."""

    def test_create_with_yaml_prompts(self) -> None:
        """Test creating manager with prompts from YAML files."""
        manager = create_prompt_manager()
        templates = manager.list_templates()

        # Should have exactly 4 common prompts from YAML
        assert len(templates) == 4
        assert "code_assistant" in templates
        assert "data_analyst" in templates
        assert "creative_writer" in templates
        assert "tutor" in templates

    def test_yaml_prompts_are_usable(self) -> None:
        """Test that YAML prompts can be rendered."""
        manager = create_prompt_manager()

        # Test code_assistant
        code_query = "Write a hello world function in Python"
        result = manager.render("code_assistant", query=code_query)
        assert "hello world" in result.lower()

        # Test creative_writer
        writer_query = "Write a sci-fi story with optimistic tone"
        result = manager.render("creative_writer", query=writer_query)
        assert len(result) > 0

        # Test tutor
        tutor_query = "Explain calculus"
        result = manager.render("tutor", query=tutor_query)
        assert "tutor" in result.lower() or "explain" in result.lower()

        # Test data_analyst
        analyst_query = "Analyze sales trends"
        result = manager.render("data_analyst", query=analyst_query)
        assert len(result) > 0
