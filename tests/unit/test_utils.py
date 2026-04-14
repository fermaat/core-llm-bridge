"""Unit tests for utilities."""

import tempfile
from pathlib import Path

import pytest

from core_llm_bridge.utils import PromptManager, PromptTemplate, create_prompt_manager


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
        assert template.template_str == "Test template"

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

        # Should include the core prompt templates from YAML
        assert len(templates) >= 4
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
