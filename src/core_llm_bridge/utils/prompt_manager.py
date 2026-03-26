"""
Prompt management and templating utilities.

Provides utilities for managing system prompts, templates, and formatting.
"""

from pathlib import Path
from string import Template
from typing import Any, Optional

import yaml


class PromptTemplate:
    """
    Simple prompt template with variable substitution.

    Supports Pythonic string templates with $variable syntax.

    Example:
        >>> template = PromptTemplate("You are a $role. Respond in $style.")
        >>> prompt = template.render(role="teacher", style="friendly")
        >>> print(prompt)
        # You are a teacher. Respond in friendly.
    """

    def __init__(self, template_str: str) -> None:
        """
        Initialize a prompt template.

        Args:
            template_str: Template string with $variable placeholders
        """
        self.template_str = template_str
        self.template = Template(template_str)

    def render(self, **variables: Any) -> str:
        """
        Render the template with given variables.

        Args:
            **variables: Variables to substitute in template

        Returns:
            Rendered template string

        Raises:
            KeyError: If a required variable is missing
        """
        try:
            return self.template.substitute(**variables)
        except KeyError as e:
            raise KeyError(f"Missing template variable: {e}")

    def get_variables(self) -> set[str]:
        """
        Get all variable names in the template.

        Returns:
            Set of variable names (without $ prefix)
        """
        # Use Template's get_identifiers method
        return set(self.template.get_identifiers())

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PromptTemplate({len(self.get_variables())} variables)"


class PromptManager:
    """
    Manages a collection of prompt templates.

    Allows registering, retrieving, and rendering prompt templates.
    Templates can be registered manually or loaded from YAML files.

    Example:
        >>> manager = PromptManager()
        >>> manager.register("greeting", "Hello $name! How are you?")
        >>> greeting = manager.render("greeting", name="Alice")
        >>> print(greeting)
        # Hello Alice! How are you?
    """

    def __init__(self) -> None:
        """Initialize the prompt manager."""
        self.templates: dict[str, PromptTemplate] = {}

    def register(self, name: str, template_str: str) -> None:
        """
        Register a new prompt template.

        Args:
            name: Unique name for the template
            template_str: Template string

        Raises:
            ValueError: If template name already exists
        """
        if name in self.templates:
            raise ValueError(f"Template '{name}' already registered")

        self.templates[name] = PromptTemplate(template_str)

    def load_from_yaml(self, yaml_path: str | Path) -> None:
        """
        Load a prompt template from a YAML file.

        YAML format:
            name: template_name
            template: |
              Your template with $variables here
            description: (optional) Description of the template

        Args:
            yaml_path: Path to YAML file

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML missing 'name' or 'template' fields
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {yaml_path}: expected dict")

        if "name" not in data or "template" not in data:
            raise ValueError(
                f"YAML must contain 'name' and 'template' fields in {yaml_path}"
            )

        self.register(data["name"], data["template"].strip())

    def load_from_directory(self, directory: str | Path) -> int:
        """
        Load all YAML prompt files from a directory.

        Returns the number of templates loaded.

        Args:
            directory: Path to directory containing .yaml files

        Returns:
            Number of templates successfully loaded

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        loaded = 0
        for yaml_file in sorted(directory.glob("*.yaml")):
            try:
                self.load_from_yaml(yaml_file)
                loaded += 1
            except (ValueError, KeyError) as e:
                # Log but continue loading other files
                print(f"Warning: Failed to load {yaml_file}: {e}")

        return loaded

    def get(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a registered template by name.

        Args:
            name: Name of the template

        Returns:
            PromptTemplate or None if not found
        """
        return self.templates.get(name)

    def render(self, name: str, **variables: Any) -> str:
        """
        Render a template by name.

        Args:
            name: Name of the template
            **variables: Variables to substitute

        Returns:
            Rendered template string

        Raises:
            ValueError: If template not found
        """
        template = self.get(name)
        if template is None:
            raise ValueError(f"Template '{name}' not found")

        return template.render(**variables)

    def list_templates(self) -> list[str]:
        """
        Get list of all registered template names.

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    def unregister(self, name: str) -> bool:
        """
        Unregister a template.

        Args:
            name: Name of the template to remove

        Returns:
            True if template was removed, False if not found
        """
        if name in self.templates:
            del self.templates[name]
            return True
        return False

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PromptManager({len(self.templates)} templates)"


def create_prompt_manager() -> PromptManager:
    """
    Create a PromptManager with common prompts preloaded from YAML files.

    Looks for YAML files in src/core_llm_bridge/prompts/ directory.

    Returns:
        PromptManager with predefined prompts registered from YAML files

    Raises:
        FileNotFoundError: If prompts directory doesn't exist
    """
    manager = PromptManager()

    # Get the prompts directory (relative to this file)
    prompts_dir = Path(__file__).parent.parent / "prompts"

    if not prompts_dir.exists():
        raise FileNotFoundError(
            f"Prompts directory not found at {prompts_dir}. "
            "Create YAML prompt files in src/core_llm_bridge/prompts/"
        )

    loaded_count = manager.load_from_directory(prompts_dir)
    if loaded_count == 0:
        raise FileNotFoundError(
            f"No prompt YAML files found in {prompts_dir}. "
            "Add YAML files with 'name' and 'template' fields."
        )

    return manager


__all__ = [
    "PromptTemplate",
    "PromptManager",
    "create_prompt_manager",
    "COMMON_PROMPTS",
]
