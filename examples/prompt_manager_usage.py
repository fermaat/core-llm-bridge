#!/usr/bin/env python
"""
Example: Using PromptManager with YAML-based prompts.

Demonstrates how to:
- Load prompts from YAML files
- Render templates with variables
- Create custom prompts
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core_llm_bridge.utils import PromptManager, create_prompt_manager


def example_1_load_default_prompts():
    """Load and use the default prompts from YAML files."""
    print("=" * 60)
    print("Example 1: Loading Default Prompts from YAML")
    print("=" * 60)

    manager = create_prompt_manager()

    print(f"\n✅ Loaded {len(manager.list_templates())} prompts:")
    for name in sorted(manager.list_templates()):
        print(f"  • {name}")

    # Render code_assistant prompt
    print("\n📝 Code Assistant Prompt:")
    print("-" * 40)
    result = manager.render(
        "code_assistant",
        query="Write a Python function to calculate Fibonacci numbers"
    )
    print(result[:200] + "...")


def example_2_custom_prompts():
    """Create and register custom prompts dynamically."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Prompts")
    print("=" * 60)

    manager = PromptManager()

    # Register custom prompts
    manager.register(
        "translator",
        "You are a professional translator. Translate the following to $language:\n\n$text"
    )

    manager.register(
        "proofreader",
        "You are a professional proofreader. Fix grammar and style issues:\n\n$content"
    )

    print(f"\n✅ Registered {len(manager.list_templates())} custom prompts:")
    for name in manager.list_templates():
        print(f"  • {name}")

    # Render custom prompts
    print("\n📝 Translator Prompt:")
    print("-" * 40)
    translator = manager.render(
        "translator",
        language="Spanish",
        text="Hello, how are you today?"
    )
    print(translator)

    print("\n📝 Proofreader Prompt:")
    print("-" * 40)
    proofreader = manager.render(
        "proofreader",
        content="teh quick brown fox jumps ovver the lazi dog"
    )
    print(proofreader)


def example_3_examine_template_variables():
    """Examine what variables a template needs."""
    print("\n" + "=" * 60)
    print("Example 3: Template Variables")
    print("=" * 60)

    manager = create_prompt_manager()

    for name in sorted(manager.list_templates()):
        template = manager.get(name)
        variables = template.get_variables()
        print(f"\n{name}:")
        print(f"  Variables: {variables if variables else 'None (static template)'}")


def example_4_load_custom_yaml_directory():
    """Demonstrate loading prompts from a custom YAML directory."""
    print("\n" + "=" * 60)
    print("Example 4: Loading from Custom Directory")
    print("=" * 60)

    manager = PromptManager()

    # Load from the built-in prompts directory
    prompts_dir = Path(__file__).parent.parent / "src" / "core_llm_bridge" / "prompts"
    loaded = manager.load_from_directory(prompts_dir)

    print(f"\n✅ Loaded {loaded} prompts from {prompts_dir}")
    print(f"Available prompts: {', '.join(sorted(manager.list_templates()))}")


if __name__ == "__main__":
    example_1_load_default_prompts()
    example_2_custom_prompts()
    example_3_examine_template_variables()
    example_4_load_custom_yaml_directory()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
