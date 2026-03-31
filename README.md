# llm-bridge

Unified interface for interacting with local and cloud-based Large Language Models (LLMs).

## Features

- 🔌 **Unified Provider Interface**: Abstract away provider-specific details
- 💾 **Conversation Memory**: Automatic management of conversation history
- 🌊 **Streaming Support**: Handle real-time responses from LLMs
- 🛠️ **Tool Integration**: Support for function calling (MCP protocol)
- 🔐 **Configuration Management**: Environment-based settings with type safety
- 📊 **Token Management**: Intelligent context window handling
- 🧪 **Comprehensive Testing**: Unit and integration tests with high coverage

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-bridge.git
cd llm-bridge

# Install with pdm
pdm install

# Or install specific extras
pdm install -d  # install dev dependencies
```

### Basic Usage

```python
from core_llm_bridge import BridgeEngine
from core_llm_bridge.providers import OllamaProvider

# Initialize provider
provider = OllamaProvider(model="llama2")

# Create bridge engine
bridge = BridgeEngine(provider=provider)

# Simple chat
response = bridge.chat("Hello! What's your name?")
print(response.text)
```

### Streaming

```python
# Stream responses in real-time
for chunk in bridge.chat_stream("Tell me a story"):
    print(chunk.text, end="", flush=True)
```

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running (for local LLM support)
- PDM as package manager

### Setup Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. Pull a model (in another terminal):
   ```bash
   ollama pull llama2
   ```

## Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama2
LOG_LEVEL=INFO
```

## Project Structure

```
llm-bridge/
├── src/core_llm_bridge/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base classes
│   │   ├── engine.py         # Main BridgeEngine
│   │   └── models.py         # Pydantic models
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py           # Provider base class
│   │   ├── ollama.py         # Ollama implementation
│   │   └── mock.py           # Mock provider for tests
│   └── utils/
│       ├── __init__.py
│       ├── token_counter.py  # Token management
│       └── prompt_manager.py # Prompt templates
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── test_models.py        # Test models
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── data/                 # Test fixtures
├── examples/
│   ├── simple_chat.py
│   ├── streaming.py
│   └── with_tools.py
├── docs/
├── pyproject.toml
├── pytest.ini
├── .env.example
└── README.md
```

## Development

### Install Development Dependencies

```bash
# Install all dependencies including dev
pdm install
```

### Run Tests

```bash
# Run all tests
pdm run pytest

# Run specific test file
pdm run pytest tests/unit/test_models.py

# Run with coverage
pdm run pytest --cov

# Run only unit tests
pdm run pytest -m unit

# Run only integration tests (requires Ollama)
pdm run pytest -m integration

# Skip tests requiring Ollama
pdm run pytest -m "not requires_ollama"
```

### Code Quality

```bash
# Format code with black
pdm run black src tests

# Check types with mypy
pdm run mypy src

# Lint with ruff
pdm run ruff check src tests

# Fix linting issues
pdm run ruff check --fix src tests
```

## Examples

### Simple Chat

See [examples/simple_chat.py](examples/simple_chat.py)

### Streaming

See [examples/streaming.py](examples/streaming.py)

### With Tools (Function Calling)

See [examples/with_tools.py](examples/with_tools.py)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
