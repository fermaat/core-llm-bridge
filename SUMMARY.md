# core-llm-bridge — Claude reference summary

## Purpose
Unified LLM abstraction layer. Wraps provider-specific APIs behind a common `BridgeEngine` interface so that downstream projects (e.g. copper) can swap models and providers without changing application code.

## Architecture

```
src/core_llm_bridge/
├── core/
│   ├── base.py          # BaseLLMProvider (abstract): generate/generate_stream/async variants; ToolProvider mixin
│   ├── engine.py        # BridgeEngine — orchestrator: history, pruning, tool registry, chat/chat_stream/async
│   └── models.py        # Pydantic models: Message, MessageRole, BridgeResponse, ConversationBuffer, LLMConfig, ToolCall
├── providers/
│   ├── factory.py       # Provider factory (create_provider by name)
│   └── ollama.py        # OllamaProvider — httpx-based, sync + async, streaming, health check
├── utils/
│   ├── prompt_manager.py  # PromptTemplate ($var syntax) + PromptManager (register/render/load from YAML)
│   └── token_counter.py   # Token counting utilities
├── config.py            # Settings (pydantic-settings, .env), loguru logger setup
└── exceptions.py        # OllamaConnectionError, OllamaModelNotFoundError, OllamaTimeoutError, LLMProviderError
```

## Key classes

**BridgeEngine** (`core/engine.py`)
- `BridgeEngine(provider, system_prompt, max_history_length)` — main entry point
- `engine.chat(user_input, config?)` → `BridgeResponse`
- `engine.chat_stream(user_input, config?)` → `Generator[BridgeResponse]`
- `engine.chat_async(...)` / `engine.chat_stream_async(...)` — async variants
- `engine.prune_history(keep_last_n)` — prunes old messages, saves summary to `internal_state`
- `engine.register_tool(func)` — registers callable for tool-call handling
- Auto-prunes when `len(history) + 2 > max_history_length`

**BaseLLMProvider** (`core/base.py`)
- Abstract: `generate()`, `generate_stream()` must be implemented
- Default async fallbacks wrap sync methods (can be overridden)
- `validate_connection()` / `health_check()` — connection/health checks

**OllamaProvider** (`providers/ollama.py`)
- Talks to Ollama via httpx (`/api/chat`, `/api/tags`)
- Config via `.env`: `OLLAMA_BASE_URL`, `OLLAMA_DEFAULT_MODEL`, `OLLAMA_TIMEOUT`
- Raises typed exceptions: `OllamaConnectionError`, `OllamaModelNotFoundError`, `OllamaTimeoutError`

**Models** (`core/models.py`)
- `BridgeResponse`: `text`, `finish_reason`, `tokens_used`, `tool_calls`, `raw_response`, `metadata`
- `ConversationBuffer`: list of `Message`, system prompt, `get_messages_for_api()` → list[dict]
- `LLMConfig`: `temperature`, `top_p`, `max_tokens`, `stop_sequences`, `system_prompt`

**PromptManager** (`utils/prompt_manager.py`)
- `PromptTemplate` — Python `string.Template` ($var substitution)
- `PromptManager.register(name, template_str)` / `.render(name, **vars)` / `.load_from_yaml(path)`
- YAML format: single (`name`+`template`) or multiple (`prompts: [...]`)

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service URL |
| `OLLAMA_DEFAULT_MODEL` | `gemma3:4b` | Default model |
| `OLLAMA_TIMEOUT` | `300` | Request timeout (s) |
| `LOG_LEVEL` | `INFO` | Logging level |

## Dependencies
- Runtime: pydantic v2, pydantic-settings, httpx, loguru, pyyaml, openai
- Dev: pytest, pytest-cov, pytest-asyncio, black, mypy, ruff

## Providers status
- Ollama ✓ — implemented (sync, async, streaming)
- OpenAI — planned (openai SDK already in deps)
- Anthropic — planned

## Consumers
- `copper` — uses `BridgeAdapter` in `copper/llm/bridge_adapter.py` to wrap `BridgeEngine` into copper's `LLMBase`
