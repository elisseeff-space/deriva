# LLM Adapter

Multi-provider LLM abstraction using pydantic-ai with caching and structured output support.

**Version:** 2.0.0

## Purpose

The LLM adapter provides a unified interface for querying multiple LLM providers (Azure OpenAI, OpenAI, Anthropic, Mistral, Ollama, LM Studio) using **pydantic-ai** for agent-based interactions with automatic retries and Pydantic-based structured output parsing.

## Key Exports

```python
from deriva.adapters.llm import (
    LLMManager,             # Main service class
    # Providers
    create_provider,        # Factory function
    AzureOpenAIProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    LMStudioProvider,
    ProviderConfig,
    CompletionResult,
    # Response types
    LLMResponse,
    BaseResponse,
    LiveResponse,
    CachedResponse,
    FailedResponse,
    ResponseType,
    StructuredOutputMixin,
    # Caching
    CacheManager,
    cached_llm_call,
    # Exceptions
    LLMError,
    ConfigurationError,
    APIError,
    CacheError,
    ValidationError,
)
```

## Basic Usage

```python
from deriva.adapters.llm import LLMManager

# Uses provider from .env (LLM_PROVIDER, LLM_MODEL, etc.)
llm = LLMManager()

# Simple query
response = llm.query("What is Python?")
if response.response_type == "live":
    print(response.content)
```

## Structured Output with pydantic-ai

Uses pydantic-ai agents for type-safe, validated responses:

```python
from pydantic import BaseModel, Field
from deriva.adapters.llm import LLMManager

class BusinessConcept(BaseModel):
    name: str = Field(description="Concept name")
    concept_type: str = Field(description="actor, service, entity, etc.")
    description: str

llm = LLMManager()
result = llm.query(
    prompt="Extract the main business concept from this code...",
    response_model=BusinessConcept
)
# result is a validated BusinessConcept instance (via pydantic-ai agent)
print(result.name)
```

## File Structure

```text
deriva/adapters/llm/
├── __init__.py           # Package exports
├── manager.py            # LLMManager class
├── providers.py          # Provider implementations
├── models.py             # Response types and exceptions
└── cache.py              # CacheManager and caching utilities
```

## Configuration

Set provider via environment variables in `.env`:

```bash
# Primary provider
LLM_PROVIDER=azure          # azure, openai, anthropic, ollama, lmstudio
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-key
LLM_API_URL=https://...     # Optional custom endpoint

# Multiple models for benchmarking
LLM_OLLAMA_LLAMA_PROVIDER=ollama
LLM_OLLAMA_LLAMA_MODEL=llama3.2
LLM_OLLAMA_LLAMA_URL=http://localhost:11434/api/chat

# LM Studio (local, OpenAI-compatible)
LLM_LMSTUDIO_LOCAL_PROVIDER=lmstudio
LLM_LMSTUDIO_LOCAL_MODEL=local-model
LLM_LMSTUDIO_LOCAL_URL=http://localhost:1234/v1/chat/completions
```

## Providers

All providers are implemented via pydantic-ai's model abstraction:

| Provider     | pydantic-ai Model  | Description                         |
|--------------|--------------------| ------------------------------------|
| Azure OpenAI | `AzureOpenAIModel` | Azure-hosted OpenAI models          |
| OpenAI       | `OpenAIModel`      | OpenAI API direct                   |
| Anthropic    | `AnthropicModel`   | Claude models                       |
| Mistral      | `MistralModel`     | Mistral AI models                   |
| Ollama       | `OllamaModel`      | Local Ollama models                 |
| LM Studio    | `OpenAIModel`      | Local LM Studio (OpenAI-compatible) |

## Response Types

| Type | When | Key Fields |
|------|------|------------|
| `LiveResponse` | Fresh API call | `content`, `usage`, `finish_reason` |
| `CachedResponse` | From cache | `content`, `cache_key`, `cached_at` |
| `FailedResponse` | Error occurred | `error`, `error_type` |

## LLMManager Methods

| Method | Description |
|--------|-------------|
| `query(prompt, response_model, temperature, max_tokens)` | Send LLM query |
| `load_benchmark_models()` | Load multiple model configs from env |

## Caching

- Responses cached to `workspace/cache/` by default
- Cache key = SHA256(prompt + model + schema)
- Disable with `LLM_NOCACHE=true` in `.env`
- Use `cached_llm_call` decorator for custom caching

## Structured Output (JSON Schema Enforcement)

Enable API-level JSON schema enforcement for guaranteed valid JSON responses:

```bash
# Per-model configuration in .env
LLM_OPENAI_GPT41MINI_STRUCTURED_OUTPUT=true
LLM_ANTHROPIC_HAIKU_STRUCTURED_OUTPUT=true
LLM_MISTRAL_DEVSTRAL_STRUCTURED_OUTPUT=true
LLM_OLLAMA_NEMOTRON_STRUCTURED_OUTPUT=true
```

**Supported Providers:**

| Provider | Support | Implementation |
|----------|---------|----------------|
| OpenAI | ✅ | `response_format: {type: "json_schema"}` |
| Azure | ✅ | Same as OpenAI |
| Anthropic | ✅ | `output_format` + beta header |
| Mistral | ✅ | `response_format: {type: "json_schema"}` |
| Ollama | ✅ | `format: <schema>` |
| LMStudio | ✅ | Same as OpenAI |
| ClaudeCode | ❌ | CLI-based, no structured output |

**Behavior:**

- `structured_output=true`: JSON schema passed to provider API for server-side enforcement
- `structured_output=false` (default): Only `json_mode` enabled, schema used for client-side validation only

**Programmatic Usage:**

```python
from deriva.adapters.llm import LLMManager
from deriva.adapters.llm.manager import load_benchmark_models

# Load model with structured_output=true from .env
models = load_benchmark_models()
llm = LLMManager.from_config(models["openai-gpt41mini"])

# The schema will be enforced at the API level
result = llm.query(
    "Extract business concepts...",
    schema={"type": "object", "properties": {...}}
)
```

## See Also

- [CONTRIBUTING.md](../../../CONTRIBUTING.md) - Architecture and LLM usage guidelines
