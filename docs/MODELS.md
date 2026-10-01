# Model Switching (Dependency Injection)

The agent supports switching between Gemini and OpenRouter via environment
variables. The same `MODEL_PROVIDER` env var controls both the agent and the
auto-optimizer.

## Available Models

| Provider | Model | Alias | Cost |
|----------|-------|-------|------|
| Gemini | `gemini-3.5-flash-lite` | — | Free tier |
| OpenRouter | `google/gemma-4-26b-a4b-it:free` | `gemma` | Free |
| OpenRouter | `qwen/qwen3.8-27b:free` | `qwen` | Free |
| OpenRouter | `nvidia/nemotron-3.5-lightning:free` | `nvidia` | Free |

Both paths return a `FallbackModel`: Gemini first, then free OpenRouter models on
quota exhaustion (HTTP 429) or transient 5xx errors.

## How It Works

In `agent.py`, **both** branches return a `FallbackModel` — a `LiteLlm` only ever
appears inside one:

```python
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "gemini")

def _openrouter_llm(model_name: str) -> LiteLlm:
    return LiteLlm(model=f"openrouter/{model_name}")

def get_model():
    if MODEL_PROVIDER == "openrouter":
        primary_name = OPENROUTER_MODELS.get(MODEL_ALIAS) or _free_openrouter_models()[0]
        fallback_names = [name for name in _free_openrouter_models() if name != primary_name]
        return FallbackModel(models=[
            _openrouter_llm(primary_name),
            *(_openrouter_llm(name) for name in fallback_names),
        ])
    # Default provider is Gemini; fall back to free OpenRouter models on quota
    # exhaustion (HTTP 429) or transient 5xx errors.
    return FallbackModel(models=[
        GEMINI_MODEL,
        *(_openrouter_llm(name) for name in _free_openrouter_models()),
    ])
```

So switching provider does not turn the fallback chain off — it changes which
model is tried *first* in it. The `:free` OpenRouter models are the safety net
either way.

## Switching Providers

Just change the env var (from the repo root — `--project text_summarizer` is
required, there is no project at the root):

```bash
# Use Gemini (default)
MODEL_PROVIDER=gemini uv run --project text_summarizer adk web text_summarizer

# Use OpenRouter free models
MODEL_PROVIDER=openrouter MODEL_ALIAS=gemma \
  uv run --project text_summarizer adk web text_summarizer
```

- `MODEL_ALIAS` picks the free model: `gemma` / `qwen` / `nvidia`
- If `MODEL_ALIAS` is empty, the first free model in `OPENROUTER_MODELS` is used
- Models referenced are free-tier aliases used by this project; don't "fix" the
  names to older released models