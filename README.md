# Text Summarizer Agent — ADK Evaluation Loop Demo

A text summarization agent built with Google's **Agent Development Kit (ADK)** that demonstrates how to use evaluation loops to systematically improve agent prompts and outputs.

## What This Project Does

The agent takes long text and converts it into concise bullet-point summaries, then persists them into an Obsidian vault (a "second brain"). It can also read your Gmail inbox. The real value of the repo is the **evaluation loop**: edit the agent's instructions in `agent.py`, run ADK evals, and watch the `response_match_score` (ROUGE-1 word overlap) change.

## Quick Start

### 1. Prerequisites

- Docker (Compose v2) — the one-command path needs nothing else
- Python 3.11+ and `uv` — only for the local (no-Docker) alternative
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- (Optional) A free OpenRouter API key from [OpenRouter](https://openrouter.ai/keys)

### 2. Run it (single command)

```bash
./run.sh
```

That picks your Obsidian vault, builds and starts both containers, and prints
the URLs:

- **Agent web UI:** http://localhost:8001
- **Onboarding guide:** [docs/getting-started.md](docs/getting-started.md)

Already use Obsidian? `./run.sh` finds your existing vaults — including the ones
in your Obsidian config — and asks which one the agent should use, then remembers
the answer. One vault needs no configuration; several are never guessed between.
See [using a vault you already have](docs/INTEGRATIONS.md#using-a-vault-you-already-have).

On the first run it also creates `.env` from `.env.example` and stops — edit it
to put in your `GEMINI_API_KEY`, then run `./run.sh` again.

> **Local (no Docker), optional:** 
1. Run `cd text_summarizer && uv sync`
2. Run `uv run adk web text_summarizer`
3. Access http://127.0.0.1:8000. 

Same agent, no containers.

### 3. Configure Environment Variables

Copy `.env.example` to your local `.env` and set what you need. This is every variable:

```env
# === Model provider ===
# "gemini" (default) or "openrouter"
MODEL_PROVIDER=gemini
# Gemini (free tier) — key at https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# OpenRouter (optional, free models) — key at https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
MODEL_ALIAS=gemma              # gemma / qwen / nvidia

# === Langfuse (optional observability) ===
LANGFUSE_PUBLIC_KEY=pk-lf-your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_langfuse_secret_key_here
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PROMPT_NAME=text_summarizer
LANGFUSE_TRACE_NAME=          # default: text_summarizer
LANGFUSE_AUTH_CHECK_TIMEOUT=5

# === Obsidian MCP (optional) — set exactly one ===
OBSIDIAN_VAULT_PATH=/absolute/path/to/vault
OBSIDIAN_MCP_URL=http://127.0.0.1:37842/mcp

# === Gmail (optional, read-only) ===
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REFRESH_TOKEN=

# === Second brain — which vault (optional) ===
SECOND_BRAIN_VAULT=/absolute/path/to/your/vault   # point at the vault's PARENT dir
OBSIDIAN_VAULT_NAME=                             # only when that dir holds several
VAULT_NAME=                                      # display label only, rarely needed

# === Cache & eval (optional) ===
# "false" bypasses the vault cache. Keep it OFF for `adk eval`.
CACHE_ENABLED=true

# === Docker only (optional) ===
OBSIDIAN_VAULT_PARENT_HOST=./vaults   # host dir holding the vault(s) (auto-creates "agent-vault") if empty
VAULT_PARENT=/vaults                  # in-container mount point, matched to the line above
OBSIDIAN_MCP_PORT=37842
```

All capabilities are optional and load automatically when configured:
**[Gmail](docs/INTEGRATIONS.md)** (read-only), **[Obsidian second brain](docs/INTEGRATIONS.md#obsidian-vault-second-brain)**,
**[Langfuse tracing](docs/ARCHITECTURE.md#observability-langfuse)**, and the
**[`**Sources**` provenance block](docs/ARCHITECTURE.md#sources-provenance)**.
`.env.example` carries the full comments for each variable.

### 4. Try It

```bash
# Web UI: http://localhost:8001
# Onboarding guide: docs/getting-started.md

# CLI (Docker, one-off turn):
docker compose exec agent-testing sh -lc \
  'cd /workspace/text_summarizer && .venv/bin/adk run text_summarizer "Summarize: Your long text here..."'

# CLI (local, no Docker):
cd text_summarizer
uv run adk run text_summarizer "Summarize: Your long text here..."
```

### 5. Run Evaluations

```bash
CACHE_ENABLED=false uv run adk eval text_summarizer \
  text_summarizer/tests/eval/simple_test.test.json \
  --config_file_path text_summarizer/tests/eval/test_config.json \
  --print_detailed_results
```

`CACHE_ENABLED=false` is mandatory — evals call the real agent and hit the vault cache otherwise.

## Documentation

| Doc | Contents |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | Onboarding: keys, run, first prompts, troubleshooting |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Eval criteria, running evals, hands-on experiments, auto-optimization, creating test cases |
| [docs/TESTING.md](docs/TESTING.md) | Prompt examples for the web UI (good / bad / edge cases) |
| [docs/MODELS.md](docs/MODELS.md) | Switching between Gemini and OpenRouter free models |
| [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) | Setting up read-only Gmail access and the Obsidian "second brain" |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Project structure, agent flow, `**Sources**` provenance, Docker topology |

## License

This project is for learning purposes. Built with [Google ADK](https://adk.dev).