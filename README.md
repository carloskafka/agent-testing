> **Live demo → <https://carloskafka.github.io/agent-testing/#/overview>**
> A Material 3 walkthrough of the whole thing: end-to-end animations of the main
> flow, the Obsidian graph and Gmail, then quick start, integrations, the vault, troubleshooting.

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
- **Onboarding guide:** [docs/getting-started.md](docs/getting-started.md), or the
  [live walkthrough](https://carloskafka.github.io/agent-testing/#/overview)

Already use Obsidian? `./run.sh` finds your existing vaults — including the ones
in your Obsidian config — and asks which one the agent should use, then remembers
the answer. One vault needs no configuration; several are never guessed between.
See [using a vault you already have](docs/INTEGRATIONS.md#using-a-vault-you-already-have).

On the first run it also creates `.env` from `.env.example` if it isn't there yet,
and prints a reminder to set your `GEMINI_API_KEY` — but it does **not** stop. It
carries on into the vault menu and `docker compose up -d --build`, so edit `.env`
and re-run `./run.sh` once your key is in.

> **Local (no Docker), optional:** 
> 1. Run `uv sync --project text_summarizer`
> 2. Run `uv run --project text_summarizer adk web text_summarizer`
> 3. Access http://127.0.0.1:8000. 

Same agent, no containers. Two things to know about this path: there is **no
project at the repo root** — `pyproject.toml` lives in `text_summarizer/` — so
every `uv run` needs `--project text_summarizer` or it fails with
`Failed to spawn: adk`; and it binds :8000 directly, where the Docker path
publishes **:8001 → 8000**. Everything below this section is run from the repo root.

### 3. Configure Environment Variables

**[`.env.example`](.env.example) is the single source of truth** for every variable
and its comments — it is the file `./run.sh` copies into a fresh `.env`, and it is
kept fuller than anything duplicated here on purpose. Two copies of a config
reference drift, and these had. Only the handful you are most likely to touch are
shown inline:

```env
# === Model provider ===  "gemini" (default) or "openrouter"
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here       # https://aistudio.google.com/app/apikey

# === Obsidian ===  set EXACTLY ONE of these two (OBSIDIAN_MCP_URL wins if both are set)
OBSIDIAN_VAULT_PATH=/absolute/path/to/your/vault
# OBSIDIAN_MCP_URL=http://127.0.0.1:37842/mcp

# === Second brain — which vault (optional) ===
SECOND_BRAIN_VAULT=/absolute/path/to/your/vault   # point at the vault's PARENT dir
OBSIDIAN_VAULT_NAME=                              # only when that dir holds several
OBSIDIAN_CONFIG_DIR=                              # only for an unusual Obsidian install

# === Cache & eval (optional) ===
# "false" bypasses the vault cache. Keep it OFF for `adk eval`.
CACHE_ENABLED=true
```

All capabilities are optional and load automatically when configured:
**[Gmail](docs/INTEGRATIONS.md)** (read-only), **[Obsidian second brain](docs/INTEGRATIONS.md#obsidian-vault-second-brain)**,
**[Langfuse tracing](docs/ARCHITECTURE.md#observability-langfuse)**, and the
**[`**Sources**` provenance block](docs/ARCHITECTURE.md#sources-provenance)**.
`.env.example` carries the full comments for each variable, and
[AGENTS.md](AGENTS.md#environment-variables-see-envexample) the full table with
what each one is read by.

### 4. Try It

```bash
# Web UI: http://localhost:8001
# Onboarding guide: docs/getting-started.md

# CLI (Docker, one-off turn):
docker compose exec agent-testing sh -lc \
  'cd /workspace/text_summarizer && .venv/bin/adk run text_summarizer "Summarize: Your long text here..."'

# CLI (local, no Docker, from the repo root):
uv run --project text_summarizer adk run text_summarizer "Summarize: Your long text here..."
```

### 5. Run Evaluations

```bash
CACHE_ENABLED=false uv run --project text_summarizer adk eval text_summarizer \
  text_summarizer/tests/eval/simple_test.test.json \
  --config_file_path text_summarizer/tests/eval/test_config.json \
  --print_detailed_results
```

`CACHE_ENABLED=false` is mandatory — evals call the real agent, which writes to the
vault, so a second run hits the cache and grades a replay. `--project
text_summarizer` is equally mandatory: there is no project at the repo root, so a
bare `uv run adk …` fails with `Failed to spawn: adk`.

## Documentation

| Doc | Contents |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | Onboarding: keys, run, first prompts, troubleshooting |
| [Live walkthrough](https://carloskafka.github.io/agent-testing/#/overview) | Same material as a Material 3 web app, published with GitHub Pages |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Eval criteria, running evals, hands-on experiments, auto-optimization, creating test cases |
| [docs/TESTING.md](docs/TESTING.md) | Prompt examples for the web UI (good / bad / edge cases) |
| [docs/MODELS.md](docs/MODELS.md) | Switching between Gemini and OpenRouter free models |
| [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) | Setting up read-only Gmail access and the Obsidian "second brain" |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Project structure, agent flow, `**Sources**` provenance, Docker topology |

## License

This project is for learning purposes. Built with [Google ADK](https://adk.dev).