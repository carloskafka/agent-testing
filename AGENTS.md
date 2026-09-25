# AGENTS.md

## What this repository is

A hands-on project demonstrating a **text summarization agent** built with Google's **Agent Development Kit (ADK)**, plus the workflow of **evaluating and auto-optimizing agent instructions** through an eval loop.

The agent takes long text and returns concise bullet-point summaries. The real value of the repo is the **evaluation loop**: you edit the agent's instructions in `agent.py`, run ADK evals, and watch the `response_match_score` (ROUGE-1 word overlap) change.

Git remote: `https://github.com/carloskafka/agent-testing.git` (branch `main`).

> Note: there is a DIFFERENT, unrelated folder at `/home/vboxuser/apps/agent-testing/` (a minimal Node.js HTTP server + docker-compose used as a Docker sandbox test). It is **not** part of this repo. Do not confuse the two.

## Layout

```
agent-testing/
|-- docker-compose.yml          # Runs obsidian-mcp + agent-testing (ADK web UI) together
|-- Dockerfile                  # Python 3.14 + uv; runs the ADK web UI on :8000
|-- Dockerfile.obsidian-mcp     # Standalone obsidian-mcp HTTP server on :37842
|-- run-agent.ps1               # Windows helper: adk web --port 8000 text_summarizer
|-- .env / .env.example         # Model provider + observability + MCP config (keys)
|-- .dockerignore / .gitignore
`-- text_summarizer/            # The ADK agent package (Python)
    |-- __init__.py             # Entrypoint: loads .env, inits observability, exposes root_agent
    |-- agent.py                # The LlmAgent definition: model DI + instructions
    |-- obsidian_tools.py       # Optional MCP tools for an Obsidian vault
    |-- observability.py        # Langfuse tracing (no-op if unconfigured)
    |-- eval_exercise.py        # Manual eval-loop helper (MODIFIES agent.py)
    |-- auto_optimize.py        # LLM-driven auto-optimizer (MODIFIES agent.py)
    |-- check_free_models.py    # Lists OpenRouter :free models
    |-- optimization_history.json # History written by auto_optimize.py
    |-- pyproject.toml          # Deps: google-adk[mcp,extensions], langfuse, ...
    |-- uv.lock
    `-- tests/eval/
        |-- simple_test.test.json            # 1 eval case
        |-- summarizer_eval_set.evalset.json # 3 eval cases
        `-- test_config.json                 # Criteria + thresholds
```

## Agent architecture

Import chain (all through `text_summarizer/__init__.py`):

1. `load_dotenv()` — loads `text_summarizer/.env` (and the repo `.env` for docker).
2. `setup_observability()` — if `LANGFUSE_PUBLIC_KEY` is set, instruments the ADK runner with OpenInference + Langfuse. Otherwise a no-op.
3. `root_agent` — the exported ADK `LlmAgent`.

### Model selection — `agent.py`

- `MODEL_PROVIDER=gemini` (default) uses `gemini-3.5-flash-lite`.
- `MODEL_PROVIDER=openrouter` uses a `:free` model chosen by `MODEL_ALIAS` (gemma/qwen/nvidia) — see `OPENROUTER_MODELS`.
- Both paths return a `FallbackModel`: Gemini first, then free OpenRouter models on 429/quota/5xx errors.
- Models referenced are the free-tier aliases used by this project; don't "fix" the names to older released models.

The LlmAgent has: name `text_summarizer`, the model above, a description, bullet-point `instruction` rules, and `tools=build_obsidian_tools()`.

### Obsidian MCP tools — `obsidian_tools.py`

- Enabled if `OBSIDIAN_VAULT_PATH` (stdio via `uvx obsidian-mcp`) **or** `OBSIDIAN_MCP_URL` (streamable HTTP) is set. Set exactly one.
- Returns `[]` (no tools) when neither is set, or when `google-adk[mcp]` is missing.
- Tool filter: `vault_list`, `note_read`, `note_create`, `note_write`, `note_insert`, `search_text`, `search_metadata`.

### Docker topology — `docker-compose.yml`

- `obsidian-mcp` service (port 37842) runs inside the `agent-testing` network namespace (`network_mode: service:agent-testing`), so the agent reaches it via `OBSIDIAN_MCP_URL=http://127.0.0.1:37842/mcp`.
- `agent-testing` service: builds from `Dockerfile`, runs `adk web` on host port **8001 → 8000**, reads `.env`, points Langfuse at `http://host.docker.internal:3099` via `extra_hosts`.
- Vault mounted into the MCP container at `/vault`.

## Common commands

All `text_summarizer` paths assume `cd /home/vboxuser/Downloads/apps/agent-testing` (the repo root) unless noted.

```bash
# Run the agent (web UI): http://127.0.0.1:8000
uv run adk web text_summarizer            # or: adk web .

# Run the agent (CLI)
uv run adk run text_summarizer "Summarize: your long text here..."

# Save/run via the Windows helper
./run-agent.ps1

# Single eval case
uv run adk eval text_summarizer \
  text_summarizer/tests/eval/simple_test.test.json \
  --config_file_path text_summarizer/tests/eval/test_config.json \
  --print_detailed_results

# Full eval set (3 cases)
uv run adk eval text_summarizer \
  text_summarizer/tests/eval/summarizer_eval_set.evalset.json \
  --config_file_path text_summarizer/tests/eval/test_config.json \
  --print_detailed_results

# Manual eval-loop exercise (installs uv deps, uses .venv/bin/python)
cd text_summarizer
python eval_exercise.py

# Auto-optimize instructions (LLM-driven loop)
model_provider=gemini python text_summarizer/auto_optimize.py \
  --max-iterations 5 --patience 3
```

`Dockerfile` copies only `text_summarizer/`, then runs `uv sync --frozen` and serves the web UI.

## CRITICAL gotchas (read before editing)

1. **`eval_exercise.py` and `auto_optimize.py` MUTATE `agent.py`.** Both locate the main agent's instructions via regex `instruction="""..."""` and rewrite them in place (`set_instructions`). If you restructure `agent.py` (rename the agent, change quoting, split instructions into a variable), these scripts will silently fail or corrupt the file. Keep `instruction="""..."""` intact unless you also update those scripts.
2. **Eval scripts run `eval` with `cwd` = repo root.** Both invoke `python -m google.adk.cli eval ...` from the repo root, passing the agent name `text_summarizer` and paths relative to it. Run them from the repo root (or from `text_summarizer/` where the script itself manages `cwd`).
3. **`auto_optimize.py` REVERTS to the best-known instructions** when a rewrite doesn't improve the score, and appends a record to `optimization_history.json`. The final `agent.py` state reflects the best run, always.
4. **`response_match_score` is ROUGE-1 word overlap, not output quality.** Higher scores come from *matching the expected words*, not from being objectively better. The README documents experiments where "better" edits *lowered* the score. Don't use it to judge semantic quality.
5. **Config thresholds** (`test_config.json`): `tool_trajectory_avg_score` exact 1.0, `response_match_score` at 0.5.
6. **`.env` files hold real API keys** (Gemini, OpenRouter, Langfuse). Both `.env` are git-ignored — never commit them, never print their contents into logs/commits. Use `.env.example` as the reference for what's configurable.
7. **`.venv`, `.adk/`, `__pycache__`, `*.egg-info` are locally-generated state**, git-ignored, safe to delete and regenerate with `uv sync`.
8. **Dependencies are managed by `uv`** (`pyproject.toml` + `uv.lock`, `uv sync --frozen`). Prefer `uv` over pip when installing or running.

## Environment variables (see `.env.example`)

| Var | Purpose |
|---|---|
| `MODEL_PROVIDER` | `gemini` (default) or `openrouter` |
| `GEMINI_API_KEY` | Required for Gemini |
| `OPENROUTER_API_KEY` / `OPENROUTER_API_BASE` | Required for OpenRouter |
| `MODEL_ALIAS` | `gemma` / `qwen` / `nvidia` (OpenRouter free models) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | Optional Langfuse observability |
| `OBSIDIAN_VAULT_PATH` | Optional MCP over stdio (set EXACTLY ONE of the two) |
| `OBSIDIAN_MCP_URL` | Optional MCP over HTTP, e.g. `http://127.0.0.1:37842/mcp` |

## Evaluating your changes

The intended workflow is: **edit instructions in `agent.py` → run eval → compare score → keep or revert**. Make one change at a time so you can attribute score deltas. Always re-check `git status`/`git diff` after running the eval scripts, since they edit `agent.py` and `optimization_history.json` for you.