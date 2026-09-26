# Architecture

## Project Structure

```
agent-testing/
|-- docker-compose.yml          # Runs obsidian-mcp + agent-testing (ADK web UI) together
|-- Dockerfile                  # Python 3.14 + uv; runs the ADK web UI on :8000
|-- Dockerfile.obsidian-mcp     # Standalone obsidian-mcp HTTP server on :37842
|-- resolve-vault.sh            # Startup vault resolver for the MCP server
|-- run-agent.ps1               # Windows helper: adk web --port 8000 text_summarizer
|-- .env / .env.example         # Model provider + observability + MCP config (keys)
|-- AGENTS.md                   # Agent-facing project instructions
|-- README.md                   # Quick start + doc index
`-- docs/                       # Detailed documentation
    |-- EVALUATION.md           # Eval loop, criteria, hands-on, auto-optimize
    |-- TESTING.md              # Web UI example prompts
    |-- MODELS.md               # Model switching
    |-- INTEGRATIONS.md         # Gmail + Obsidian setup
    `-- ARCHITECTURE.md         # This file
`-- text_summarizer/            # The ADK agent package (Python)
    |-- __init__.py             # Entrypoint: loads .env, inits observability, exposes root_agent
    |-- agent.py                # The LlmAgent definition: model DI + instructions + scoring callbacks
    |-- eval_scoring.py         # ROUGE-1 (response_match_score) matching eval-set golden answers
    |-- obsidian_tools.py       # Optional MCP tools for an Obsidian vault
    |-- gmail_tools.py          # Optional MCP tools for read-only Gmail access
    |-- gmail_mcp_server.py     # Stdio Gmail MCP server spawned by gmail_tools
    |-- gmail_oauth.py          # One-time helper that mints GOOGLE_REFRESH_TOKEN
    |-- observability.py        # Langfuse tracing (no-op if unconfigured)
    |-- sources.py              # Deterministic **Sources** rendering: vault name + served model
    |-- vaults.py               # Which vault is active: discovery, selection, import
    |-- second_brain.py         # Vault writes: notes, chat log, index, cache lookup
    |-- eval_exercise.py        # Manual eval-loop helper (MODIFIES agent.py)
    |-- auto_optimize.py        # LLM-driven auto-optimizer (MODIFIES agent.py)
    |-- check_free_models.py    # Lists OpenRouter :free models
    |-- pyproject.toml          # Deps: google-adk[mcp,extensions,eval], langfuse, ...
    |-- uv.lock
    `-- tests/
        |-- conftest.py                    # Blanks LANGFUSE_PUBLIC_KEY so imports don't hit the network
        |-- test_sources.py                # **Sources** renderer unit tests
        |-- test_agent_callback.py         # after_agent_callback unit tests
        |-- test_prompt_name_span.py       # prompt-name span tagging tests
        |-- test_adk_wiring.py             # In-process ADK run with a stub Llm (no API cost)
        |-- test_vaults.py                # Vault selection + resolve-vault.sh parity
        |-- test_run_script.py            # run.sh: discovery, menu, import, .env persistence
        `-- eval/
            |-- simple_test.test.json            # 1 eval case
            |-- summarizer_eval_set.evalset.json # 4 eval cases
            `-- test_config.json                 # Criteria + thresholds
```

## Agent Flow

Import chain (all through `text_summarizer/__init__.py`):

1. `load_dotenv()` — loads `text_summarizer/.env` (and the repo `.env` for docker).
2. `setup_observability()` — if `LANGFUSE_PUBLIC_KEY` is set, instruments the ADK
   runner with OpenInference + Langfuse. Otherwise a no-op.
3. `root_agent` — the exported ADK `LlmAgent`.

The agent's tools = `build_obsidian_tools()` + `build_gmail_tools()` (each returns
`[]` when unconfigured) plus two `FunctionTool`s:
`save_summary_to_second_brain` and `log_conversation`.

Key callbacks:

- `before_model_callback=cache_hit_before_model` — replays a stored summary from
  the vault (zero LLM calls) when the exact text was summarized before.
- `after_model_callback=tag_current_span` — reports the prompt name on the
  generation span.
- `after_agent_callback=report_scores_after_agent` — pushes `quality.*` and
  `response_match_score` scores to Langfuse **and** rewrites the `**Sources**`
  block with the real vault name + served model.

## `**Sources**` Provenance

Every fresh, uncached response ends with a block like:

```
**Sources**
[obsidian][ck][gemini-3.5-flash-lite][[2026-09-25 - dogs-summary]]: why it is relevant
```

- The model is never asked for its own name. Rule 7 emits sentinel tokens
  `@@ADK_VAULT@@` / `@@ADK_MODEL@@` that `sources.render_sources` replaces in
  code after the run. The heading and real identifiers are added by the callback.
- The vault name resolves at runtime; the served model comes from
  `Event.model_version` after the run.

## Observability (Langfuse)

- `setup_observability()` no-ops without `LANGFUSE_PUBLIC_KEY`.
- Every turn records a `cache-hit` / `cache-miss` / `cache-disabled` tag plus
  `cache.outcome` / `cache.lookup_ms` metadata and scores, so cached vs live
  traces can be told apart.
- `quality.bullet_count`, `quality.source_overlap`, `quality.fidelity` are
  deterministic heuristics computed per call.
- `CACHE_ENABLED=false` bypasses the vault cache (required for `adk eval`).

## Vault Selection

`obsidian-mcp` serves exactly one vault, so `text_summarizer/vaults.py` decides
which one and refuses to guess. Resolution order: `OBSIDIAN_VAULT_NAME` → the
mounted directory is itself a vault → its single child → auto-create
`agent-vault` when empty → error listing the candidates. `run.sh` turns the same
discovery into a menu (adding the vaults from `~/.config/obsidian/obsidian.json`)
and persists the choice to `.env`; `resolve-vault.sh` mirrors the rules for the
MCP container. See [INTEGRATIONS.md](INTEGRATIONS.md#using-a-vault-you-already-have).

## Vault Cache

Identical prompts are answered from the vault with no model calls. The key is
the SHA-256 of the **user's message**, computed in code on both sides of the
round-trip — never supplied by the model, which paraphrases and would key the
cache on its own output. Only the first model call of a turn consults it, so the
note the agent writes mid-turn cannot be replayed over its own answer. See
[AGENTS.md](https://github.com/carloskafka/agent-testing/blob/main/AGENTS.md#the-vault-cache--one-key-computed-in-code).

## Docker Topology

`docker-compose.yml` runs two services:

- `obsidian-mcp` — builds from `Dockerfile.obsidian-mcp`; runs the obsidian-mcp
  server on port 37842 inside the `agent-testing` network namespace
  (`network_mode: service:agent-testing`), so the agent reaches it via
  `OBSIDIAN_MCP_URL=http://127.0.0.1:37842/mcp`.
- `agent-testing` — builds from `Dockerfile`, runs `adk web` on host port
  **8001 → 8000**, reads `.env`, points Langfuse at
  `http://host.docker.internal:3099` via `extra_hosts`.

```bash
./run.sh                          # one-command bootstrap: build & start
docker compose up -d --build      # build & start (same as run.sh internals)
docker compose logs -f            # follow logs
docker compose down               # stop
```

The vault's **parent** is mounted into both containers at `/vaults` — not the
vault itself — so the real directory name survives. The host path defaults to
`./vaults` and is overridable via `OBSIDIAN_VAULT_PARENT_HOST` in `.env`.
`resolve-vault.sh` (the MCP container's entrypoint) and
`second_brain.resolve_vault_root()` agree on the active vault, fail loudly on an
ambiguous mount (2+ candidate vaults), and **auto-create** the `agent-vault`
folder (with its `Second Brain/` dir) when the mount is empty — so a fresh
clone-and-run works with zero vault setup.

Gmail works inside Docker too: the agent container spawns `gmail_mcp_server.py`
with its own Python, so the only requirement is the rebuilt image
(`docker compose up -d --build` after pulling the Gmail changes).