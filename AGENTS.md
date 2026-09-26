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
|-- run.sh                      # One-command bootstrap: .env scaffold + docker compose up --build
|-- Dockerfile                  # Python 3.14 + uv; runs the ADK web UI on :8000
|-- Dockerfile.obsidian-mcp     # Standalone obsidian-mcp HTTP server on :37842
|-- resolve-vault.sh            # Startup vault resolver for that server (mirrors second_brain.resolve_vault_root)
|-- run-agent.ps1               # Windows helper: adk web --port 8000 text_summarizer
|-- patch-adk-devui-mobile.py   # Injects 100dvh/safe-area CSS into the bundled ADK dev UI (see below)
|-- .env / .env.example         # Model provider + observability + MCP config (keys)
|-- .dockerignore / .gitignore
|-- docs/                       # Split-out documentation (EVALUATION, TESTING, MODELS,
|                               #   INTEGRATIONS, ARCHITECTURE, getting-started.md) — README links here
`-- text_summarizer/            # The ADK agent package (Python)
    |-- __init__.py             # Entrypoint: loads .env, inits observability, exposes root_agent
    |-- agent.py                # The LlmAgent definition: model DI + instructions + Langfuse scoring callback
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
    # optimization_history.json is written by auto_optimize.py but is NOT tracked:
    # it is local run state, like .adk/ (see gotcha 8).
    `-- tests/
        |-- conftest.py                    # Blanks LANGFUSE_PUBLIC_KEY so imports don't do a network auth check
        |-- test_sources.py                # **Sources** renderer unit tests
        |-- test_agent_callback.py         # before/after agent callbacks (rewrite, cache hit/miss, key)
        |-- test_adk_wiring.py             # In-process ADK run with a stub Llm (no API cost)
        |-- test_obsidian_tool_schema.py   # MCP JSON-Schema sanitiser (the fallback-path 400)
        |-- test_prompt_name_span.py       # after_model_callback: Langfuse prompt-name tagging
        |-- test_trace_identity.py         # before_agent_callback: trace name + userId/sessionId
        |-- test_vaults.py                 # Vault selection + resolve-vault.sh parity
        |-- test_run_script.py             # run.sh: discovery, menu, import, .env persistence
        |-- test_adk_devui_patch.py       # The mobile patch + its guards against ADK drift
        `-- eval/
            |-- simple_test.test.json            # 1 eval case
            |-- summarizer_eval_set.evalset.json # 4 eval cases
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

The LlmAgent has: name `text_summarizer`, the model above, a description, bullet-point `instruction` rules, and `tools` = `build_obsidian_tools()` + `build_gmail_tools()` — each returns `[]` when unconfigured, so the tool set is additive and never breaks without the relevant env vars. It also registers `after_agent_callback=report_scores_after_agent`, which does two independent jobs: it pushes deterministic `quality.*` scores AND — when the incoming prompt matches an eval-set golden answer — a `response_match_score` (protocol-identical ROUGE-1) to Langfuse per call; and it returns rewritten content so the response's `**Sources**` block carries the real vault name and the real served model (see below). `before_agent_callback=tag_trace_identity` names the Langfuse trace and attaches `userId`/`sessionId` (see "Trace identity").

### `**Sources**` provenance — `sources.py` + `agent.py`

Every fresh, uncached response ends with a block like:

```
**Sources**
[obsidian][ck][gemini-3.5-flash-lite][[2026-09-25 - dogs-summary]]: why it is relevant
```

- **The model is never asked for its own name.** Rule 7 makes the model emit the *shape* of a source line with two sentinel tokens, `@@ADK_VAULT@@` and `@@ADK_MODEL@@`; `sources.render_sources` replaces them in code after the run. The sentinels are collision-proof by construction (doubled `@` + SCREAMING_SNAKE is not markdown-significant and never appears in prose), and any stray occurrence is scrubbed.
- **One block, always.** The renderer discards whatever heading the model wrote (`**Sources**`, `## Sources`, none at all) and re-emits exactly one canonical block in the same position, so a model that ignores the format still cannot produce duplicates. Legacy `[vault] [[Note]]` lines from pre-existing output are adopted and normalized, not dropped.
- **Served model — `Event.model_version`.** `Event` subclasses `LlmResponse`, and `_finalize_model_response_event` (`google/adk/flows/llm_flows/base_llm_flow.py`) merges every non-`None` `LlmResponse` field onto the model-response event. `model_version` is set by `LlmResponse.create` from `generate_content_response.model_version` (Gemini, `google/adk/models/google_llm.py`) and from `response.model` (`google/adk/models/lite_llm.py`). `FallbackModel` yields the sub-model's response unmodified, so the value is whichever backend actually answered. Read it via `sources.served_model_from_events`, which walks the session **backwards** and takes the last non-partial model event that has text *or* a function call.
- **Unresolvable provenance renders as the literal `unknown`** — never a guess, never back-filled from the currently-serving model. A cached replay carries no `model_version`, so if the callback ever ran on one it would say `unknown`; in practice it returns `None` first (below).
- **Notes record their own provenance.** `save_summary_to_second_brain` writes `generated_by_model` and `generated_in_vault` into the note frontmatter, read from the live invocation via the framework-injected `tool_context` (a parameter named `tool_context` is supplied by ADK and hidden from the model's schema — do not "fix" its absence from the JSON schema). Notes written **before** these fields existed have no recorded model; that is surfaced as `unknown` and never invented. Such a field is omitted entirely when unresolvable.
- **Cache hits are untouched.** `report_scores_after_agent` returns `None` before any rendering when `state["vault_cache_hit"]` is true, so a stored note replays verbatim (including the old `## From the vault` shape in notes saved before this change). Scoring is likewise skipped on a hit, as before.

#### Which vault is active — `vaults.py`

`obsidian-mcp` takes **exactly one vault path** as a CLI argument, so a deployment has exactly one *active* vault. The question is therefore not "which of many" but "which one, and how does the user say so" — and the answer has to be explicit the moment there is more than one candidate, because the cost of a wrong choice is a user's notes landing in the wrong vault, silently.

All of it lives in `text_summarizer/vaults.py`; `second_brain.resolve_vault_root()` is a thin wrapper that supplies the configured parent plus the `OBSIDIAN_VAULT_NAME` choice.

| # | Situation | Result |
|---|---|---|
| 1 | `OBSIDIAN_VAULT_NAME` is set | That vault, whatever else is in the parent. A name matching no child is an **error**, not a silent fallback, so a typo is visible on the first boot. |
| 2 | The parent is itself a vault (holds `Second Brain`) | Used as-is. Covers pointing `SECOND_BRAIN_VAULT` straight at a vault. |
| 3 | Exactly one child | That child — the common case, zero configuration. |
| 4 | No children | `agent-vault`, created lazily on first write (so a read-only turn against a fresh mount creates nothing). |
| 5 | Several children, no name | **Refuses**, printing the candidates and the variable to set. |

`vaults.obsidian_vaults()` additionally reads `~/.config/obsidian/obsidian.json` (honouring `$OBSIDIAN_CONFIG_DIR`), which is where Obsidian itself lists every vault the user has. That is what makes "I already have a lot of vaults" work **without the user rearranging anything** — the answer is already on disk. It is a cache, so entries whose path no longer exists are dropped, and a missing or malformed file yields `[]`: discovery must never be why the agent fails to start.

`run.sh` turns the same discovery into a numbered menu (candidates from the mount *and* from Obsidian's config, de-duplicated by real path) and persists the answer to `.env`, so later runs and a bare `docker compose up` are non-interactive. Options beyond the listed vaults are "create a new empty vault" and "import a vault by path". Importing is usually *not* what you want: pointing `OBSIDIAN_VAULT_PARENT_HOST` at the vault's own parent and setting `OBSIDIAN_VAULT_NAME` achieves the same thing with nothing copied, and that is what `run.sh` does automatically when you pick a vault from outside the current mount.

**Import copies, and that is not a detail.** `vaults.import_vault` defaults to `mode="copy"`. A symlink would be correct on the host and *broken inside the container*: Docker bind-mounts the parent, and a link whose target lies outside the mounted tree dangles, because that path does not exist in the container's filesystem. Verified live, not assumed — a `ck -> /home/.../ck-vault/ck` link inside the mount lists fine on the host and gives `No such file or directory` at `/vaults/ck/`, which stops `obsidian-mcp` at startup. There is no in-tree case that would prefer a link either: a source already inside the parent resolves to the very path the import would create, so it is refused as an overwrite. `mode="link"` stays available for a run outside Docker, where there is no mount to dangle in; the trade-off is that a copy keeps the agent's notes entirely apart instead of sharing one source of truth. Either way an existing entry is never overwritten — a silent replacement is the worst failure mode available here.

#### Vault name resolution

The vault name is resolved **at runtime, with nothing hardcoded** — no `VAULT_NAME` default in `docker-compose.yml`, no literal in Python.

Mounting the vault *itself* (`.../ck-vault/ck:/vault`) flattens its name to `vault`: Docker presents the mount point, so the host directory name is genuinely unrecoverable from inside the container. Verified, not assumed — `basename` gives `vault`, the obsidian-mcp `vault_info` tool also returns `vault` (it derives from the launch path), `vault_list` returns top-level entries with no name at all, and nothing inside the vault records its own name.

The fix is to mount the vault's **parent** instead, so the real name stays in the path:

```yaml
volumes:
  - ${OBSIDIAN_VAULT_PARENT_HOST:-./vaults}:/vaults   # the PARENT (host path override in .env)
environment:
  SECOND_BRAIN_VAULT: /vaults                        # no vault name anywhere
  OBSIDIAN_VAULT_NAME: ${OBSIDIAN_VAULT_NAME:-}      # only needed with several vaults
```

`Dockerfile.obsidian-mcp` does the same in its `CMD` so both services agree on the vault. Rename the vault folder on the host and everything follows.

`sources.resolve_vault_name` then resolves in this order:

| # | Source | When it applies |
|---|---|---|
| 1 | `basename(resolved vault root)` | Authoritative. Callers pass the root **after** `resolve_vault_root()`, and the path must exist — a missing mount renders `unknown` rather than a plausible-looking wrong name. |
| 2 | `VAULT_NAME` env var | Only when the vault itself is mounted directly, so its name is truly not in the path. |
| 3 | `vault_info` MCP tool result | Opportunistic: only when a turn actually called it. |
| 4 | `"unknown"` | Nothing else resolved. |

`VaultIdentity.source` records which of these fired, for debugging.

**The MCP server resolves the same vault independently.** `resolve-vault.sh` runs as the container's entrypoint and mirrors `vaults.select_vault` in POSIX shell: `VAULT_NAME` (fed from the same `OBSIDIAN_VAULT_NAME` by compose) wins outright, a `VAULT_PARENT` that is itself a vault is used as-is, an empty parent auto-creates `agent-vault` with `Second Brain`, exactly one child is used, and anything else **exits non-zero after printing the candidates** instead of guessing. The deliberate asymmetry: the Python path degrades to the unresolved root (so a read-only turn still returns notes), whereas the server cannot meaningfully run against a wrong path and must fail at startup — an agent pointed at a different vault than its MCP server is worse than an agent that does not start. The two implementations are covered by the same pytest module so they cannot drift.

`OBSIDIAN_MCP_PORT` overrides the listen port; `VAULT_PARENT` overrides the in-container mount point; the host side of the mount is `${OBSIDIAN_VAULT_PARENT_HOST:-./vaults}` in `docker-compose.yml`. `tests/test_vaults.py` drives the real script with a stub `obsidian-mcp` on `PATH`, so no daemon is needed:

```bash
printf '#!/bin/sh\necho "ARGS: $*"\n' > /tmp/stub/obsidian-mcp && chmod +x /tmp/stub/obsidian-mcp
PATH=/tmp/stub:$PATH VAULT_PARENT=/some/parent sh ./resolve-vault.sh
```

### The vault cache — one key, computed in code

The cache exists so a repeated request is answered from the vault with **zero model calls**. It only works if the key written and the key looked up are the same string, and getting that wrong is silent: the lookup simply never matches, and every turn re-pays in full.

**Both sides now use the user's own message, and neither asks the model.** `agent.cache_hit_before_model` fingerprints the prompt from `llm_request.contents`; `second_brain.cache_key_text` reads the same turn's text from `InvocationContext.user_content` (via the injected `tool_context`). Three things make this the only workable design:

- **The model paraphrases.** It used to be given a `source_text` argument to fingerprint, and produced something different on every run — three identical questions wrote three different keys. On a tool-driven turn it tended to pass *its own summary*, so the recorded key was the output, not the input.
- **The consequence is a whole turn, not a lookup.** Measured on the Gmail flow: `call_llm` 2.7s + `gmail_search` 1.8s + `call_llm` 7.6s + `call_llm` 1.2s + `call_llm` 1.1s ≈ 12s per repeat, while `save_summary_to_second_brain` took **2 ms** and `cache.lookup_ms` 0.5ms. The dev-ui event gap that looks like time inside the save tool is really the *next* generation. Asking the model to echo the source back as a tool argument is also what inflated that 7.6s call, so the argument is gone from both the signature and instruction rule 8.
- **`user_content`, not the request contents.** ADK models a tool result as a `Content` with role `"user"`, so on any follow-up model call the *last* user-role part of `llm_request.contents` is the tool's JSON payload, not the prompt. Keying on it turns every mid-turn lookup into a miss on a key nothing will ever match. Both sides read `user_content` instead — `CallbackContext.user_content` on the lookup, `ToolContext.user_content` on the write.
- **`user_content`, not the session events.** `session.events` is unreliable twice over: in a resumed session it can be a snapshot taken *before* the current user message was appended, and under `adk web`'s database-backed session service it is **not populated at all**. The event scan is only a last-resort fallback.

**Only the first model call of a turn consults the cache** (`agent._first_model_call_of_invocation`). `before_model_callback` fires before *every* model call, and the agent persists its note from inside the turn, so by the follow-up call the fingerprint for the current prompt is already on disk — replaying it there would return the note instead of the answer and truncate the turn. Eligibility is tracked by `invocation_id` in the session state, **not** by inspecting `session.events`: an earlier positional version of this check passed every unit test and silently never fired in the container, because an empty event list reads as "first call" every time. Both live-only failures were found by running a real turn under `adk web`; `test_agent_callback.py` now pins them against the shape the live runner produces.

A hit therefore also skips `log_conversation`: the replay happens before any tool runs, so the exchange is not recorded twice. That is what keeps `CACHE_ENABLED=false` meaningful for `adk eval`.

**Pre-existing notes are not reachable by the cache and that is correct.** Notes written before this change carry a fingerprint of the model's paraphrase, which the lookup never computed, so they were already unreachable — they simply looked like misses. They are never rewritten or back-filled; re-asking the question writes a fresh, correctly keyed note.

`test_adk_wiring.py` pins all of it: the write key equals the lookup key, a second identical turn costs zero model calls, the follow-up call in a turn is not short-circuited, and a later turn in the same session is not mistakenly treated as a cache opportunity.

### Langfuse scoring — `agent.py` + `eval_scoring.py`

- `report_scores_after_agent` runs after every agent invocation and posts numeric scores to the current OTel trace: `quality.bullet_count`, `quality.source_overlap`, `quality.fidelity` (deterministic heuristics, zero extra LLM calls). Scores attach to the **trace**, not the generation — in a multi-turn session each turn overwrites the previous turn's score of the same name. Scores are **skipped entirely on a cache hit** (the response was replayed, not generated).
- The same callback also rewrites the response's `**Sources**` block (see the section above). The two jobs are independent: rendering runs first and is wrapped so a failure can never suppress the scores, and a turn with nothing to rewrite returns `None` so the callback stays a pure no-op.
- `quality.*` is computed on `sources.summary_only(response)`, i.e. with the Sources block removed — otherwise source lines would be counted as summary bullets.
- **Vault cache observability — `cache_hit_before_model` + `observability.report_cache_outcome`.** Every turn emits a trace tag `cache-hit` / `cache-miss` / `cache-disabled`, trace metadata `cache.outcome` + `cache.lookup_ms`, and two scores: `cache.hit` (1.0/0.0) and `cache.lookup_ms`. This is what makes the two cohorts separable — filter traces by the `cache-hit` tag to compare latency and cost against `cache-miss`. `CACHE_ENABLED=false` bypasses the cache and tags the trace `cache-disabled`; **keep it off for `adk eval`** (see gotcha 10).
- `eval_scoring.py` loads `tests/eval/*.test.json` and `*.evalset.json`, matches the live user prompt to a case's golden answer, and computes `response_match_score` using ADK's own `google.adk.evaluation.final_response_match_v1._calculate_rouge_1_scores` — the exact same scored value the `adk eval` loop reports. Requires the `google-adk[eval]` extra (provides `rouge_score`); degrades to no-op if it's missing.
- Note the callback attaches scores to whatever trace is active; live runs get their own trace.

### Trace identity — `observability.tag_trace_identity`

`tag_trace_identity` is wired as the agent's `before_agent_callback` and sets three OTel attributes on the currently open span: `langfuse.trace.name` (from `LANGFUSE_TRACE_NAME`, default `text_summarizer`), `user.id` and `session.id`. Without it every trace lists as a blank row in the UI and the "User consumption" widget has nothing to group by.

Three things that are easy to get wrong, all pinned by `test_trace_identity.py`:

- **Only `langfuse.trace.*` is namespaced.** `user.id` and `session.id` are plain OTel semantic conventions that Langfuse's ingestion maps onto the trace. Writing `langfuse.user.id` would be accepted by OTel and then silently dropped by Langfuse — no error, just a blank column.
- **It must run while the span is open.** OTel discards attributes set on an ended span without complaint, which is why this is a `before_agent_callback` and not a trailing one. Same reason `after_model_callback` (not `before_model_callback`) carries the prompt name.
- **`before_agent_callback` is the earliest hook that is still inside the trace.** ADK opens the runner's `invocation` span before any agent hook runs, so that span never carries these attributes. It does not matter: Langfuse folds trace-level attributes from *any* span in the trace, and confirmed live — `user_id`/`session_id` land on every observation row of the trace.

Langfuse's v4 `events_only` mode is worth knowing when reading the database directly: **the `traces` and `observations` ClickHouse tables stay empty and that is not a bug.** Data lands in `events_core`/`events_full` and the UI reads those. Debug a trace with:

```bash
docker exec langfuse-clickhouse-1 clickhouse-client -q "
  SELECT type, name, trace_name, user_id,
         dateDiff('millisecond', start_time, end_time) AS ms
  FROM events_core WHERE start_time > now() - interval 5 minute
  ORDER BY start_time FORMAT TSV"
```

A healthy turn is one `CHAIN invocation` → one `AGENT agent_run` → alternating `GENERATION call_llm` and `TOOL execute_tool` rows. A turn with a single `call_llm` and nothing after it is a turn that died on the first model call.

### Obsidian MCP tools — `obsidian_tools.py`

- Enabled if `OBSIDIAN_VAULT_PATH` (stdio via `uvx obsidian-mcp`) **or** `OBSIDIAN_MCP_URL` (streamable HTTP) is set. Set exactly one.
- Returns `[]` (no tools) when neither is set, or when `google-adk[mcp]` is missing.
- Tool filter: `vault_info`, `vault_list`, `note_read`, `note_create`, `note_write`, `note_insert`, `search_text`, `search_metadata`. `vault_info` is exposed so the vault name can be discovered at runtime (see "Vault name resolution"); the agent is never required to call it.

#### Tool-schema sanitisation — the fallback-path 400

`sanitize_tool_schema` normalises every MCP tool's JSON Schema before the model sees it, and the toolset is a subclass (`_sanitizing_toolset_class`) that applies it. It is not defensive decoration; without it **the fallback path cannot run at all**.

`obsidian-mcp` declares `search_metadata.value` as a union — `type: ["array","boolean","null","number","object","string"]` — with no `items` on the array member. Gemini is reached three ways and only the third cares:

| Path | How the schema travels | Array union without `items` |
|---|---|---|
| Primary `GeminiModel` | `parameters_json_schema` verbatim | accepted |
| Fallback → LiteLLM → OpenRouter → **non-Google** upstream | OpenAI function params | accepted |
| Fallback → LiteLLM → OpenRouter → **Google AI Studio** upstream | lowered to a native `Schema` message, so the union becomes `any_of` | **`INVALID_ARGUMENT` 400** |

The error reads `GenerateContentRequest.tools[0].function_declarations[6].parameters.properties[value].any_of[0].items: missing field` and surfaces as `litellm.BadRequestError`. Declaration index 6 is `search_metadata` (the two `FunctionTool`s occupy 0–1 and the MCP tools are sorted by name). Which upstream serves a given free model is not under our control, so the same model may pass or fail on successive calls.

**Why this was easy to miss:** it only fires on the *fallback* path, i.e. exactly when the primary model already failed. A transient Gemini 429 therefore turned into a dead turn with a schema error that has nothing to do with the actual cause. When a turn dies on a 400 from `litellm`, check the tool schemas before suspecting the prompt.

The transform is two rules, both no-ops on an already-valid schema:
1. **`$ref` inlined, `$schema`/`$defs` dropped.** Gemini answers `reference to undefined schema` for a dangling ref, and `search_text.fields` points at `#/$defs/SearchField`.
2. **`items` added to any array-typed subschema that lacks one.** The original left element types unconstrained; `{"type": "string"}` is the schema that survives every backend above. Verified against all three paths.

Only the schema sent to the model changes — arguments are still forwarded to the MCP server untouched, so no tool's behaviour changes. The rewrite happens in place on the raw tool, which ADK caches per connection, and is idempotent. `test_obsidian_tool_schema.py` pins all of it, including the depth cap that stops a self-referential `$defs` from hanging the process.

**Both** MCP toolsets go through it — `sanitizing_mcp_toolset_class()` is shared by `obsidian_tools.py` and `gmail_tools.py`, so a third server cannot reintroduce the bug by omission. The in-repo Gmail server declares no union types today, so it does not strictly need the rewrite; it gets it anyway because the failure is silent until a fallback run happens to route to a Google upstream.

### Gmail MCP tools — `gmail_tools.py`

- Read-only Gmail access via a stdio MCP server defined in this repo (`gmail_mcp_server.py`), spawnable from any interpreter that has `google-adk[mcp]` + `google-api-python-client` installed.
- Enabled only when all three of `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN` are set; otherwise returns `[]` (no tools).
- Tool filter: `gmail_search`, `gmail_get_latest_messages`, `gmail_read`, `gmail_get_thread`.
- Credentials: the client id/secret come from a Desktop-app OAuth client in the Google Cloud Console; the refresh token is minted once via `python -m text_summarizer.gmail_oauth` (opens a browser, read-only scope `gmail.readonly`).

### The ADK dev UI on a phone — `patch-adk-devui-mobile.py`

`adk web` serves a **prebuilt** Angular app from `google/adk/cli/browser/index.html`
inside the installed wheel. It is a desktop developer tool with no mobile support,
and the failure is specific: **messages render fine, the chat composer does not.**

Measured across both of its style blocks and all 100 JS bundles:

| | occurrences |
|---|---|
| `dvh` / `svh` | **0** |
| `safe-area-inset-*` | **0** |
| `viewport-fit=cover` | absent |

Three layers pin their height to the legacy `100vh`, and two of them hide overflow:

```css
body                     { height:100vh; overflow:hidden }
[_nghost-*]              { height:100vh; overflow:hidden }   /* app shell */
.builder-mode-container  { height:100vh }
```

On a phone `100vh` is the **largest** viewport — the height you get with the URL bar
retracted — not the visible one, so the shell is 15–20% taller than the screen. The
composer is the last row of that fixed-height flex column, so it lands below the
fold, and because `body` and the shell are both `overflow:hidden` the overflow is
**unreachable**: no scrolling, no pinch-zoom. The on-screen keyboard then halves the
visible height and the composer is gone entirely.

It is not a width problem, which is why the symptom is so specific: the chat bubbles
are `max-width:800px` in a centred flex column and shrink correctly. The breakage is
purely vertical.

**The patch** (`RUN` step in `Dockerfile`, after `uv sync` so it cannot be clobbered)
injects one stylesheet before `</head>` rather than rewriting vendor rules, so it is
verifiable and removable. Everything is wrapped in `@supports (height: 100dvh)`, so a
browser without it keeps vendor behaviour untouched:

- `100dvh` heights on `html`, `body` and `app-root` — this is the actual fix, and it
  is what makes the URL bar *and* the keyboard behave
- `padding-bottom: calc(20px + env(safe-area-inset-bottom))` on `.chat-input-container`
- `.assistant-panel` capped to `min(400px, 100vw)` (it is a hard 400px, which
  overflows a 390px phone)

It also edits the viewport meta, which CSS cannot reach: `viewport-fit=cover`
(without it `env(safe-area-inset-*)` is always `0`, so the CSS above would be inert
on a notched device) and `interactive-widget=resizes-content` (Chromium overlays the
keyboard by default, covering the composer; Safari has no equivalent and relies on
`dvh`).

**Why bare class names, not the app's own selectors.** Every rule in the bundle is
qualified with `[_ngcontent-<hash>]`, and that hash is rebuilt per ADK release, so
there is no stable selector to write. The overrides match `.chat-input-container` and
`.assistant-panel` by class and use `!important`, which beats a higher-specificity
vendor rule. `app-root` is a tag name and is stable.

**The known weakness, stated plainly:** because the overrides key on class names, an
ADK release that renames them would make the fix silently stop applying — no error,
just the old broken composer. The script therefore asserts the strings it depends on
and **exits non-zero if they are gone, failing the Docker build** rather than shipping
a patch that does nothing. `tests/test_adk_devui_patch.py` additionally asserts those
assumptions against the *installed* package, so the next `uv sync` surfaces a
dependency upgrade in seconds instead of on someone's phone. Idempotent: the marker
comment makes a re-run a no-op, and it is verified byte-for-byte on the real file.

**Not verified:** no mobile browser was available here, so the visual result is
unconfirmed. What *is* confirmed is that the served page carries the patch (marker
present, 6 × `100dvh`, correct meta, vendor rule intact, injected exactly once before
`</head>`). Check it on a real device before calling it fixed.

### Docker topology — `docker-compose.yml`

- `obsidian-mcp` service (port 37842) runs inside the `agent-testing` network namespace (`network_mode: service:agent-testing`), so the agent reaches it via `OBSIDIAN_MCP_URL=http://127.0.0.1:37842/mcp`.
- `agent-testing` service: builds from `Dockerfile`, runs `adk web` on host port **8001 → 8000**, reads `.env`, points Langfuse at `http://host.docker.internal:3099` via `extra_hosts`.
- The vault's **parent** is mounted into both containers at `/vaults` (not the vault itself), so the real vault directory name survives in the path. The host path defaults to `./vaults` and is overridable via `OBSIDIAN_VAULT_PARENT_HOST` in `.env`; an empty mount auto-creates the `agent-vault` folder on first boot, so a fresh clone-and-run needs zero vault setup. `SECOND_BRAIN_VAULT: /vaults` points at the same place. The MCP container resolves the actual vault at startup via `resolve-vault.sh`, which mirrors `second_brain.resolve_vault_root()`; the two must never disagree about which vault is in use, so the script **fails loudly** on an ambiguous mount (>1 child directory) rather than picking one, and the agent surfaces the same ambiguity as `unknown` provenance instead of a wrong name.

## Common commands

All `text_summarizer` paths assume `cd /home/vboxuser/Downloads/apps/agent-testing` (the repo root) unless noted.

```bash
# One-command bootstrap (Docker): scaffolds .env.vaults/, builds + starts both containers
./run.sh

# Run the agent (web UI): http://127.0.0.1:8000
uv run adk web text_summarizer            # or: adk web .

# Run the agent (CLI)
uv run adk run text_summarizer "Summarize: your long text here..."

# Save/run via the Windows helper
./run-agent.ps1

# Single eval case
CACHE_ENABLED=false uv run adk eval text_summarizer \
  text_summarizer/tests/eval/simple_test.test.json \
  --config_file_path text_summarizer/tests/eval/test_config.json \
  --print_detailed_results

# Full eval set (4 cases: 3 prose-only + 1 with a **Sources** block)
CACHE_ENABLED=false uv run adk eval text_summarizer \
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

### Unit tests (no API cost, no vault writes)

`pytest` is not a declared dependency, so it is installed out-of-tree and put on
`PYTHONPATH` — this keeps `.venv/` and `uv.lock` untouched:

```bash
mkdir -p /tmp/agent-testing-testlibs
~/.local/bin/uv pip install \
  --python ~/Downloads/apps/agent-testing/text_summarizer/.venv/bin/python \
  --target /tmp/agent-testing-testlibs pytest

cd /home/vboxuser/Downloads/apps/agent-testing
PYTHONPATH=/tmp/agent-testing-testlibs \
  text_summarizer/.venv/bin/python -m pytest text_summarizer/tests -q
```

- `tests/conftest.py` blanks `LANGFUSE_PUBLIC_KEY` before any import, because importing the package runs `setup_observability()`, which otherwise does a network auth check against an unreachable host.
- `test_adk_wiring.py` runs a real `LlmAgent` through a real `InMemoryRunner` with a stub `BaseLlm`. That is how the three ADK facts the provenance feature depends on are verified without spending quota. **If you change how the model or tool context is read, keep these tests green — they are the only automated guard on that wiring.**

There is no linter or type-checker configured in this repo (no `ruff`/`mypy`/`pyright` config, no `Makefile`, no CI). The only automated gate is the pytest run above.

`Dockerfile` copies only `text_summarizer/`, then runs `uv sync --frozen` and serves the web UI.

## CRITICAL gotchas (read before editing)

1. **`eval_exercise.py` and `auto_optimize.py` MUTATE `agent.py`.** Both locate the main agent's instructions via regex `instruction="""..."""` and rewrite them in place (`set_instructions`). If you restructure `agent.py` (rename the agent, change quoting, split instructions into a variable), these scripts will silently fail or corrupt the file. Keep `instruction="""..."""` intact unless you also update those scripts.
2. **Eval scripts run `eval` with `cwd` = repo root.** Both invoke `python -m google.adk.cli eval ...` from the repo root, passing the agent name `text_summarizer` and paths relative to it. Run them from the repo root (or from `text_summarizer/` where the script itself manages `cwd`).
3. **`auto_optimize.py` REVERTS to the best-known instructions** when a rewrite doesn't improve the score, and appends a record to `optimization_history.json`. The final `agent.py` state reflects the best run, always.
4. **`response_match_score` is ROUGE-1 word overlap, not output quality.** Higher scores come from *matching the expected words*, not from being objectively better. The README documents experiments where "better" edits *lowered* the score. Don't use it to judge semantic quality.
5. **The two eval criteria do very different jobs — don't treat them as interchangeable.**
   - `tool_trajectory_avg_score` is the **hard gate** for rules 8–9. The golden answers declare `tool_uses: [save_summary_to_second_brain, log_conversation]`, and `test_config.json` sets `matchType: "IN_ORDER"` + `ignoreArgs: true` so the retrieval calls (`search_text`, `note_read`, `vault_info`) may interleave but the two mandatory calls must appear, in order. A run where the agent skips persistence scores **0.00**.
   - `response_match_score` (threshold 0.5) is only a **coarse regression guard** on wording. Known limitation: the `**Sources**` block is *conditional on vault state*, so a run that cites nothing scores lower ROUGE-1 than a golden that has the block, and vice versa. That's expected — don't "fix" it by deleting the Sources rule.
   - The `vault_sources_block` case is the only golden that contains a `**Sources**` block, and it bakes in this deployment's identifiers (`ck`, `gemini-3.5-flash-lite`). Changing `VAULT_NAME`, `MODEL_PROVIDER` or the served backend lowers the score **for that case only** (its floor is ~0.70 without the block, still above the 0.5 threshold). The other three cases cite nothing and are unaffected.
   - Both criteria are needed. Trajectory alone can't judge prose; ROUGE-1 alone can't tell whether the second brain was written at all.
6. **The eval extra is a runtime dependency.** `response_match_score` (both in `adk eval` and in the Langfuse callback via `eval_scoring.py`) needs `google-adk[eval]` → `rouge_score`. It's declared in `pyproject.toml`; if you trim deps, `response_match_score` degrades silently (returns no score) while the rest still works. As of 2026-09-26 the checked-in `text_summarizer/.venv` does **not** have `rouge_score` installed (the venv is out of sync with `pyproject.toml`/`uv.lock`), so `response_match_score` is currently a no-op for local runs until `uv sync --frozen` is re-run.
7. **`.env` files hold real API keys** (Gemini, OpenRouter, Langfuse). Both `.env` are git-ignored — never commit them, never print their contents into logs/commits. Use `.env.example` as the reference for what's configurable.
8. **`.venv`, `.adk/`, `optimization_history.json`, `__pycache__`, `*.egg-info` are locally-generated state**, git-ignored, safe to delete and regenerate with `uv sync`. Note: `adk eval` writes into `Second Brain/` and `Chat Log/` in the real vault as a side effect — eval runs are not read-only.
9. **Dependencies are managed by `uv`** (`pyproject.toml` + `uv.lock`, `uv sync --frozen`). Prefer `uv` over pip when installing or running.
10. **`adk eval` is NOT hermetic — run it with `CACHE_ENABLED=false`.** `adk eval` calls the real agent, which calls `save_summary_to_second_brain` against the real vault. So a first run writes a summary for each eval prompt; every later run on the same eval set then hits the cache, skips the model *and* the tool calls, and reports a high `response_match_score` while testing nothing. `basic_summary` is already in this state (it collides with a note in the live vault). Always: `CACHE_ENABLED=false uv run adk eval ...`.
11. **Retrieval tools are absent unless MCP is configured.** With neither `OBSIDIAN_VAULT_PATH` nor `OBSIDIAN_MCP_URL` set, `build_obsidian_tools()` returns `[]` and rule 7 is unactionable — the agent will have only the two `FunctionTool`s. The Docker compose path sets `OBSIDIAN_MCP_URL`, so local and container runs are not equivalent.
12. **`tool_context.session` is not a reliable view of the current turn.** In a resumed multi-turn session it can be a snapshot taken before the current user message was appended, so its most recent `user` event still belongs to the *previous* turn. Anything that needs the current turn's input must read `InvocationContext.user_content` (`tool_context.get_invocation_context().user_content`) and fall back to the event scan only if that is unavailable. This bit `note_provenance` too: `generated_by_model` on a note written in a later turn of a resumed session can name the previous turn's backend.
13. **Notes saved before 2026-09-26 have no provenance.** Their frontmatter has no `generated_by_model`, and their bodies use the old `## From the vault` / `[vault] [[Note]]` shape. They are never rewritten, and citing one does not invent a model. They are also unreachable by the vault cache (their `source_fingerprint` is a fingerprint of the model's paraphrase — see gotcha 15). To backfill either, re-ask the question: the note is then written fresh with provenance and a usable key. There is no migration script.
14. **`save_summary_to_second_brain`'s `tool_context` parameter is intentional.** ADK injects it and keeps it out of the model's JSON schema; deleting it as "unused" silently removes both `generated_by_model` and the cache key from every new note.
15. **`save_summary_to_second_brain` has no `source_text` parameter, on purpose.** It used to, and the model was asked to fill it in (rule 8). That was the cache bug: the note was keyed on the model's paraphrase while the lookup used the user's prompt, so the two could never agree. Do not re-add it "for provenance" — the key is computed in code from the live turn. Removing the argument also removed ~1k output tokens per call, which is where a 7.6s generation went. A stale tool call that still carries the argument is harmless: `FunctionTool` filters unknown args before invoking (`google/adk/tools/function_tool.py`, `_prepare_invocation_args`).

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
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_REFRESH_TOKEN` | Optional read-only Gmail MCP. All three must be set; mint the refresh token once with `text_summarizer/gmail_oauth.py` |
| `SECOND_BRAIN_VAULT` | Absolute path of the directory holding the vault(s) for direct writes; defaults to `/vault`. Under Docker point this at the vault's **parent** (`/vaults`) so the real name survives — see "Which vault is active" |
| `OBSIDIAN_VAULT_NAME` | Picks the active vault by directory name when the parent holds several. Never guessed. `./run.sh` can prompt for it and writes it to `.env` |
| `VAULT_NAME` | Name rendered in every `**Sources**` line. Usually **not needed** — the name is derived from the resolved vault path. Set it only when the vault itself is bind-mounted and its name is not in the path. |
| `OBSIDIAN_VAULT_PARENT_HOST` | Docker only. Host path whose children are the vault(s), bind-mounted at `/vaults`. Defaults to `./vaults`; empty mount auto-creates `agent-vault` on first boot. |
| `VAULT_PARENT` | Docker only. In-container mount point that holds the vault (`/vaults`), read by `resolve-vault.sh` and `Dockerfile.obsidian-mcp`. |
| `OBSIDIAN_MCP_PORT` | Docker only. HTTP listen port for the obsidian-mcp server (default in-compose `37842`). |
| `LANGFUSE_PROMPT_NAME` | Prompt name reported on every generation span (Langfuse `prompt_name`, the dashboards' prompt dimension). Defaults to `text_summarizer`. |
| `LANGFUSE_TRACE_NAME` | Trace name reported on every run. Defaults to `text_summarizer`; blank/whitespace falls back to the default. |
| `LANGFUSE_AUTH_CHECK_TIMEOUT` | Seconds to wait for Langfuse's blocking `auth_check()` before instrumenting anyway. Default `5`. Prevents a slow Langfuse from silently disabling all tracing. |
| `CACHE_ENABLED` | `false` bypasses the vault cache and tags traces `cache-disabled`. **Set `false` for `adk eval`.** |

## Evaluating your changes

The intended workflow is: **edit instructions in `agent.py` → run eval → compare score → keep or revert**. Make one change at a time so you can attribute score deltas. Always re-check `git status`/`git diff` after running the eval scripts, since they edit `agent.py` and `optimization_history.json` for you.

## Known gaps (as of 2026-09-26)

Tracked, not yet fixed. Ordered by how much they mislead.

1. ~~The vault cache is invisible in Langfuse.~~ **Done** — `cache-hit`/`cache-miss`/`cache-disabled` trace tags, `cache.outcome` + `cache.lookup_ms` metadata, `cache.hit` + `cache.lookup_ms` scores, and a `CACHE_ENABLED` flag. Still worth doing: the cache.lookup filesystem scan is not its own span, so its cost is only visible as the `cache.lookup_ms` score rather than in the trace waterfall (measured at 0.1–5 ms, so low value).
2. ~~Cache hits inflate the quality scores.~~ **Done** — `quality.*` scores are skipped when `state["vault_cache_hit"]` is true.
3. **`quality.bullet_count` is not a score** — it returns the raw bullet count (`agent.py`), contradicting its own docstring. The normalized 0..1 `bullet_score` is computed and discarded.
4. **`quality.fidelity` is lexical recall, not faithfulness** — word overlap including stopwords. A reply made of stopwords plus one content word scores high. Rename to `quality.lexical_recall` or replace with a real groundedness check.
5. **The harness rewards verbatim copying.** ROUGE-1, `source_overlap`, and instruction rule 3 ("closely mirroring the key terms, phrasing, and sentence structures") all push the same direction. The agent is optimized toward extractive copying.
6. ~~**No `userId` / `sessionId` on traces.**~~ **Done** — `before_agent_callback=tag_trace_identity` sets `langfuse.trace.name`, `user.id` and `session.id` on the open `agent_run` span; Langfuse folds them onto the trace and every observation row. Name is overridable via `LANGFUSE_TRACE_NAME`. See "Trace identity".
7. **The free OpenRouter fallback is only as reliable as its free tier.** With the schema bug fixed (see "Tool-schema sanitisation") the fallback now reaches the model, but `google/gemma-4-26b-a4b-it:free` returns HTTP 429 (`temporarily rate-limited upstream`, `limit_source: upstream_provider_shared_pool`) often enough that a Gemini quota error is as likely to end in a rate-limit error as in a served turn. `FallbackModel` cannot distinguish "this model is throttled" from "this model is broken" and retries the whole chain identically. If fallback reliability matters, the fix is a paid key on OpenRouter (`OPENROUTER_API_KEY`) rather than more `:free` aliases.
8. **Untrusted LLM output used as file paths and YAML** — `title` and `topics` go straight into `os.path.join(VAULT_ROOT, ...)` and the frontmatter block (`second_brain.py`). Sanitize with `_slug()` and quote YAML values.
9. **Non-atomic writes** — `_write` truncates then writes; `log_conversation` and the index do read-modify-write on shared files. Concurrent sessions lose entries. Use append mode for the chat log plus a lock for the index.
10. **`find_cached_summary` is O(n) full-file reads per model call** and regexes the whole note body rather than just the frontmatter. An index file (fingerprint → path) written by `save_summary_to_second_brain` makes it O(1). Deliberately not done yet: measured at 0.5ms against a 19-note vault, so the payoff only appears at a vault size nobody has reached. Note this is now the *only* thing standing between a cache miss and a hit — the key mismatch that used to guarantee a miss is fixed, so this cost is on the path of every single turn.
11. **The vault cache is now exercised end to end, on built code.** `docker compose build` and `up -d --build` both work on this host as of 2026-09-26 (the "two daemons, neither socket serving these containers" problem is gone), and the rebuilt containers resolved the real vault to `/vaults/ck` on both sides. Measured on a real prompt: first ask 23.5s / 11 events, the same prompt in two further new sessions 0.04s / 2 events each with `vault_cache_hit=True`, `cache.hit=1` and no `quality.*` scores. Two bugs were found only by doing this and are now pinned by tests — see "The vault cache" above. Note the containers still stop on their own here; `docker start agent-testing obsidian-mcp` brings them back.
