# Getting Started

A **text summarization agent** built with Google's [Agent Development Kit](https://google.github.io/adk-docs/) (ADK), with an evaluation loop, an Obsidian "second brain", and optional read-only Gmail access.

Everything is optional except one model key. The interesting part of this repo is the **evaluation loop**: you edit the agent's instructions in `agent.py`, run ADK evals, and watch `response_match_score` (ROUGE-1 word overlap) change.

---

## 1. Set your API key

`./run.sh` creates a `.env` from `.env.example` on first run. Open it and set **`GEMINI_API_KEY`** — get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

| Optional | What it adds | Vars |
|---|---|---|
| OpenRouter | free models as a fallback when Gemini is rate-limited | `MODEL_PROVIDER=openrouter`, `OPENROUTER_API_KEY` |
| Langfuse | traces every turn: latency, cache hit/miss, quality scores | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` |
| Gmail | read-only inbox tools | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN` |
| Obsidian | retrieval tools over your vault | set automatically by `run.sh` |

`.env` holds real keys and is git-ignored. Never commit it.

## 2. Run it

```bash
./run.sh
```

That picks your Obsidian vault, builds both containers (the agent web UI and the
`obsidian-mcp` server), starts them, and prints the URLs.

- **Agent web UI:** <http://localhost:8001>

Already use Obsidian? `./run.sh` finds your existing vaults — including the ones
in your Obsidian config — and asks which one the agent should use, then remembers
the answer. One vault needs no configuration; several are never guessed between.
See [Using a vault you already have](INTEGRATIONS.md#using-a-vault-you-already-have).

## 3. Try it

In the web UI, ask for a summary:

```
Summarize: Dogs are domesticated animals known for loyalty and companionship. They are often called man's best friend. Dogs come in many breeds, varying in size, color, and temperament.
```

With Gmail configured, the agent reads your inbox too:

```
What are my latest 5 emails? Summarize them.
```

Every turn is summarised into your vault as a linked note and appended to that
day's chat log. **Ask the same thing again and it is answered from the vault with
no model call at all** — see [the vault cache](ARCHITECTURE.md#vault-cache).

---

## The vault, in one paragraph

Summaries, chat logs, topic stubs, and an index graph land in a real Obsidian
vault, so the notes form a graph you can browse. With no configuration you get
`agent-vault`, auto-created on first boot. With a vault of your own, either point
`OBSIDIAN_VAULT_PARENT_HOST` at the directory it lives in and set
`OBSIDIAN_VAULT_NAME`, or just let `./run.sh` prompt you. The mount is always the
vault's **parent**, never the vault itself, so the real folder name survives into
the container and every `**Sources**` line names the real vault. Details, including
what happens with several vaults and why it refuses to guess, are in
[Integrations](INTEGRATIONS.md#using-a-vault-you-already-have).

## What else is here

| Doc | Contents |
|---|---|
| [EVALUATION.md](EVALUATION.md) | Eval criteria, running evals, hands-on experiments, auto-optimization |
| [TESTING.md](TESTING.md) | Prompt examples for the web UI (good / bad / edge cases) |
| [MODELS.md](MODELS.md) | Switching between Gemini and free OpenRouter models |
| [INTEGRATIONS.md](INTEGRATIONS.md) | Read-only Gmail access and the Obsidian second brain |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How a turn flows, provenance, vault selection, the cache, Docker topology |

`AGENTS.md` is the deep end: it records *why* each design decision was made and
which traps are easy to fall into again. Read it before changing anything in
`text_summarizer/`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Gemini API key not valid` | `GEMINI_API_KEY` unset or wrong in `.env` |
| `'/vaults' holds N vaults; refusing to guess` | several vaults in the mount with no `OBSIDIAN_VAULT_NAME` — re-run `./run.sh` and pick one |
| `'/x' is not a directory under '/vaults'` | `OBSIDIAN_VAULT_NAME` has a typo; the error lists the valid names |
| `obsidian-mcp` restarting | the MCP server exits rather than serve a wrong vault — read its error, it names the candidates |
| No Gmail tools in the UI | all three `GOOGLE_*` vars must be set; see [Gmail setup](INTEGRATIONS.md#gmail-read-only) |
| Eval scores look great but nothing is being tested | run evals with `CACHE_ENABLED=false` — see [AGENTS.md gotcha 10](../AGENTS.md) |
