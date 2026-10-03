# Model Chains

The agent runs on a **chain**, not a single model. `MODEL_PROVIDER` chooses which
chain is built; both are wrapped so one provider never sees another's conversation
history. The same `MODEL_PROVIDER` env var controls the agent and the auto-optimizer.

## The default chain — `MODEL_PROVIDER=gemini`

```
 1. hosted_vllm/space-bunny-free     OpenCode Zen
 2+. openrouter/<model>:free          every free model, once per configured key
 14. gemini-3.5-flash-lite
```

**Order matters and it is not alphabetical.** `FallbackModel` tries each entry in
sequence, so the **first entry serves every ordinary turn** and the rest exist only for
when it fails (429, quota, 5xx). The default order was inverted on 2026-10-03 — Gemini
used to lead, which meant nothing else was ever reached. Measured before the change on
session `068f3e5c`: Gemini served all nine generations, because it was first and it did
not fail.

Gemini is **last, and unconditional**. It is the only tier here that needs no API key,
so it is also the floor — a chain without it would end the turn outright on a quota
error instead of degrading. Everything ahead of it is a free tier, and free tiers rate
limit.

### What that costs

Every turn now goes to a **free** endpoint, which is the honest trade: you get a
preferred provider at the cost of reliability on the hot path. See
[known gap 7](../AGENTS.md#known-gaps) — `gemma:free` and its neighbours 429 often
enough that a quota error is as likely to end in a rate-limit error as in a served turn.

If a free tier times out, **the chain does not advance** — see
[When a timeout kills the turn](#when-a-timeout-kills-the-turn) below.

## The OpenRouter-led chain — `MODEL_PROVIDER=openrouter`

```
 1. openrouter/<MODEL_ALIAS>         the model you named
 2+. the other free models
```

Unchanged, and it contains **no Gemini at all** — the history guard is therefore
vacuous on this path. `MODEL_ALIAS` picks the primary:

| Alias | Model |
|---|---|
| `lfm` | `liquid/lfm-2.5-2.6b:free` |
| `ling` | `inclusionai/ling-3.0-flash-sante:free` |
| `dots` | `dots-studio/dots-3-note-preview:free` |
| `gemma` | `google/gemma-4-26b-a4b-it:free` |
| `qwen` | `qwen/qwen3.8-27b:free` |
| `nvidia` | `nvidia/nemotron-3.5-lightning:free` |

Empty or unrecognised falls back to the first free model OpenRouter lists. These are
free-tier aliases this project actually uses; don't "fix" them to older released models.

## The history guard — `model_chain.py`

Both paths return a `HistorySafeFallbackModel`, which wraps **two** `FallbackModel`
chains and picks per call from the payload:

- **`models`** — the full chain
- **`unsigned_history_models`** — the same chain with every signature-requiring entry removed

Gemini validates the *entire* conversation on each call, including turns it did not
take part in, and rejects any `functionCall` part with no `thought_signature`. So a
mid-turn fallback to a free model poisons the history for the primary, and the next call
dies on a 400 that propagates instead of failing over. The wrapper exists so that a
signature-requiring model is **never offered a conversation it will reject**.

It is stateless by construction: there is no turn identity on `LlmRequest` to pin
against, so deriving the choice from the payload means no turn boundary exists to get
wrong.

## When a timeout kills the turn

**Known limitation.** `FallbackModel` only advances on a failure that carries one of
`DEFAULT_STATUS_CODES` — 429, 500, 502, 503, 504. A **client-side timeout carries no
status**, so it propagates and ends the turn with the remaining 13 models untried.

ADK excludes 408 deliberately:

> *"a request that timed out costs one more attempt at worst… litellm reports even a
> client-side timeout as 408, so failing over on it risks paying for and acting on one
> prompt twice. Add it back for a provider whose 408 is known to mean the request was
> dropped."*

Observed live on session `flow-r6`, after the reorder put a free endpoint first:

```
  9. ts=…511   last tool response
 10. ts=…572   ERR litellm.Timeout: Connection timed out. Timeout passed=60.0
```

One error event, no second model tried. **OpenCode Zen is exactly the case ADK's
escape hatch describes** — `Connection timed out` means the request never reached the
service — so the fix is to surface that one provider's timeouts as `504` rather than
blanket-adding 408 for a paid tier where the double-charge risk is real.

## Environment

| Variable | Meaning |
|---|---|
| `MODEL_PROVIDER` | `gemini` (default) or `openrouter`. **The default no longer means "Gemini first"** — that branch is OpenCode → free OpenRouter → Gemini since 2026-10-03, so `gemini` names the *tier that is always present*, not the one that leads. |
| `MODEL_ALIAS` | Which free OpenRouter model leads the `openrouter` chain. Ignored on the default path. |
| `OPENROUTER_API_KEY` | One or more keys, `;`-separated. Several keys give several times the daily budget, and the chain contains each account's models. |
| `OPENCODE_API_KEY` | Gates the Zen tier. Without it the tier is absent, not broken. |
| `OPENCODE_API_BASE` | Zen's OpenAI-compatible endpoint. |
| `GEMINI_API_KEY` | Required for the Gemini floor. |

## How the chain is built

In `agent.py`, `get_model()`:

```python
if MODEL_PROVIDER == "openrouter":
    ordered = [MODEL_ALIAS's model, *the other free models]
    chain = [_BreakerLiteLlm(name, key) for key in keys for name in ordered]
else:
    chain = []
    if opencode_enabled():
        chain.append(_opencode_llm())
    if keys:
        chain += [_BreakerLiteLlm(name, key) for key in keys for name in free_models]
    chain.append(GEMINI_MODEL)          # the floor, always present

return HistorySafeFallbackModel(
    models=chain,
    unsigned_history_models=signature_free_chain(chain),
)
```

Two things worth noticing. **The breaker skips spent keys at build time** *and*
`model_chain`'s `_BreakerLiteLlm` makes the same decision per call — the first shrinks
the chain, the second covers an account that runs dry during a process's lifetime,
which a chain built at import cannot see.

**Only `space-bunny-free` is reachable on Zen.** Four of its five free models answer
`403 FreeTierError: "OpenCode's free tier can only be used from within OpenCode"` from
inside a container.

## Switching providers

```bash
# Default: OpenCode, then free OpenRouter, then Gemini
MODEL_PROVIDER=gemini uv run adk web text_summarizer

# OpenRouter-led, with a named free model at the front
MODEL_PROVIDER=openrouter MODEL_ALIAS=lfm uv run adk web text_summarizer
```

To see which model actually served a turn, read `**Sources**` in the answer — it carries
the served model — or query Langfuse by `session.id`.