# PLAN — bold bot name at the head of every answer

> **Status:** implemented. §8 is done; §9's eleven tests are in
> `text_summarizer/tests/test_bot_name.py`. `make lint` and `make types` are clean for
> every file this touched, and `make test` is green apart from nine pre-existing
> `test_tools_gif.py` failures that belong to the other workstream's uncommitted work.
> **Not yet verified live** — the running container is an image built before this change
> (gotcha 17), so a real turn needs `docker compose up -d --build` of *both* services.
>
> **File name is deliberate.** `PLAN.md` exists on `origin/main` (from `d54c55b`) and is
> absent from this working tree only because local `main` is 11 commits behind. Creating a
> local `PLAN.md` would collide with it on the next pull. This file is `PLAN-bot-name.md`.
>
> **Claims marked **[verified]** were read in the installed ADK or executed, not inferred.

---

## §1 — What is being asked

Every answer the user sees begins with the bot's name in bold:

```
**Text Summarizer Agent**

- Dogs were domesticated from wolves roughly 15,000 years ago.
- …

**Sources**
- [obsidian][ck][gemini-3.5-flash-lite][[…]](…): why it is relevant
```

The name goes **first**, the `**Sources**` block stays **last** (rule 7 already requires the
source lines at the very end of the answer).

**It is not redundant with the dev UI.** **[verified]** The bundled Angular app keys each
message on the literal string `role:"bot"` and aligns on `e.role==="bot" ? 0 : 1`; there is
no per-message agent-name label anywhere in the bundle. Today the dev UI shows *which side
of the conversation* a message is on and nothing else. The name line is the only visible
attribution, which is why this is a real gap rather than a nicety.

## §2 — Recommendation: render it in code, never ask the model

The existing design already refuses to ask the model for anything it can compute — the
`**Sources**` block is emitted from sentinels precisely so the vault and model names are
substituted in code, and rule 7 says so explicitly. The name belongs to that family.

| | In code (recommended) | As an instruction rule |
|---|---|---|
| Determinism | always present, or never | the model may drop it, vary it, or bold it wrongly |
| Output tokens | 0 | ~15 per turn, on every turn |
| Vault notes | untouched | polluted — the name is inside `summary_content` and gets persisted |
| Eval | inert by construction | shifts every golden answer |

## §3 — **[verified]** The one implementation detail that decides the design

`after_model_callback` is a **list**, and ADK stops at the first callback that returns a
response:

- `_run_callbacks(agent.canonical_after_model_callbacks, _stop_on_truthy, …)` —
  `google/adk/utils/_callback_pipeline.py:94-101`, called from
  `google/adk/flows/llm_flows/base_llm_flow.py:330-336`.
- `_stop_on_truthy` is `bool(result)` — `_callback_pipeline.py:116-118`.

So the obvious implementation — appending a third entry to the existing list at
`agent.py:462` — **silently does nothing on exactly the turns that have a Sources block**,
because `render_sources_after_model` returns a response on those turns and the chain stops
there. It would appear to work on turns with no citations and fail on turns with them, which
is the hardest kind of bug to notice.

**Therefore: one callback that does both steps, in order, returning one response.**

```
finalize_answer_after_model(callback_context, llm_response)
  ├─ render_sources_after_model's existing logic   (sources.py:577)
  └─ prepend_bot_name(rendered)                     (new, ~8 lines)
```

## §4 — **[verified]** A latent bug this would make worse, and must be fixed first

`_finalize_model_response_event` replaces the event's `content` **wholesale** with whatever
the callback returned (`google/adk/flows/llm_flows/base_llm_flow.py:125-142`, `updates` is
built from every non-`None` field, and `content` is one of them).

`render_sources_after_model` returns

```python
llm_response.model_copy(
    update={"content": Content(role="model", parts=[Part(text=rendered)])}
)                                                            # agent.py:331-333
```

— a `Content` with **one text part and nothing else**. If the model ever returns text *and*
a function call in the same response, the function call is deleted from the event, the flow
never sees it (`base_llm_flow.py:858` dispatches on `get_function_calls()`), and the agent
silently stops calling tools for that turn. The model is currently *instructed* to put its
source lines at the very end of the answer, after the tool calls, so this has probably never
fired — **unverified, and it costs a live turn with a stub to check.**

The name stamp makes it materially worse: it applies to **every** response with text, not
just the ones carrying a Sources block, so it multiplies the number of responses exposed to
this path.

**Fix, in the same change:**

1. **Guard** — return `None` (leave the response alone) when the response carries a function
   call. That response is not the final answer, so it needs neither the name nor a
   substituted block.
2. **Preserve** — rebuild `Content` from the original parts, substituting the first text part
   in place and dropping any *further* text parts, so `function_call` / `inline_data` parts
   survive by construction rather than by the guard alone.

## §5 — Emit points, and what each one does

There are exactly two ways a user-visible answer is produced. Both need the name, and
neither may put it on disk.

| # | Path | Site | Note stays clean? |
|---|---|---|---|
| 1 | Fresh answer | `finalize_answer_after_model` — new wrapper around `render_sources_after_model` (`agent.py:287`) | yes — the model never writes the name, so `summary_content` and the persisted note are unchanged |
| 2 | Cache hit | `cache_hit_before_model`, on the `LlmResponse` it returns (`agent.py:206-209`) | yes — the stamp is applied to the emitted copy, never to the stored note |

**Path 2 — decided: stamp it.** `test_cache_hit_replays_the_stored_note_verbatim`
(`test_adk_wiring.py:218-253`) asserts `final.strip() == stored`, and AGENTS.md's
"Cache hits are untouched" documents verbatim replay, so both are amended. What is *not*
amended is the invariant underneath them: **the note on disk is never rewritten.** The stamp
is applied to the emitted copy only. Without it the same question shows a name on the first
ask and none on every repeat, which reads as a cache bug.

## §6 — Metrics: strip it, so a presentation change moves no score

`report_scores_after_agent` reads the emitted text off the session
(`agent.py:384-397`) and hands it to `_score_generation` and `response_match_for_agent`.

| Metric | Effect of an unstripped name line |
|---|---|
| `quality.bullet_count` | **none** — the regex is `(?m)^\s*-\s+` (`agent.py:233`); a `**name**` line is not a bullet |
| `quality.source_overlap` / `quality.fidelity` | small **drop** — `text_summarizer` is in the response but not in the user's message |
| `response_match_score` (ROUGE-1) | small **drop** on all 4 eval cases, and it would land in the middle of a series people compare across instruction edits |

The `**Sources**` block already has a precedent: `summary_only()` strips it before scoring,
because source lines would otherwise be counted as summary bullets. The name line is the
same class of artefact. **Decided: strip it at both call sites**, and leave
`summarizer_eval_set.evalset.json` alone — the goldens describe the ideal *summary*, and a
bold name is not part of it. `tool_trajectory_avg_score` is unaffected either way.

## §7 — Configuration

**Decided.** One variable, `BOT_NAME`.

| Var | State | Displayed name |
|---|---|---|
| `BOT_NAME` | unset (the default) | **derived from the agent's own name** — see below |
| `BOT_NAME` | set | exactly that value |
| `BOT_NAME` | set to `""` | **feature off** — text is emitted unchanged |

The default is derived, not hardcoded, so there is one source of truth and the label cannot
drift from `LlmAgent(name=…)`:

```python
AGENT_NAME = "text_summarizer"          # used by LlmAgent(name=…) and by the label

def bot_name() -> str:
    if (raw := os.environ.get("BOT_NAME")) is None:
        return f"{' '.join(w.capitalize() for w in AGENT_NAME.split('_'))} Agent"
    return raw.strip()                  # "" means off
```

`text_summarizer` → `Text Summarizer Agent`. Rename the agent to `research_digest` and the
label becomes `Research Digest Agent` with no second edit. `BOT_NAME="Second Brain"` gives a
product name without touching code.

The three-way distinction (`None` vs `""` vs a value) is what makes the feature measurable:
an eval run with `BOT_NAME=""` is the control for a run without it. This repo already
depends on exactly that idea for `CACHE_ENABLED` (gotcha 10).

`AGENT_NAME` becomes a module constant used by both `LlmAgent(name=…)` and the resolver.
Gotcha 1 is unaffected: `eval_exercise.py` and `auto_optimize.py` regex on
`instruction="""…"""`, not on `name=`.

## §8 — Change surface

| File | Change |
|---|---|
| `text_summarizer/agent.py` | `AGENT_NAME` constant; `bot_name()` resolver; `prepend_bot_name()`; `finalize_answer_after_model()` wrapper; function-call guard + parts-preserving rebuild; stamp on the cache-hit response; strip before scoring |
| `text_summarizer/sources.py` | `strip_bot_name()` alongside `summary_only()` (or a `BOT_NAME_LINE` regex constant shared by add/strip) |
| `text_summarizer/tests/test_bot_name.py` | **new** — the tests in §9 |
| `text_summarizer/tests/test_adk_wiring.py` | `test_cache_hit_replays_the_stored_note_verbatim:247` (`== stored` → name line + stored); the one-event test at `:128` gains a first-line assertion |
| `text_summarizer/tests/test_agent_callback.py` | callback unit tests for the new guard and idempotency |
| `.env.example` | document `BOT_NAME` |
| `AGENTS.md` | answer shape; extend gotcha 16's rule ("to change what the user sees, use `after_model_callback`"→ `after_model`) with the `_stop_on_truthy` fact; note the verbatim-replay caveat |
| `docs/ARCHITECTURE.md`, `docs/index.html` | the answer format as documented/displayed |

## §9 — Tests (all zero API cost)

1. The name line is the first line of a fresh answer, exactly once — `**Text Summarizer
   Agent**` by default.
2. The default label is derived, not literal: monkeypatching `AGENT_NAME` changes it, which
   is what proves there is one source of truth.
3. `BOT_NAME="Second Brain"` overrides the label; `BOT_NAME=""` turns the feature off and
   the answer comes back byte-identical to the model's.
4. The answer is still **one event** — asserted on event *count*, per gotcha 16, not on text.
5. Stamping is idempotent: a second pass yields one line.
6. The persisted note contains no name line (`Second Brain/*.md` unchanged by a turn).
7. A cache hit emits the name line **and** leaves the stored note byte-identical.
8. **Regression, §4:** a response carrying a function call is left untouched *and the
   function call survives* — i.e. `get_function_calls()` is non-empty after the callback.
9. No name on an empty response, a tool-only response, or a `partial` (streamed) response.
10. `render_sources(render_sources(x)) == render_sources(x)` still holds with the name present.
11. `_score_generation` / `response_match_for_agent` see the text **without** the name line.

## §10 — What this does not do

- **Not in the prompt**, so nothing changes in `instruction` and `eval_exercise.py` /
  `auto_optimize.py` are unaffected.
- **Not in the vault.** The note, the chat log and the `**Sources**` block are unchanged.
- **Not on streamed chunks** (gotcha 12). Inert as deployed — the dev UI posts
  `streaming: false` — and the `partial` guard is the same one the renderer already uses.
- **Not retroactive.** It stamps emitted text, so nothing about existing notes or old
  sessions changes.
- **Not verified live.** Everything above is read from the installed ADK and the repo. The
  function-call exposure in §4 and the visual result both need one real turn after
  `docker compose up -d --build` (both services, per AGENTS.md) to confirm.

## §11 — Decisions (locked)

| # | Question | Answer |
|---|---|---|
| 1 | Display name | **Derived from the agent's own name** → `text_summarizer` renders as **`Text Summarizer Agent`**. Human-readable rather than machine-readable, still one source of truth. `BOT_NAME` overrides. |
| 2 | Format | `**Text Summarizer Agent**` on its own line, blank line, then the bullets — so rule 1's bullet block stays contiguous and rule 3's phrasing is untouched. |
| 3 | Cache hits | **Stamp them.** Emitted text only; the stored note is never rewritten. |
| 4 | Metrics | **Strip the name** before `quality.*` and `response_match_score`. Eval goldens unchanged. |

**Awaiting approval to implement.** Nothing in §8 has been touched.