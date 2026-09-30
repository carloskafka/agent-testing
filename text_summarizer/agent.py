import os
import re
import time

from google.adk.agents import LlmAgent
from google.adk.models import FallbackModel, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.genai.types import Content, Part
from opentelemetry import trace as otel_trace

from .eval_scoring import response_match_for_agent
from .gmail_tools import build_gmail_tools
from .observability import (
    langfuse_client,
    report_cache_outcome,
    tag_current_span,
    tag_trace_identity,
)
from .obsidian_tools import build_obsidian_tools
from .second_brain import (
    VAULT_ROOT,
    find_cached_summary,
    log_conversation,
    save_summary_to_second_brain,
)
from .sources import (
    mcp_vault_name_from_events,
    render_sources,
    resolve_vault_name,
    summary_only,
)

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "gemini")

# Set CACHE_ENABLED=false to bypass the vault cache. Keep this OFF for `adk eval`:
# a cache hit skips the model AND the tool calls, so the run tests nothing while
# still reporting a high response_match_score against the stored summary.
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

CACHE_HIT_STATE_KEY = "vault_cache_hit"

# Invocation id of the turn whose cache lookup has already been done. See
# _first_model_call_of_invocation for why the cache must only be consulted once
# per turn. Prefixed to stay clear of anything the model or the UI might use.
CACHE_CHECKED_INVOCATION_KEY = "_vault_cache_checked_invocation"

GEMINI_MODEL = "gemini-3.5-flash-lite"

# Only models with a ":free" suffix are real free-tier models on OpenRouter.
OPENROUTER_MODELS = {
    "gemma": "google/gemma-4-26b-a4b-it:free",
    "qwen": "qwen/qwen3.8-27b:free",
    "nvidia": "nvidia/nemotron-3.5-lightning:free",
}

MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "")


def _free_openrouter_models() -> list[str]:
    return [
        name for name in OPENROUTER_MODELS.values() if name.endswith(":free")
    ]


def _openrouter_llm(model_name: str) -> LiteLlm:
    return LiteLlm(model=f"openrouter/{model_name}")


def get_model():
    if MODEL_PROVIDER == "openrouter":
        primary_name = OPENROUTER_MODELS.get(MODEL_ALIAS) or _free_openrouter_models()[0]
        fallback_names = [
            name for name in _free_openrouter_models() if name != primary_name
        ]
        return FallbackModel(
            models=[
                _openrouter_llm(primary_name),
                *(_openrouter_llm(name) for name in fallback_names),
            ]
        )
    # Default provider is Gemini; fall back to free OpenRouter models on quota
    # exhaustion (HTTP 429) or transient 5xx errors.
    return FallbackModel(
        models=[
            GEMINI_MODEL,
            *(_openrouter_llm(name) for name in _free_openrouter_models()),
        ]
    )


model = get_model()


def _last_user_text(llm_request) -> str:
    """Extract the text of the most recent user message from the LLM request."""
    contents = llm_request.contents or []
    for content in reversed(contents):
        if getattr(content, "role", None) == "user":
            return "".join(part.text or "" for part in content.parts or [])
    return ""


def _user_content_text(context) -> str:
    """The verbatim user message of the current turn, from the live invocation.

    ``CallbackContext.user_content`` is the field ADK documents for exactly this,
    and it is the *same* field ``second_brain.cache_key_text`` reads on the write
    side, so the cache key cannot drift between the two.

    Deliberately not read from ``llm_request.contents``: ADK models a tool result
    as a ``Content`` with role ``"user"``, so the last user-role part of a
    follow-up request is the tool's JSON output rather than the prompt. Keying on
    that silently turns every mid-turn lookup into a miss on a garbage key.
    """
    content = getattr(context, "user_content", None)
    if content is None:
        return ""
    return "".join(
        part.text or ""
        for part in (getattr(content, "parts", None) or [])
        if getattr(part, "text", None) is not None
    )


def _first_model_call_of_invocation(callback_context) -> bool:
    """True the first time the model is consulted in this invocation, then False.

    ``before_model_callback`` fires before *every* model call in a turn, not just
    the first. That matters because the agent persists its note from inside the
    turn: by the time the model is called again -- to write the actual answer
    after ``save_summary_to_second_brain`` returned -- the fingerprint for the
    current prompt is already on disk. Consulting the cache then would find the
    turn's own note and replay it instead of the answer, truncating the turn.

    Tracked by ``invocation_id`` in the session state rather than by inspecting
    ``session.events``: those events are **not populated** for a database-backed
    session service (the default under ``adk web``), so a positional check sees an
    empty list, decides every call is the first, and the guard silently never
    fires. The invocation id is per-turn, so it is both reliable and self-resetting
    on the next turn.
    """
    invocation_id = getattr(callback_context, "invocation_id", None)
    if not invocation_id:
        # No invocation to key on: fall back to letting the cache be consulted,
        # which is the pre-existing behaviour rather than a silent no-op.
        return True
    already = False
    try:
        already = callback_context.state.get(CACHE_CHECKED_INVOCATION_KEY) == invocation_id
    except Exception:  # pragma: no cover - state is best-effort signalling only
        return True
    if already:
        return False
    try:
        callback_context.state[CACHE_CHECKED_INVOCATION_KEY] = invocation_id
    except Exception:  # pragma: no cover
        pass
    return True


def cache_hit_before_model(callback_context, llm_request):
    """If the vault already has a summary for this exact text, return it with zero LLM calls.

    Returns an LlmResponse (which skips the model invocation) on a fingerprint
    match, otherwise None to let the model run normally.

    The key is the fingerprint of the user's own message, read from the same
    invocation field on both sides of the round-trip. Neither side asks the model
    what the input was: when the model supplied that value it paraphrased it per
    run, so the key never matched and every repeat of a prompt re-paid the full
    turn (three model round-trips, ~12s in the Gmail flow).

    Only the first model call of a turn consults the cache -- see
    :func:`_first_model_call_of_invocation` for why the follow-up calls must not.

    Because the replay happens before any tool runs, a cache hit also skips
    ``log_conversation`` -- the exchange is not recorded a second time. That is
    what keeps ``CACHE_ENABLED=false`` meaningful for ``adk eval``.

    Every outcome is reported to Langfuse (tag + `cache.hit` / `cache.lookup_ms`
    scores) so cache-hit and live-LLM traces can be told apart and compared.
    """
    user_text = _user_content_text(callback_context) or _last_user_text(llm_request)
    if not CACHE_ENABLED or not user_text:
        _mark_cache(callback_context, enabled=CACHE_ENABLED, hit=False, elapsed_ms=0.0)
        return None

    if not _first_model_call_of_invocation(callback_context):
        # Not a cache opportunity, and not a miss worth reporting either: the
        # model is already committed to running for this turn.
        return None

    started = time.perf_counter()
    cached = find_cached_summary(user_text)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _mark_cache(callback_context, enabled=True, hit=bool(cached), elapsed_ms=elapsed_ms)

    if not cached:
        return None
    return LlmResponse(
        content=Content(role="model", parts=[Part(text=cached)]),
        turn_complete=True,
    )


def _mark_cache(callback_context, *, enabled: bool, hit: bool, elapsed_ms: float) -> None:
    """Record the cache outcome on the invocation state and on the Langfuse trace."""
    try:
        callback_context.state[CACHE_HIT_STATE_KEY] = bool(hit and enabled)
    except Exception:  # pragma: no cover - state is best-effort signalling only
        pass
    report_cache_outcome(enabled=enabled, hit=hit, elapsed_ms=elapsed_ms)


def _score_generation(user_text: str = "", model_text: str = "") -> dict:
    """Compute cheap, deterministic quality metrics for the agent response.

    Returns a name->score map that is pushed to Langfuse per call:
      - bullet_count: number of '-' bullet points (0..1 normalized, ideal 3-5)
      - source_overlap: fraction of source words present in the summary
      - fidelity: capped bullets + overlap combined 0..1

    ``model_text`` is passed through ``summary_only()`` first so the Sources
    block is not mistaken for summary bullets.
    """
    model_text = summary_only(model_text or "")
    bullets = re.findall(r"(?m)^\s*-\s+", model_text or "")
    n = len(bullets)
    bullet_score = 1.0 if 3 <= n <= 5 else max(0.0, 1.0 - abs(n - 4) * 0.2)

    source_words = set(re.findall(r"\b\w+\b", (user_text or "").lower()))
    model_words = set(re.findall(r"\b\w+\b", (model_text or "").lower()))
    overlap = 0.0
    if source_words and model_words:
        overlap = len(source_words & model_words) / len(source_words)

    fidelity = min(1.0, (bullet_score + overlap) / 2)
    return {
        "bullet_count": round(n, 3),
        "source_overlap": round(overlap, 3),
        "fidelity": round(fidelity, 3),
    }


def _current_trace_id() -> str | None:
    """Best-effort OTel trace id (32-char hex) of the running agent invocation."""
    span = otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.trace_id and ctx.trace_id != otel_trace.INVALID_TRACE_ID:
        return f"{ctx.trace_id:032x}"
    return None


def report_scores_after_agent(callback_context, agent_response_list=None):
    """Attach the turn's quality scores to the current trace, and return ``None``.

    Scoring only. The ``**Sources**`` block is rewritten by
    :func:`render_sources_after_model`, in place, on the response the model
    produced.

    **This callback must never return content.** Content returned from an
    ``after_agent_callback`` does not replace the agent's response, it becomes an
    *additional* ``Event`` (``BaseAgent._handle_after_agent_callback`` builds a
    new one and yields it after the flow is done). Both events are
    assistant-authored, so the dev UI draws the same answer twice -- see
    :func:`render_sources_after_model` for the session that showed it.
    """
    _report_scores_after_agent(callback_context)
    return None


def _llm_response_text(llm_response) -> str:
    """Every text part of a model response, concatenated."""
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(
        part.text for part in parts if getattr(part, "text", None) is not None
    )


def render_sources_after_model(callback_context, llm_response):
    """Substitute the real vault and served model into the response's Sources block.

    Wired as an ``after_model_callback``: an ``LlmResponse`` returned from there
    **replaces** the model's own response (``_handle_after_model_callback`` ->
    ``_finalize_model_response_event``), so the answer exists once, on the event
    the model produced.

    It used to be an ``after_agent_callback``, which appends a second event
    instead of editing the first. The user saw the answer twice, the copy on the
    end being the rendered one and the first copy still carrying the raw
    ``@@ADK_VAULT@@`` / ``@@ADK_MODEL@@`` sentinels. Confirmed against session
    ``55b1f828-6ff0-498e-953f-8cc29b84cb93``: nine events, the last two the same
    answer, one of them the pre-substitution text.

    Two things this gets for free by running here rather than at the end of the
    turn: the served model is ``llm_response.model_version`` directly, with no
    backwards scan of the session; and the substituted text is what the scoring
    callback later reads, so ``response_match_score`` is computed on the response
    the user actually sees rather than on the model's raw draft.

    Returns ``None`` -- leaving the response untouched -- when there is nothing
    to substitute, which is the case for every model call that is not the final
    answer: a tool-calling response has no text, and text without a Sources block
    renders to itself.
    """
    if getattr(llm_response, "partial", False):
        # A streamed chunk is not the whole answer, and a sentinel token can
        # straddle two chunks. `adk web` does not stream (the dev UI posts
        # streaming=false), so this guard is inert in this deployment.
        return None

    text = _llm_response_text(llm_response)
    if not text.strip():
        return None

    try:
        rendered = _render_sources_text(callback_context, llm_response, text)
    except Exception as exc:  # pragma: no cover - never break the turn
        print(f"[sources] render failed: {exc}")
        return None

    if rendered == text:
        return None
    return llm_response.model_copy(
        update={"content": Content(role="model", parts=[Part(text=rendered)])}
    )


def _render_sources_text(callback_context, llm_response, text: str) -> str:
    """Substitute both identifiers into ``text``. See :func:`render_sources`.

    Both are resolved at runtime: the vault name via ``resolve_vault_name`` and
    the served model via ``LlmResponse.model_version`` (see ``sources.py``).
    Nothing is hardcoded and nothing is asked of the model.

    The resolved vault root has to be passed in explicitly. Under Docker
    ``SECOND_BRAIN_VAULT`` is the *parent* that gets bind-mounted (``/vaults``)
    and the active vault is its single child (``/vaults/ck``), so letting
    ``resolve_vault_name`` fall back to the env var renders the parent's name --
    a plausible-looking ``[vaults]`` where every note's frontmatter correctly
    says ``ck``. ``second_brain.note_provenance`` already passes the root; this
    is the same call on the response side.
    """
    vault = resolve_vault_name(
        vault_root=VAULT_ROOT,
        mcp_reported=mcp_vault_name_from_events(_session_events(callback_context)),
    )
    return render_sources(
        text,
        vault_name=vault.name,
        # Same field as `Event.model_version` -- Event subclasses LlmResponse --
        # but reachable without walking the session.
        model_name=getattr(llm_response, "model_version", None) or None,
        # Makes the note titles clickable: a title that resolves to a real file
        # in the vault is linked to the dev UI's /vault route, one that does not
        # stays a plain [[wikilink]]. See sources.note_href.
        vault_root=VAULT_ROOT,
    )


def _report_scores_after_agent(callback_context) -> None:
    """Attach deterministic quality scores to the current trace in Langfuse."""
    client = langfuse_client()
    if client is None:
        return None
    trace_id = _current_trace_id()
    if not trace_id:
        print("[observability] no active trace id — skipping scores")
        return None

    if _was_cache_hit(callback_context):
        # The response was replayed from the vault, not generated. Scoring it
        # would report the same text as N fresh outputs and bias the quality
        # charts; the cache outcome is already recorded by report_cache_outcome.
        return None

    user_text = _last_session_user_text(callback_context)
    model_text = _last_session_model_text(callback_context)
    for name, value in _score_generation(user_text, model_text).items():
        try:
            client.create_score(
                trace_id=trace_id,
                name=f"quality.{name}",
                value=value,
                data_type="NUMERIC",
            )
        except Exception as exc:  # pragma: no cover - observability must never break the agent
            print(f"[observability] score '{name}' failed: {exc}")

    rouge1 = response_match_for_agent(user_text, model_text)
    if rouge1 is not None:
        try:
            client.create_score(
                trace_id=trace_id,
                name="response_match_score",
                value=rouge1,
                data_type="NUMERIC",
            )
        except Exception as exc:  # pragma: no cover - observability must never break the agent
            print(f"[observability] score 'response_match_score' failed: {exc}")
    return None


def _was_cache_hit(callback_context) -> bool:
    """True when this turn's answer was served from the vault rather than the model."""
    try:
        return bool(callback_context.state.get(CACHE_HIT_STATE_KEY))
    except Exception:  # pragma: no cover
        return False


def _part_texts(content) -> list[str]:
    if content is None:
        return []
    parts = getattr(content, "parts", None) or []
    return [str(p.text) for p in parts if getattr(p, "text", None) is not None]


def _session_events(callback_context) -> list:
    return list((callback_context.session.events or []) if callback_context.session else [])


def _last_session_user_text(callback_context) -> str:
    for event in reversed(_session_events(callback_context)):
        if getattr(event, "author", "") == "user":
            return "".join(_part_texts(getattr(event, "content", None)))
    return ""


def _last_session_model_text(callback_context) -> str:
    for event in reversed(_session_events(callback_context)):
        if getattr(event, "author", "") == "user":
            continue
        text = "".join(_part_texts(getattr(event, "content", None)))
        if text.strip():
            return text
    return ""


root_agent = LlmAgent(
    name="text_summarizer",
    model=model,
    description="A text summarization agent that converts long text into concise bullet-point summaries, stores them in an Obsidian vault (second brain), and can read the user's Gmail inbox.",
    before_model_callback=cache_hit_before_model,
    # Names the Langfuse trace and attaches userId/sessionId while the agent_run span
    # is still open, so traces are identifiable instead of listing as blank rows.
    before_agent_callback=tag_trace_identity,
    # after_model (not before_model) so the prompt name lands on the still-open
    # generation span -- OTel drops attributes set on an ended span, and the
    # dashboards group by prompt name. The Sources block is substituted here
    # too, and for the same reason: an altered LlmResponse returned from
    # after_model_callback *replaces* the model's response, so the answer is one
    # event. after_agent_callback cannot do that (it appends a second event, and
    # the dev UI then shows the answer twice) -- see render_sources_after_model.
    after_model_callback=[tag_current_span, render_sources_after_model],
    after_agent_callback=report_scores_after_agent,
    instruction="""You are a text summarization agent backed by an Obsidian vault that acts as a second brain. Your job is to take long text provided by the user, convert it into a short, clear bullet-point summary, and persist it in the vault so the knowledge is graph-aware and reusable.

Rules:
1. Always respond with bullet points using the - prefix (dash followed by space).
2. Each bullet point should be a single concise sentence.
3. Capture the main ideas by closely mirroring the key terms, phrasing, and sentence structures found in the source text.
4. Aim for 3-5 bullet points depending on the length and complexity of the input.
5. Do not add information that is not present in the original text.
6. Use clear, professional language, maintaining a direct, factual tone that reflects the core statements of the input.
7. USE THE SECOND BRAIN - RETRIEVE FIRST: Before summarizing, identify the 1-3 main topics of the input. Use the vault search tools (search_text) and note_read to look up existing notes on those topics and on the Second Brain Index. If relevant related notes exist, list them at the very END of your answer, one note per line, copying this template EXACTLY: [obsidian][@@ADK_VAULT@@][@@ADK_MODEL@@][[Exact Note Title]]: short reason this note is relevant. The two tokens @@ADK_VAULT@@ and @@ADK_MODEL@@ are placeholders that a later step replaces with the real vault and model names - copy them verbatim and NEVER write a real vault or model name there yourself. Do NOT write a heading of any kind (no "Sources", no "## Sources", no "**Sources**"), do NOT add a bullet or number in front of the lines, and do not write anything after the last source line - the heading, the bullets and the real identifiers are added for you afterwards. If nothing relevant exists, list nothing at all.
8. ALWAYS persist the summary: after producing the summary, call save_summary_to_second_brain with a short descriptive title, the bullet-point content, and a comma-separated list of the 2-5 main topics it covers. This keeps the vault graph connected and enables zero-cost dedup on repeated requests. Pass ONLY those three arguments - do not repeat the user's text back in any argument; the tool records the request itself for deduplication, and echoing the text back wastes a large number of output tokens on every call.
9. ALWAYS log the conversation: after producing and persisting your final answer, call log_conversation with the user's exact message and your final answer, so every exchange is recorded in the vault's chat log for later recall.
10. Mention in your final answer that the summary was saved to the second brain and its title.
11. When the user asks about their emails, use the Gmail tools: gmail_search or gmail_get_latest_messages to find relevant messages, then gmail_read or gmail_get_thread to read full contents. Summarize what you find as bullet points using the rules above. Never invent email content - only report what the tools actually return.""",
    tools=[
        FunctionTool(save_summary_to_second_brain),
        FunctionTool(log_conversation),
        *build_obsidian_tools(),
        *build_gmail_tools(),
    ],
)