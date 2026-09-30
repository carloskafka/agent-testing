"""Unit tests for the callbacks that inject Sources provenance and score a turn.

These cover the contract between ``agent.py`` and ADK, without an LLM call:

* ``render_sources_after_model`` (``after_model_callback``) returns an altered
  ``LlmResponse`` for a fresh generation, so the substituted text *replaces* the
  model's own response and the answer exists on exactly one event;
* ``report_scores_after_agent`` (``after_agent_callback``) scores the turn and
  returns ``None`` -- content returned there would become a second event, and
  the dev UI would then show the same answer twice;
* the served model comes from ``LlmResponse.model_version``, and the callback is
  a no-op on a response with nothing to substitute.

``report_scores_after_agent`` is exercised with a fake callback context rather
than a live runner; the scoring half is guarded by Langfuse being absent, which
is the same no-op path a real unconfigured deployment takes.
"""

from __future__ import annotations

from types import SimpleNamespace

from _helpers import _Client, _event, _point_vault_at
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part
from text_summarizer import agent as agent_module
from text_summarizer.agent import render_sources_after_model
from text_summarizer.observability import tag_current_span
from text_summarizer.sources import MODEL_TOKEN, SOURCES_HEADING, VAULT_TOKEN

BULLETS = "- Dogs are social animals.\n- Dogs vary in size and temperament."


def _context(events, *, cache_hit=False):
    return SimpleNamespace(
        state={agent_module.CACHE_HIT_STATE_KEY: cache_hit},
        session=SimpleNamespace(events=list(events)),
    )


def _sources_text(*notes):
    return "\n".join(
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[{note}]]: related"
        for note in notes
    )


def _model_text(*notes, tail='\n\nSaved to the second brain as "Dogs Summary".'):
    return f"{BULLETS}\n\n{_sources_text(*notes)}{tail}"


def _llm_response(text, *, model_version="gemini-3.5-flash-lite", partial=False):
    return LlmResponse(
        content=Content(role="model", parts=[Part(text=text)]),
        model_version=model_version,
        partial=partial,
    )


def _rendered_text(response) -> str:
    """The text of an altered LlmResponse, or '' when nothing was substituted."""
    if response is None:
        return ""
    return "".join(part.text for part in response.content.parts or [])


def _render(context, text, *, model_version="gemini-3.5-flash-lite", partial=False):
    return agent_module.render_sources_after_model(
        context, _llm_response(text, model_version=model_version, partial=partial)
    )


# --- fresh generation --------------------------------------------------------


def test_fresh_generation_returns_an_altered_response(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [
        _event("summarize the dogs text", author="user"),
        _event(_model_text("Dogs Overview"), model_version="gemini-3.5-flash-lite"),
    ]
    result = _render(_context(events), _model_text("Dogs Overview"))
    assert result is not None
    assert isinstance(result, LlmResponse)
    text = _rendered_text(result)
    assert SOURCES_HEADING in text
    assert (
        "- [obsidian][ck][gemini-3.5-flash-lite][[Dogs Overview]]: related" in text
    )
    assert VAULT_TOKEN not in text and MODEL_TOKEN not in text
    # The rest of the answer is untouched.
    assert text.startswith(BULLETS)
    assert 'Saved to the second brain as "Dogs Summary".' in text


def test_the_altered_response_keeps_the_model_and_its_provenance(monkeypatch):
    """The copy is the model's response with the text swapped, not a new one.

    ``_finalize_model_response_event`` merges the returned response into the
    event, so anything dropped here would vanish from the session and from
    Langfuse along with it.
    """
    monkeypatch.setenv("VAULT_NAME", "ck")
    original = LlmResponse(
        content=Content(role="model", parts=[Part(text=_model_text("Dogs Overview"))]),
        model_version="gemini-3.5-flash-lite",
        turn_complete=True,
        finish_reason="STOP",
    )
    result = agent_module.render_sources_after_model(_context([]), original)
    assert result is not None
    assert result.model_version == "gemini-3.5-flash-lite"
    assert result.turn_complete is True
    assert result.finish_reason == "STOP"


def test_multi_source_block_gets_one_line_per_note(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [
        _event(
            _model_text("Dogs Overview", "Wolves", "Second Brain Index"),
            model_version="openrouter/qwen/qwen3.8-27b:free",
        ),
    ]
    text = _rendered_text(
        _render(
            _context(events),
            _model_text("Dogs Overview", "Wolves", "Second Brain Index"),
            model_version="openrouter/qwen/qwen3.8-27b:free",
        )
    )
    assert text.count(SOURCES_HEADING) == 1
    for note in ("Dogs Overview", "Wolves", "Second Brain Index"):
        assert f"- [obsidian][ck][openrouter/qwen/qwen3.8-27b:free][[{note}]]: related" in text
    assert "openrouter/qwen/qwen3.8-27b:free" in text


def test_missing_model_provenance_renders_as_unknown(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [_event(_model_text("Dogs Overview"), model_version=None)]
    text = _rendered_text(
        _render(_context(events), _model_text("Dogs Overview"), model_version=None)
    )
    assert "- [obsidian][ck][unknown][[Dogs Overview]]: related" in text


def test_response_without_a_sources_block_is_left_alone(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [
        _event(
            f"{BULLETS}\n\nSaved to the second brain as \"Dogs Summary\".",
            model_version="gemini-3.5-flash-lite",
        ),
    ]
    assert _render(_context(events), f"{BULLETS}\n\nSaved to the second brain.") is None


def test_empty_response_is_left_alone(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    assert _render(_context([_event("")]), "") is None


def test_a_tool_calling_response_is_left_alone(monkeypatch):
    """A model call that asks for a tool carries no text to substitute."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    call = LlmResponse(
        content=Content(
            role="model",
            parts=[Part(function_call={"name": "search_text", "args": {}})],
        ),
        model_version="gemini-3.5-flash-lite",
    )
    assert agent_module.render_sources_after_model(_context([]), call) is None


def test_a_streamed_chunk_is_left_alone(monkeypatch):
    """A partial response is not the whole answer, and a token can straddle chunks.

    `adk web` does not stream, so this only guards the SSE path.
    """
    monkeypatch.setenv("VAULT_NAME", "ck")
    assert _render(_context([]), _model_text("Dogs Overview"), partial=True) is None


# --- the answer must exist exactly once --------------------------------------


def test_after_agent_callback_never_returns_content(monkeypatch):
    """The regression: the answer was rendered twice, at the end of the turn.

    Content returned from ``after_agent_callback`` does not replace the agent's
    response -- ADK builds an extra ``Event`` for it
    (``BaseAgent._handle_after_agent_callback``). Both are assistant-authored, so
    the dev UI drew the model's raw text and then the substituted copy: session
    ``55b1f828-6ff0-498e-953f-8cc29b84cb93`` ends with two text events carrying
    the same answer. Rewriting happens in ``render_sources_after_model`` instead,
    which replaces the response on the model's own event.
    """
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [_event(_model_text("Dogs Overview"), model_version="m1")]
    assert agent_module.report_scores_after_agent(_context(events)) is None


def test_the_substitution_happens_in_the_model_response(monkeypatch):
    """What replaces the response is an LlmResponse, not a Content.

    An ``after_model_callback`` may only return an ``LlmResponse``; returning a
    ``Content`` here is not how the answer gets rewritten at all.
    """
    monkeypatch.setenv("VAULT_NAME", "ck")
    result = _render(_context([]), _model_text("Dogs Overview"))
    assert isinstance(result, LlmResponse)
    assert not isinstance(result, Content)


# --- cache hits must replay verbatim ----------------------------------------
#
# A cache hit short-circuits in ``before_model_callback``, so the model is never
# called and ``after_model_callback`` never fires: a stored note keeps whatever
# shape it was saved with. Pinned against a real runner in test_adk_wiring.py.


def test_scoring_is_skipped_on_a_cache_hit(monkeypatch):
    client = _Client()
    monkeypatch.setattr(agent_module, "langfuse_client", lambda: client)
    monkeypatch.setattr(agent_module, "_current_trace_id", lambda: "0" * 32)
    events = [_event(f"{BULLETS}\n\n## From the vault\n[[Dogs Overview]]")]
    agent_module.report_scores_after_agent(_context(events, cache_hit=True))
    assert client.scores == {}


# --- scoring side effects ----------------------------------------------------


def test_quality_scores_ignore_source_lines(monkeypatch):
    """Source lines are not summary bullets, so bullet_count stays accurate."""
    client = _Client()
    monkeypatch.setattr(agent_module, "langfuse_client", lambda: client)
    monkeypatch.setattr(agent_module, "_current_trace_id", lambda: "0" * 32)
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [_event(_model_text("Dogs Overview"), model_version="m1")]
    agent_module.report_scores_after_agent(_context(events))
    # Two summary bullets, so the raw count is 2 -- not 3, even though every
    # source line is a bullet of its own.
    assert client.scores["quality.bullet_count"] == 2
    assert "quality.fidelity" in client.scores


def test_a_broken_renderer_leaves_the_response_untouched(monkeypatch):
    """Rendering is cosmetic: a failure must not take the turn down with it."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    monkeypatch.setattr(
        agent_module,
        "render_sources",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _render(_context([]), _model_text("Dogs Overview")) is None


# --- vault identity in the callback ------------------------------------------


def test_vault_name_comes_from_the_vault_root_when_unset(monkeypatch, tmp_path):
    _point_vault_at(monkeypatch, tmp_path / "ck", name="", modules=("agent",))
    text = _rendered_text(_render(_context([]), _model_text("Dogs Overview")))
    assert "- [obsidian][ck][gemini-3.5-flash-lite][[Dogs Overview]]: related" in text


def test_response_names_the_vault_not_the_mounted_parent(monkeypatch, tmp_path):
    """The Sources block must name the vault, not the parent directory holding it.

    This is the shape Docker actually runs: ``SECOND_BRAIN_VAULT`` is the parent
    that gets bind-mounted (``/vaults``) and the active vault is its single child
    (``/vaults/ck``). Every other test in this module sets ``VAULT_NAME`` instead,
    which is a resolution step the compose file deliberately never takes, so the
    path production uses was covered by nothing -- and rendering the parent's
    basename is a wrong answer that looks right.
    """
    parent = tmp_path / "vaults"
    _point_vault_at(
        monkeypatch, parent, env_root=parent, modules=("agent",)
    )
    text = _rendered_text(_render(_context([]), _model_text("Dogs Overview")))
    assert "- [obsidian][ck][gemini-3.5-flash-lite]" in text
    assert f"[{parent.name}]" not in text


def test_response_and_note_agree_on_the_vault_name(monkeypatch, tmp_path):
    """The frontmatter and the response are written by different call sites.

    ``second_brain.note_provenance`` reads ``second_brain.VAULT_ROOT`` and
    ``agent._render_sources_text`` reads the name imported into ``agent``, so
    they can drift apart; a real recorded turn put ``ck`` in the note and
    ``vaults`` in the answer. They are the same value in a real deployment, so
    pin them to the same one here.
    """
    from text_summarizer import second_brain

    parent = tmp_path / "vaults"
    _point_vault_at(monkeypatch, parent, env_root=parent)
    from_note = second_brain.note_provenance(None)[second_brain.GENERATED_IN_VAULT_KEY]
    text = _rendered_text(_render(_context([]), _model_text("Dogs Overview")))
    assert from_note == "ck"
    assert f"- [obsidian][{from_note}][gemini-3.5-flash-lite]" in text


def test_vault_name_comes_from_a_mcp_vault_info_call(monkeypatch):
    """An opportunistic `vault_info` result is used when nothing is configured."""
    monkeypatch.delenv("VAULT_NAME", raising=False)
    monkeypatch.setenv("SECOND_BRAIN_VAULT", "/vault")
    info = SimpleNamespace(
        text=None,
        function_response=SimpleNamespace(
            name="vault_info",
            response={"result": '{"vault_name": "ck", "vault_path": "/vault"}'},
        ),
    )
    events = [_event(parts=[info]), _event(_model_text("Dogs Overview"), model_version="m1")]
    text = _rendered_text(_render(_context(events), _model_text("Dogs Overview")))
    assert "- [obsidian][ck][gemini-3.5-flash-lite]" in text


# --- instruction contract ----------------------------------------------------


def test_instruction_asks_for_sentinels_and_forbids_a_heading():
    instruction = agent_module.root_agent.instruction
    assert VAULT_TOKEN in instruction
    assert MODEL_TOKEN in instruction
    assert 'no "**Sources**"' in instruction
    # The old heading-emitting wording is gone.
    assert '"## Sources" section' not in instruction


def test_callbacks_are_registered():
    agent = agent_module.root_agent
    assert callable(agent.before_model_callback)
    assert callable(agent.after_agent_callback)
    # A list, because the model callback is where the Sources block is rewritten.
    assert [tag_current_span, render_sources_after_model] == list(
        agent.after_model_callback
    )


# --- cache key and the once-per-turn guard ------------------------------------
#
# Both of these were found by running the agent for real under `adk web`, not by
# the in-memory tests above, which pass either way. Each is pinned here against
# the shape the live runner actually produces.

def _cache_context(*, user_text="Summarize: dogs.", invocation_id="inv-1", events=None):
    """A CallbackContext shaped like the live one.

    ``events`` defaults to empty on purpose: a database-backed session service
    (the default under ``adk web``) does not populate ``session.events``, which is
    what made a positional "is this the first model call" check silently always
    answer yes.
    """
    return SimpleNamespace(
        state={},
        session=SimpleNamespace(events=list(events or [])),
        user_content=SimpleNamespace(
            parts=[SimpleNamespace(text=user_text, function_response=None)]
        ),
        invocation_id=invocation_id,
    )


def _llm_request(*user_role_texts):
    return SimpleNamespace(
        contents=[
            SimpleNamespace(
                role="user",
                parts=[SimpleNamespace(text=t, function_response=None) for t in texts],
            )
            for texts in user_role_texts
        ]
    )


def test_cache_key_comes_from_user_content_not_the_request_contents():
    """ADK gives a tool result role="user", so the request contents lie.

    On any follow-up model call the last user-role part of ``llm_request.contents``
    is the tool's JSON payload, not the prompt. Keying on it turns every mid-turn
    lookup into a miss on a key nothing will ever match.
    """
    context = _cache_context(user_text="Summarize: the real prompt.")
    request = _llm_request(["Summarize: the real prompt."])
    # A function response is role "user" and is the LAST such part.
    request.contents.append(
        SimpleNamespace(
            role="user",
            parts=[SimpleNamespace(text=None, function_response=SimpleNamespace(name="t"))],
        )
    )

    # The trap, stated as an assertion rather than left implied by the construction:
    # the last user-role part is the tool payload, and it carries no prompt text at
    # all -- so anything that fingerprinted `request.contents[-1]` would key the
    # cache on the tool's return value.
    last = request.contents[-1].parts[-1]
    assert last.text is None and last.function_response is not None
    assert agent_module._user_content_text(context) == "Summarize: the real prompt."


def test_no_user_content_yields_an_empty_key_and_therefore_a_miss():
    """An unavailable ``user_content`` must not be papered over from the request.

    The previous name of this test claimed a fallback that lives one level up, in
    ``cache_hit_before_model`` (``_user_content_text(context) or
    _last_user_text(llm_request)``) -- and it built a request, then never asserted
    on it, so nothing said which of the two was actually consulted.

    What is pinned here is the part that is load-bearing: ``_user_content_text`` is
    empty, and an empty key is a *miss*, not a lookup of the empty string. A miss
    re-runs the model, which is the safe direction -- a wrong hit would replay an
    unrelated note. ``conftest`` pins ``CACHE_ENABLED`` at import, so the callback is
    already live here.
    """
    context = _cache_context()
    context.user_content = None
    request = _llm_request(["Summarize: dogs."])

    assert agent_module._user_content_text(context) == ""
    # The request is not read by the key helper, and the callback short-circuits on
    # the empty key: a miss is recorded and the model is allowed to run.
    assert agent_module.cache_hit_before_model(context, request) is None
    assert context.state[agent_module.CACHE_HIT_STATE_KEY] is False


def test_lookup_runs_once_per_invocation_not_once_per_model_call():
    """The guard that stops a turn replaying the note it just wrote."""
    context = _cache_context(invocation_id="inv-1")
    assert agent_module._first_model_call_of_invocation(context) is True
    assert agent_module._first_model_call_of_invocation(context) is False
    assert agent_module._first_model_call_of_invocation(context) is False

    # Next turn: a new invocation id, so the cache is eligible again.
    context.invocation_id = "inv-2"
    assert agent_module._first_model_call_of_invocation(context) is True


def test_the_guard_works_when_session_events_are_not_populated():
    """The live failure: an empty event list must not read as "first call" forever."""
    context = _cache_context(events=[])
    assert agent_module._first_model_call_of_invocation(context) is True
    assert agent_module._first_model_call_of_invocation(context) is False


def test_the_guard_does_not_block_when_there_is_no_invocation_id():
    """Degrade to the old behaviour rather than silently never consulting the cache."""
    context = _cache_context(invocation_id=None)
    assert agent_module._first_model_call_of_invocation(context) is True
    assert agent_module._first_model_call_of_invocation(context) is True


def test_a_cache_hit_is_not_overwritten_by_a_later_call_in_the_same_turn():
    """A follow-up call must not report a miss over a genuine hit.

    ``_mark_cache`` writes the state key on every call it reports, so anything that
    reports after a hit would clear it -- and the turn would then be scored and
    its Sources block rewritten as if freshly generated. Follow-up calls return
    before reporting anything, which is what keeps the flag intact.
    """
    context = _cache_context()
    request = _llm_request(["Summarize: dogs."])

    # First model call of the turn: records the invocation, reports the miss.
    assert agent_module.cache_hit_before_model(context, request) is None
    assert context.state[agent_module.CACHE_CHECKED_INVOCATION_KEY] == "inv-1"

    # As if it had hit instead.
    agent_module._mark_cache(context, enabled=True, hit=True, elapsed_ms=0.1)
    assert context.state[agent_module.CACHE_HIT_STATE_KEY] is True

    # A later model call in the same turn must leave that alone.
    assert agent_module.cache_hit_before_model(context, request) is None
    assert context.state[agent_module.CACHE_HIT_STATE_KEY] is True

    # The next turn is eligible again, and a miss there legitimately clears it.
    context.invocation_id = "inv-2"
    assert agent_module.cache_hit_before_model(context, request) is None
    assert context.state[agent_module.CACHE_HIT_STATE_KEY] is False
