"""Unit tests for the ``after_agent_callback`` that injects Sources provenance.

These cover the contract between ``agent.py`` and ADK, without an LLM call:

* the callback returns rewritten content for a fresh generation, so ADK emits
  the substituted response (``BaseAgent._handle_after_agent_callback``);
* it returns ``None`` on a cache hit, so a replayed note is served verbatim;
* the served model comes from ``Event.model_version`` on the session's last
  complete model event, and the callback is a no-op without one.

``report_scores_after_agent`` is exercised with a fake callback context rather
than a live runner; the scoring half is guarded by Langfuse being absent, which
is the same no-op path a real unconfigured deployment takes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from text_summarizer import agent as agent_module
from text_summarizer.sources import MODEL_TOKEN, SOURCES_HEADING, VAULT_TOKEN

BULLETS = "- Dogs are social animals.\n- Dogs vary in size and temperament."


def _event(text="", *, author="text_summarizer", model_version=None):
    return SimpleNamespace(
        author=author,
        content=SimpleNamespace(
            parts=[SimpleNamespace(text=text, function_response=None)]
        ),
        model_version=model_version,
        partial=False,
    )


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


def _rendered_text(content):
    return "".join(part.text for part in content.parts)


# --- fresh generation --------------------------------------------------------


def test_fresh_generation_returns_rewritten_content(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [
        _event("summarize the dogs text", author="user"),
        _event(_model_text("Dogs Overview"), model_version="gemini-3.5-flash-lite"),
    ]
    result = agent_module.report_scores_after_agent(_context(events))
    assert result is not None
    assert result.role == "model"
    text = _rendered_text(result)
    assert SOURCES_HEADING in text
    assert (
        "[obsidian][ck][gemini-3.5-flash-lite][[Dogs Overview]]: related" in text
    )
    assert VAULT_TOKEN not in text and MODEL_TOKEN not in text
    # The rest of the answer is untouched.
    assert text.startswith(BULLETS)
    assert 'Saved to the second brain as "Dogs Summary".' in text


def test_multi_source_block_gets_one_line_per_note(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [
        _event(_model_text("Dogs Overview", "Wolves", "Second Brain Index"),
               model_version="openrouter/qwen/qwen3.8-27b:free"),
    ]
    text = _rendered_text(agent_module.report_scores_after_agent(_context(events)))
    assert text.count(SOURCES_HEADING) == 1
    for note in ("Dogs Overview", "Wolves", "Second Brain Index"):
        assert f"[[{note}]]" in text
    assert "openrouter/qwen/qwen3.8-27b:free" in text


def test_missing_model_provenance_renders_as_unknown(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [_event(_model_text("Dogs Overview"), model_version=None)]
    text = _rendered_text(agent_module.report_scores_after_agent(_context(events)))
    assert "[obsidian][ck][unknown][[Dogs Overview]]" in text


def test_response_without_a_sources_block_is_left_alone(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [
        _event(f"{BULLETS}\n\nSaved to the second brain as \"Dogs Summary\".",
               model_version="gemini-3.5-flash-lite"),
    ]
    assert agent_module.report_scores_after_agent(_context(events)) is None


def test_empty_response_is_left_alone(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    assert agent_module.report_scores_after_agent(_context([_event("")])) is None


# --- cache hits must replay verbatim ----------------------------------------


def test_cache_hit_returns_none_so_the_cached_note_replays_verbatim(monkeypatch):
    """A stored note keeps whatever shape it was saved with -- no re-rendering."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    cached = (
        f"{BULLETS}\n\n## From the vault\n"
        "[[2026-09-25 - characteristics-and-purpose-of-dogs-summary]]"
    )
    events = [_event(cached, model_version=None)]
    assert (
        agent_module.report_scores_after_agent(_context(events, cache_hit=True))
        is None
    )


def test_cache_hit_is_not_rewritten_even_with_a_model_event(monkeypatch):
    monkeypatch.setenv("VAULT_NAME", "ck")
    cached = f"{BULLETS}\n\n## From the vault\n[[Dogs Overview]]"
    events = [_event(cached, model_version="gemini-3.5-flash-lite")]
    assert (
        agent_module.report_scores_after_agent(_context(events, cache_hit=True))
        is None
    )


def test_scoring_is_still_skipped_on_a_cache_hit(monkeypatch):
    scored: list[str] = []

    class _Client:
        def create_score(self, *, trace_id, name, value, data_type):
            scored.append(name)

    monkeypatch.setattr(agent_module, "langfuse_client", lambda: _Client())
    monkeypatch.setattr(agent_module, "_current_trace_id", lambda: "0" * 32)
    events = [_event(f"{BULLETS}\n\n## From the vault\n[[Dogs Overview]]")]
    agent_module.report_scores_after_agent(_context(events, cache_hit=True))
    assert scored == []


# --- scoring side effects survive the rewrite --------------------------------


def test_quality_scores_ignore_source_lines(monkeypatch):
    """Source lines are not summary bullets, so bullet_count stays accurate."""
    scored: dict[str, float] = {}

    class _Client:
        def create_score(self, *, trace_id, name, value, data_type):
            scored[name] = value

    monkeypatch.setattr(agent_module, "langfuse_client", lambda: _Client())
    monkeypatch.setattr(agent_module, "_current_trace_id", lambda: "0" * 32)
    monkeypatch.setenv("VAULT_NAME", "ck")
    events = [_event(_model_text("Dogs Overview"), model_version="m1")]
    agent_module.report_scores_after_agent(_context(events))
    # Two summary bullets, so the raw count is 2 -- not 3.
    assert scored["quality.bullet_count"] == 2
    assert "quality.fidelity" in scored


def test_a_broken_renderer_still_scores(monkeypatch):
    scored: list[str] = []

    class _Client:
        def create_score(self, *, trace_id, name, value, data_type):
            scored.append(name)

    monkeypatch.setattr(agent_module, "langfuse_client", lambda: _Client())
    monkeypatch.setattr(agent_module, "_current_trace_id", lambda: "0" * 32)
    monkeypatch.setattr(
        agent_module,
        "render_sources",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    events = [_event(_model_text("Dogs Overview"), model_version="m1")]
    assert agent_module.report_scores_after_agent(_context(events)) is None
    assert "quality.bullet_count" in scored


# --- vault identity in the callback ------------------------------------------


def test_vault_name_comes_from_the_vault_root_when_unset(monkeypatch, tmp_path):
    from text_summarizer import second_brain

    vault = tmp_path / "ck"
    vault.mkdir()
    monkeypatch.delenv("VAULT_NAME", raising=False)
    monkeypatch.setenv("SECOND_BRAIN_VAULT", str(vault))
    monkeypatch.setattr(second_brain, "VAULT_ROOT", str(vault))
    events = [_event(_model_text("Dogs Overview"), model_version="m1")]
    text = _rendered_text(agent_module.report_scores_after_agent(_context(events)))
    assert "[obsidian][ck][m1]" in text


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
    events = [
        SimpleNamespace(
            author="text_summarizer",
            content=SimpleNamespace(parts=[info]),
            model_version=None,
            partial=False,
        ),
        _event(_model_text("Dogs Overview"), model_version="m1"),
    ]
    text = _rendered_text(agent_module.report_scores_after_agent(_context(events)))
    assert "[obsidian][ck][m1]" in text


# --- instruction contract ----------------------------------------------------


def test_instruction_asks_for_sentinels_and_forbids_a_heading():
    instruction = agent_module.root_agent.instruction
    assert VAULT_TOKEN in instruction
    assert MODEL_TOKEN in instruction
    assert "no \"**Sources**\"" in instruction
    # The old heading-emitting wording is gone.
    assert '"## Sources" section' not in instruction


@pytest.mark.parametrize("field", ["before_model_callback", "after_agent_callback"])
def test_callbacks_are_registered(field):
    assert callable(getattr(agent_module.root_agent, field))


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

    assert agent_module._user_content_text(context) == "Summarize: the real prompt."


def test_cache_key_falls_back_to_the_request_when_there_is_no_user_content():
    context = _cache_context()
    context.user_content = None
    request = _llm_request(["Summarize: dogs."])
    assert agent_module._user_content_text(context) == ""


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
