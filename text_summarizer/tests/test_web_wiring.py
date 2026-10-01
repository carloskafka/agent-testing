"""Wiring: the web tier reaches the answer only through the permitted path.

These are the tests that would catch a *miswiring*, which is the failure mode this
feature is most exposed to. The renderer can be perfect and the tools can be
correct, and the feature still misbehaves if:

* the ``after_model_callback`` list grows a third entry (silently skipped on
  exactly the turns that carry a Sources block -- see
  ``_callback_pipeline._stop_on_truthy``);
* the permitted-URL set is not threaded into ``render_sources``;
* the URL harvester reads a session that ``adk web`` does not populate;
* a function call riding on the same response loses it.
"""

from __future__ import annotations

import json

from text_summarizer import agent
from text_summarizer.sources import MODEL_TOKEN, VAULT_TOKEN, WEB_TOKEN

# --- the callback list is still exactly two entries ---------------------------


def test_after_model_callback_is_still_two_entries():
    """A third entry would be silently skipped on every web-cited turn.

    ``after_model_callback`` stops at the first callback that returns a response
    (``google/adk/utils/_callback_pipeline.py:94-101``, ``_stop_on_truthy`` at
    ``:116``). ``finalize_answer_after_model`` returns a response on exactly the
    turns that carry a Sources block, so anything appended after it would run on
    uncited turns only -- looking correct everywhere it was not needed.
    """
    assert list(agent.root_agent.after_model_callback) == [
        agent.tag_current_span,
        agent.finalize_answer_after_model,
    ]


def test_the_tools_are_present_and_gated():
    """``build_web_search_tools()`` is spread unconditionally, like the others."""
    assert callable(agent.build_web_search_tools)
    toolset = agent.build_web_search_tools()
    assert isinstance(toolset, list)
    names = [getattr(t, "name", "") for t in toolset]
    assert names in ([], ["web_search", "web_fetch"])


def test_after_agent_callback_returns_no_content():
    """Gotcha 16: content returned here becomes a *second* event."""
    assert agent.report_scores_after_agent(None, None) is None


# --- the permission set -------------------------------------------------------


class _FakeState(dict):
    pass


class _FakeContext:
    """The minimum surface ``_web_urls_this_turn`` and the harvester touch."""

    def __init__(self, state=None, events=None, invocation_id="inv-1"):
        self.state = _FakeState(state or {})
        self.invocation_id = invocation_id
        self.session = type("S", (), {"events": events or []})()


def test_no_urls_means_no_web_citations():
    assert agent._web_urls_this_turn(_FakeContext()) == set()


def test_urls_are_read_for_the_matching_invocation():
    ctx = _FakeContext({agent.WEB_URLS_STATE_KEY: {"inv-1": ["https://example.com/p"]}})
    assert agent._web_urls_this_turn(ctx) == {"https://example.com/p"}


def test_a_later_invocation_does_not_inherit_the_previous_turn_urls():
    """Keyed by invocation id precisely so a stale citation cannot survive a turn."""
    state = {agent.WEB_URLS_STATE_KEY: {"inv-1": ["https://example.com/p"]}}
    assert agent._web_urls_this_turn(_FakeContext(state, invocation_id="inv-2")) == set()


def test_missing_invocation_id_yields_nothing():
    ctx = _FakeContext({agent.WEB_URLS_STATE_KEY: {"inv-1": ["https://example.com/p"]}})
    ctx.invocation_id = None
    assert agent._web_urls_this_turn(ctx) == set()


def test_a_broken_state_store_yields_nothing_rather_than_raising():
    class Hostile:
        def get(self, key, default=None):
            raise RuntimeError("state unavailable")

        def __setitem__(self, key, value):
            raise RuntimeError("state unavailable")

    ctx = _FakeContext()
    ctx.state = Hostile()
    assert agent._web_urls_this_turn(ctx) == set()
    agent.record_web_urls(ctx)  # must not raise


# --- the harvester ------------------------------------------------------------


def test_harvests_urls_from_a_json_encoded_tool_result():
    """``FunctionTool`` returns whatever the tool returned; the agent stores JSON."""
    payload = json.dumps([{"title": "OpenCode", "url": "https://opencode.ai/"}])
    urls: set[str] = set()
    agent._harvest_web_urls(payload, urls)
    assert "https://opencode.ai/" in urls


def test_harvests_urls_from_a_dict_result():
    urls: set[str] = set()
    agent._harvest_web_urls(
        {"url": "https://example.com/p", "nested": {"url": "https://example.com/q"}}, urls
    )
    assert urls == {"https://example.com/p", "https://example.com/q"}


def test_harvests_urls_from_nested_wrappers():
    urls: set[str] = set()
    agent._harvest_web_urls(
        {"function_response": {"payload": {"link": "https://deep.example/x"}}}, urls
    )
    assert "https://deep.example/x" in urls


def test_non_http_urls_are_not_collected():
    """Only the schemes ``search_web`` hands out may enter the permitted set."""
    urls: set[str] = set()
    agent._harvest_web_urls(
        '{"a":"javascript:alert(1)","b":"file:///etc/passwd","c":"https://ok.example/"}', urls
    )
    assert urls == {"https://ok.example/"}


def test_trailing_punctuation_is_trimmed():
    urls: set[str] = set()
    agent._harvest_web_urls("see https://example.com/p.", urls)
    assert urls == {"https://example.com/p"}


def test_harvest_stops_at_the_depth_cap():
    """A pathological payload must not recurse without bound."""
    payload: dict = {"url": "https://deep.example/"}
    for _ in range(20):
        payload = {"next": payload}
    urls: set[str] = set()
    agent._harvest_web_urls(payload, urls)
    assert urls == set()


def test_record_web_urls_is_a_no_op_without_configuration(monkeypatch):
    monkeypatch.setattr(agent, "searxng_url", lambda: "")
    ctx = _FakeContext()
    agent.record_web_urls(ctx)
    assert agent.WEB_URLS_STATE_KEY not in ctx.state


def test_record_web_urls_reads_only_web_tool_responses(monkeypatch):
    """A vault note path in another tool's result is not a citable web URL."""
    monkeypatch.setattr(agent, "searxng_url", lambda: "http://searxng:8080")

    class _Resp:
        def __init__(self, name, response):
            self.name = name
            self.response = response

    class _Event:
        def __init__(self, responses):
            self.function_responses = responses

    events = [
        _Event([_Resp("save_summary_to_second_brain", "https://vault.example/note.md")]),
        _Event([_Resp("web_search", json.dumps([{"url": "https://real.example/p"}]))]),
    ]
    ctx = _FakeContext(events=events)
    agent.record_web_urls(ctx)
    stored = ctx.state.get(agent.WEB_URLS_STATE_KEY, {}).get("inv-1", [])
    assert stored == ["https://real.example/p"]


def test_record_web_urls_survives_an_unreadable_session():
    class Hostile:
        @property
        def events(self):
            raise RuntimeError("session unavailable")

    ctx = _FakeContext()
    ctx.session = Hostile()
    agent.record_web_urls(ctx)  # must not raise


# --- rendering end to end through the callback --------------------------------


class _Resp:
    """Minimal stand-in for the function-call check."""

    def __init__(self, text, function_call=False):
        self.content = type(
            "C",
            (),
            {
                "parts": [
                    type(
                        "P",
                        (),
                        {"text": text, "function_call": ({"name": "x"} if function_call else None)},
                    )()
                ]
            },
        )()


def test_a_web_line_the_search_returned_is_rendered(monkeypatch):
    monkeypatch.setattr(agent, "searxng_url", lambda: "http://searxng:8080")
    ctx = _FakeContext({agent.WEB_URLS_STATE_KEY: {"inv-1": ["https://example.com/p"]}})
    text = f"- a bullet\n[web][{WEB_TOKEN}][{MODEL_TOKEN}]<https://example.com/p>: cited"
    out = agent._render_sources_text(ctx, _Resp(""), text)
    assert "- [web][searxng]" in out
    assert "https://example.com/p" in out


def test_an_invented_web_line_is_dropped_end_to_end(monkeypatch):
    monkeypatch.setattr(agent, "searxng_url", lambda: "http://searxng:8080")
    ctx = _FakeContext({agent.WEB_URLS_STATE_KEY: {"inv-1": ["https://example.com/real"]}})
    text = f"- a bullet\n[web][{WEB_TOKEN}][{MODEL_TOKEN}]<https://example.com/invented>: cited"
    out = agent._render_sources_text(ctx, _Resp(""), text)
    assert "invented" not in out


def test_a_turn_with_no_web_tool_call_carries_no_web_citation(monkeypatch):
    monkeypatch.setattr(agent, "searxng_url", lambda: "http://searxng:8080")
    ctx = _FakeContext()
    text = f"- a bullet\n[web][{WEB_TOKEN}][{MODEL_TOKEN}]<https://example.com/p>: cited"
    assert "[web]" not in agent._render_sources_text(ctx, _Resp(""), text)


def test_a_vault_citation_survives_alongside_a_dropped_web_line(monkeypatch):
    monkeypatch.setattr(agent, "searxng_url", lambda: "http://searxng:8080")
    ctx = _FakeContext()
    text = "\n".join(
        [
            "- a bullet",
            f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: vault",
            f"[web][{WEB_TOKEN}][{MODEL_TOKEN}]<https://example.com/p>: invented",
        ]
    )
    out = agent._render_sources_text(ctx, _Resp(""), text)
    assert "[[Dogs Overview]]" in out
    assert "[web]" not in out


def test_a_response_carrying_a_function_call_is_left_untouched():
    """The renderer must not delete the call the flow is about to dispatch."""
    response = _Resp("- interim text", function_call=True)
    assert agent.finalize_answer_after_model(_FakeContext(), response) is None


def test_a_streamed_chunk_is_left_untouched():
    response = _Resp("- partial")
    response.partial = True
    assert agent.finalize_answer_after_model(_FakeContext(), response) is None


def test_an_empty_response_is_left_untouched():
    assert agent.finalize_answer_after_model(_FakeContext(), _Resp("   ")) is None
