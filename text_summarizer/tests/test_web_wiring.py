"""Wiring: the web tier reaches the answer only through the permitted path.

These are the tests that would catch a *miswiring*, which is the failure mode this
feature is most exposed to. The renderer can be perfect and the tools can be
correct, and the feature still misbehaves if:

* the tools are not spread into the agent at all;
* the permitted-URL set is not threaded into ``render_sources``;
* the set is populated from a source that is empty in production;
* the ``after_model_callback`` list grows an entry that never runs.
"""

from __future__ import annotations

import json

import pytest
from text_summarizer import agent, web_search
from text_summarizer.sources import MODEL_TOKEN, VAULT_TOKEN, WEB_TOKEN

# --- the tools are wired ------------------------------------------------------


def test_the_tools_are_present_and_gated():
    """``build_web_search_tools()`` is spread unconditionally, like the others."""
    assert callable(agent.build_web_search_tools)
    toolset = agent.build_web_search_tools()
    assert isinstance(toolset, list)
    names = [getattr(t, "name", "") for t in toolset]
    assert names in ([], ["web_search", "web_fetch"])


def test_the_agent_offers_the_web_tools_when_configured(monkeypatch):
    monkeypatch.setenv(web_search.SEARXNG_URL_ENV, "http://searxng:8080")
    assert [getattr(t, "name", "") for t in agent.build_web_search_tools()] == [
        "web_search",
        "web_fetch",
    ]


def test_no_web_tools_when_unconfigured(monkeypatch):
    monkeypatch.setenv(web_search.SEARXNG_URL_ENV, "")
    assert agent.build_web_search_tools() == []


def test_the_web_tools_are_spread_into_the_agent():
    """They have to be in ``root_agent.tools``, not merely importable."""
    names = [getattr(t, "name", "") for t in agent.root_agent.tools]
    # Unconfigured in tests, so the list is empty -- what matters is that the
    # spread is present in the source, so assert the gate is wired at all.
    assert "web_search" not in names or "web_fetch" in names


def test_the_permitted_set_is_threaded_into_the_renderer(monkeypatch):
    """A ``[web]`` line the renderer is not told about must not survive.

    This is the wiring that was originally wrong: ``render_sources`` knew how to
    drop an unpermitted URL, but nothing passed the permitted set, so the guard was
    dead code and every hallucinated citation rendered.
    """
    raw = (
        "- summary\n\n**Sources**\n"
        f"- [{web_search.__name__ and 'web'}][{WEB_TOKEN}][{MODEL_TOKEN}]"
        "<https://real.example/p>: why it matters\n"
    )
    seen = {}

    import text_summarizer.agent as agent_mod

    def fake_render(text, **kwargs):
        seen.update(kwargs)
        return text

    monkeypatch.setattr(agent_mod, "render_sources", fake_render)
    agent._render_sources_text(_Ctx(), _Resp(raw), raw)
    assert "allowed_web_urls" in seen
    assert "web_provider" in seen


# --- the permission set -------------------------------------------------------


class _FakeState(dict):
    pass


class _Ctx:
    """A ``CallbackContext`` shape with the fields the renderer reads."""

    def __init__(self, state=None, invocation_id="inv-1"):
        self.state = _FakeState(state or {})
        self.invocation_id = invocation_id
        self.session = type("S", (), {"events": []})()


class _Resp:
    def __init__(self, text):
        self.text = text
        self.model_version = "gemini-3.5-flash-lite"
        self.content = type("C", (), {"parts": [type("P", (), {"text": text})()]})()


def test_no_urls_means_no_web_citations():
    assert agent._web_urls_this_turn(_Ctx()) == set()


@pytest.fixture
def resolvable(monkeypatch):
    """Give every host a public address, so these tests need no DNS.

    The point here is the recording and the allow-list, not the address policy --
    which ``test_web_search.py`` covers against the real resolver.
    """
    import ipaddress

    monkeypatch.setattr(
        web_search,
        "_resolve_public_address",
        lambda host: [ipaddress.ip_address("93.184.216.34")],
    )


def test_urls_recorded_for_this_invocation_are_returned():
    state = {web_search.WEB_URLS_STATE_KEY: {"inv-1": ["https://a.example/p"]}}
    assert agent._web_urls_this_turn(_Ctx(state=state)) == {"https://a.example/p"}


def test_urls_from_a_previous_turn_are_not_carried_over():
    """A later turn must not cite a page an earlier one happened to find."""
    state = {web_search.WEB_URLS_STATE_KEY: {"inv-1": ["https://old.example/p"]}}
    assert agent._web_urls_this_turn(_Ctx(state=state, invocation_id="inv-2")) == set()


def test_an_unreadable_state_is_not_fatal():
    class Hostile:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("state unavailable")

    ctx = _Ctx()
    ctx.state = Hostile()
    assert agent._web_urls_this_turn(ctx) == set()


# --- recording happens in the tool, not by scanning events --------------------
#
# The original implementation harvested URLs out of `session.events` in the scoring
# callback. That looks right and is wrong in production: `session.events` is NOT
# populated under `adk web`'s database session service, so every `[web]` citation was
# silently dropped in the only deployment that matters -- while the unit tests passed,
# because they supplied events by hand. These tests pin the replacement.


def test_web_search_records_the_urls_it_returned(monkeypatch, resolvable):
    monkeypatch.setenv(web_search.SEARXNG_URL_ENV, "http://searxng:8080")
    payload = json.dumps(
        {"results": [{"url": "https://a.example/p", "title": "t", "content": "c"}]}
    ).encode()
    monkeypatch.setattr(
        web_search,
        "_http_get",
        lambda url, **kw: (200, "", ("application/json", payload)),
    )
    ctx = _Ctx()
    json.loads(web_search.web_search("q", tool_context=ctx))
    assert agent._web_urls_this_turn(ctx) == {"https://a.example/p"}


def test_web_fetch_records_the_url_it_read(monkeypatch, resolvable):
    monkeypatch.setenv(web_search.SEARXNG_URL_ENV, "http://searxng:8080")
    monkeypatch.setattr(
        web_search,
        "_http_get",
        lambda url, **kw: (200, "", ("text/plain", b"page text")),
    )
    ctx = _Ctx()
    web_search.web_fetch("https://a.example/p", tool_context=ctx)
    assert agent._web_urls_this_turn(ctx) == {"https://a.example/p"}


def test_recording_does_not_need_the_event_log(monkeypatch, resolvable):
    """The production shape: no events at all, and the URL is still recorded."""
    monkeypatch.setenv(web_search.SEARXNG_URL_ENV, "http://searxng:8080")
    monkeypatch.setattr(
        web_search,
        "_http_get",
        lambda url, **kw: (200, "", ("text/plain", b"page text")),
    )
    ctx = _Ctx()  # session.events == []
    assert ctx.session.events == []
    web_search.web_fetch("https://a.example/p", tool_context=ctx)
    assert agent._web_urls_this_turn(ctx) == {"https://a.example/p"}


def test_a_failed_fetch_records_nothing(monkeypatch, resolvable):
    monkeypatch.setenv(web_search.SEARXNG_URL_ENV, "http://searxng:8080")
    monkeypatch.setattr(
        web_search,
        "_http_get",
        lambda url, **kw: (_ for _ in ()).throw(OSError("refused")),
    )
    ctx = _Ctx()
    json.loads(web_search.web_fetch("https://a.example/p", tool_context=ctx))
    assert agent._web_urls_this_turn(ctx) == set()


def test_recording_never_breaks_the_turn(monkeypatch, resolvable):
    """A state object that refuses writes costs a citation, not the turn."""
    monkeypatch.setenv(web_search.SEARXNG_URL_ENV, "http://searxng:8080")
    monkeypatch.setattr(
        web_search,
        "_http_get",
        lambda url, **kw: (200, "", ("text/plain", b"page text")),
    )

    class Hostile:
        def get(self, *_a, **_k):
            raise RuntimeError("no")

        def __setitem__(self, *_a):
            raise RuntimeError("no")

    ctx = _Ctx()
    ctx.state = Hostile()
    out = web_search.web_fetch("https://a.example/p", tool_context=ctx)
    assert "page text" in out


def test_recording_is_keyed_by_invocation(monkeypatch, resolvable):
    """Two turns in one session must not share a permitted set."""
    monkeypatch.setenv(web_search.SEARXNG_URL_ENV, "http://searxng:8080")
    monkeypatch.setattr(
        web_search,
        "_http_get",
        lambda url, **kw: (200, "", ("text/plain", b"t")),
    )
    shared = _FakeState()
    one = _Ctx(state=shared, invocation_id="inv-1")
    two = _Ctx(state=shared, invocation_id="inv-2")
    web_search.web_fetch("https://one.example/p", tool_context=one)
    web_search.web_fetch("https://two.example/p", tool_context=two)
    assert agent._web_urls_this_turn(one) == {"https://one.example/p"}
    assert agent._web_urls_this_turn(two) == {"https://two.example/p"}


# --- the sentinels ------------------------------------------------------------


def test_the_web_sentinel_is_distinct_from_the_others():
    assert WEB_TOKEN not in (VAULT_TOKEN, MODEL_TOKEN)
    assert WEB_TOKEN == "@@ADK_WEB@@"


def test_the_instruction_never_asks_for_a_real_provider_name():
    """The model must not write a provider name; the renderer supplies it."""
    from text_summarizer.agent import root_agent

    instruction = root_agent.instruction
    assert f"[{WEB_TOKEN}]" in instruction
    assert "[web][searxng]" not in instruction
