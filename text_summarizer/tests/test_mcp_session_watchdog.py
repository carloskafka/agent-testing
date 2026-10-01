"""Tests for the MCP session watchdog.

The bug is not theoretical and it was found live. ``obsidian-mcp`` runs as a
sidecar in the agent's network namespace; its streamable-HTTP session was
terminated server-side while the client's own transport stayed open. ADK's reuse
check (``MCPSessionManager.create_session``) probes only *client-side* state --
``_read_stream._closed`` / ``_write_stream._closed`` plus the background task -- so
the dead entry passed every check and was handed back out on every later call.

``retry_on_errors`` cannot fix that, and the reason matters for the test
arithmetic below. It retries the whole wrapped call, but ``create_session``'
reuse check passes again, so it returns the *same* dead session. Worse, ADK
decorates ``McpToolset.get_tools`` with ``@retry_on_errors`` too, so a failed
listing costs two round trips *inside* the base class before our override sees
anything.

The symptom, from the agent's side, was a lie. With the toolset yielding zero
tools, the model reported *"No direct note lookup tools were provided in the
execution environment"* and *"the second brain currently contains no notes"* --
both true, and both indistinguishable from a correct answer. A restart fixed it,
which is what made it so hard to believe.

No MCP server, network or event loop is involved: the recovery logic is a pure
function over a stand-in for ``MCPSessionManager``, because everything it touches
is a private attribute whose exact shape is ADK's business, not ours.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from text_summarizer.obsidian_tools import (
    evict_pooled_session,
    sanitizing_mcp_toolset_class,
)

#: How many round trips a *single* ``McpToolset.get_tools`` call costs when the
#: server is dead. The base method carries ``@retry_on_errors``, so a failure is
#: attempted twice before it escapes. Asserted directly by
#: :func:`test_the_base_class_already_retries_once` so that a change in ADK's
#: decoration shows up here rather than as a silent shift in every other count.
CALLS_PER_FAILED_LISTING = 2


class FakeSessionManager:
    """The three private surfaces :func:`evict_pooled_session` reaches for.

    Shaped after ``google.adk.tools.mcp_tool.mcp_session_manager.MCPSessionManager``
    at the point of the bug: a pool of ``(session, exit_stack, loop)`` triples
    keyed by a session key derived from the headers.
    """

    def __init__(self, *, with_cleanup=True, with_forget=True, with_key_for=True):
        self._sessions = {"session_no_headers": ("sess", "stack", "loop")}
        self._session_contexts = {"session_no_headers": object()}
        self.closed = []
        self.forgotten = []
        if with_key_for:
            self._session_key_for = lambda headers: "session_no_headers"
        if with_forget:
            self._forget_session = self.forgotten.append
        if with_cleanup:

            async def _cleanup(key, exit_stack, stored_loop):
                self.closed.append((key, exit_stack, stored_loop))
                self._sessions.pop(key, None)

            self._cleanup_session = _cleanup


# --- the eviction primitive ---------------------------------------------------


def test_a_live_session_is_teared_down_through_cleanup():
    manager = FakeSessionManager()
    assert asyncio.run(evict_pooled_session(manager, None)) is True
    # _cleanup_session, not _forget_session: the transport must actually close.
    assert manager.closed == [("session_no_headers", "stack", "loop")]
    assert manager.forgotten == []


def test_forget_only_is_used_when_cleanup_is_absent():
    manager = FakeSessionManager(with_cleanup=False)
    assert asyncio.run(evict_pooled_session(manager, None)) is True
    assert manager.forgotten == ["session_no_headers"]


def test_a_manager_with_different_internals_reports_cannot_heal():
    """A rename upstream must degrade to 'could not heal', never to a crash."""

    class Renamed:
        pass

    assert asyncio.run(evict_pooled_session(Renamed(), None)) is False


@pytest.mark.parametrize("attr", ["_session_key_for"])
def test_a_missing_private_surface_reports_cannot_heal(attr):
    """The pool cannot be emptied, so the caller must be told it could not heal."""
    manager = FakeSessionManager()
    delattr(manager, attr)
    assert asyncio.run(evict_pooled_session(manager, None)) is False


def test_a_raising_key_lookup_reports_cannot_heal():
    class Hostile:
        def _session_key_for(self, headers):
            raise RuntimeError("nope")

    assert asyncio.run(evict_pooled_session(Hostile(), None)) is False


def test_a_raising_cleanup_reports_cannot_heal():
    class Hostile(FakeSessionManager):
        def __init__(self):
            super().__init__(with_cleanup=False)
            self._forget_session = self._boom

        def _boom(self, key):
            raise RuntimeError("nope")

    assert asyncio.run(evict_pooled_session(Hostile(), None)) is False


def test_an_empty_pool_is_not_an_error():
    manager = FakeSessionManager()
    manager._sessions.clear()
    assert asyncio.run(evict_pooled_session(manager, None)) is True


# --- the toolset-level contract ----------------------------------------------


def _toolset_stub(manager, *, results):
    """The real subclass, with only the MCP round trip scripted out.

    Construction goes through the real ``__init__`` so the tool-list cache and
    every attribute the real ``get_tools`` reads are present, and only
    ``_execute_with_session`` -- the call that fails in production -- is replaced.
    That keeps the shipped ``get_tools`` override on the path under test.
    """
    from google.adk.tools.mcp_tool.mcp_session_manager import (
        StreamableHTTPConnectionParams,
    )
    from mcp.types import Tool as McpTool

    toolset_cls = sanitizing_mcp_toolset_class()
    instance = toolset_cls(
        connection_params=StreamableHTTPConnectionParams(url="http://127.0.0.1:1/mcp"),
        tool_filter=["search_text"],
    )
    instance._mcp_session_manager = manager
    calls = []

    async def fake_execute(coroutine_func, error_message, readonly_context=None, headers=None):
        calls.append("fetch")
        outcome = results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        tools = [
            McpTool(name=name, description=name, inputSchema={"type": "object"}) for name in outcome
        ]
        return SimpleNamespace(tools=tools)

    instance._execute_with_session = fake_execute
    return instance, calls


def test_a_healthy_listing_never_touches_the_pool():
    manager = FakeSessionManager()
    instance, calls = _toolset_stub(manager, results=[["search_text"]])

    assert [t.name for t in asyncio.run(instance.get_tools(None))] == ["search_text"]
    assert calls == ["fetch"]
    assert manager.closed == [] and manager.forgotten == []


def test_a_permanently_dead_server_costs_four_round_trips_and_then_raises():
    """Pins the arithmetic every other count in this file depends on.

    ``McpToolset.get_tools`` carries ``@retry_on_errors``, so each attempt at a
    dead server costs two round trips before the failure escapes. Our override
    then evicts and tries again, which costs the same two. If ADK drops or adds
    that decoration this test fails loudly, instead of every other count here
    quietly shifting.
    """
    manager = FakeSessionManager()
    instance, calls = _toolset_stub(
        manager, results=[RuntimeError(f"boom {i}") for i in range(2 * CALLS_PER_FAILED_LISTING)]
    )

    with pytest.raises(RuntimeError):
        asyncio.run(instance.get_tools(None))

    assert len(calls) == 2 * CALLS_PER_FAILED_LISTING
    assert manager.closed == [("session_no_headers", "stack", "loop")]


def test_a_dead_session_is_evicted_and_the_listing_retried():
    manager = FakeSessionManager()
    results = [RuntimeError("boom")] * CALLS_PER_FAILED_LISTING + [["search_text"]]
    instance, _ = _toolset_stub(manager, results=results)

    tools = asyncio.run(instance.get_tools(None))

    assert [t.name for t in tools] == ["search_text"]
    assert manager.closed == [("session_no_headers", "stack", "loop")]


def test_a_second_failure_propagates_rather_than_yielding_no_tools():
    """The whole point: an unreachable vault must fail the turn, not lie in it."""
    manager = FakeSessionManager()
    results = [RuntimeError(f"boom {i}") for i in range(2 * CALLS_PER_FAILED_LISTING)]
    instance, _ = _toolset_stub(manager, results=results)

    with pytest.raises(RuntimeError):
        asyncio.run(instance.get_tools(None))

    assert manager.closed == [("session_no_headers", "stack", "loop")]


def test_an_unhealable_pool_still_fails_the_turn():
    """Cannot heal is not a licence to answer from an empty toolset."""
    manager = FakeSessionManager(with_cleanup=False, with_forget=False, with_key_for=False)
    results = [RuntimeError(f"boom {i}") for i in range(2 * CALLS_PER_FAILED_LISTING)]
    instance, _ = _toolset_stub(manager, results=results)

    with pytest.raises(RuntimeError):
        asyncio.run(instance.get_tools(None))


def test_the_recovery_counter_moves_only_on_a_real_eviction():
    from text_summarizer import obsidian_tools

    before = obsidian_tools.MCP_SESSION_RECOVERIES

    healthy = FakeSessionManager()
    instance, _ = _toolset_stub(healthy, results=[["search_text"]])
    asyncio.run(instance.get_tools(None))
    assert before == obsidian_tools.MCP_SESSION_RECOVERIES

    healing = FakeSessionManager()
    results = [RuntimeError("boom")] * CALLS_PER_FAILED_LISTING + [[]]
    instance, _ = _toolset_stub(healing, results=results)
    asyncio.run(instance.get_tools(None))
    assert before + 1 == obsidian_tools.MCP_SESSION_RECOVERIES
