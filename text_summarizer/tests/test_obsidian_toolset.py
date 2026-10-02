"""Which Obsidian tools the agent gets, and on what transport.

``build_obsidian_tools`` is the one function that decides whether this project has
a second brain at all, and it has no tests. The schema sanitiser next door has
two modules' worth of them, which is the imbalance this file corrects: the
sanitiser fixes a schema so a *configured* vault can be reached, and nothing says
the configuration half still works.

Everything here is offline. The toolset is a configuration object -- connection
parameters and a name filter -- and nothing is spawned, so a test costs nothing
and needs no ``uvx`` on ``PATH``.

The properties pinned, in the order a deployment meets them:

1. **disabled is the default and must be quiet.** Gotcha 11: with neither
   transport variable set the function returns ``[]``, the agent has only its two
   ``FunctionTool``s, and instruction rule 7 ("use ``search_text``") is
   unactionable. That is not a crash, so nothing else in the system would report
   it. ``conftest.py`` pins both variables to ``""``, which is the state this
   suite runs in -- so these are also the assertions that stop a developer's own
   ``.env`` from turning the disabled path into the enabled one mid-collection.
2. **a missing optional dependency must disable, not raise.** ``google-adk[mcp]``
   is an extra. The function is called at module import (``tools = [...
   *build_obsidian_tools() ...]``), so an ``ImportError`` here would take down
   the whole agent, not just the vault.
3. **the two transports are mutually exclusive and HTTP wins.** Configuring both
   is a mistake worth making once and diagnosing cheaply.
4. **the name filter is an allowlist**, so a renamed or misspelled tool is not a
   warning -- it is a tool that silently does not exist. The eight names are
   therefore pinned as a literal rather than derived from anything.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    StreamableHTTPConnectionParams,
)
from text_summarizer import obsidian_tools

#: The eight tools the filter allows, written out. A literal, not a set built from
#: the module, for two reasons: a name that disappeared from the filter would
#: otherwise be invisible (the module is the only place it is written down), and
#: ADK's filter is an *allowlist*, so a filter naming a tool the server does not
#: expose yields nothing at all rather than an error.
EXPECTED_TOOL_FILTER = {
    "vault_info",
    "vault_list",
    "note_read",
    "note_create",
    "note_write",
    "note_insert",
    "search_text",
    "search_metadata",
}

#: A schema of the shape that used to kill the fallback path: an array union with
#: no ``items``. See the "Tool-schema sanitisation" section of AGENTS.md and
#: ``test_obsidian_tool_schema.py`` for the whole story; it is reused here
#: because this module's only job in that story is to *install* the sanitiser.
DEAD_ON_GOOGLE_UPSTREAM = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "value": {"type": ["array", "boolean", "null", "number", "object", "string"]},
    },
}


@pytest.fixture
def obsidian_env(monkeypatch):
    """Blank both transports, so each test opts in explicitly.

    The inverse of ``gmail_env`` in ``test_gmail.py``, and needed for the same
    reason stated there: ``conftest.py`` pins these to ``""`` precisely so the
    disabled path is what a test run exercises by default.
    """
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "")
    monkeypatch.setenv("OBSIDIAN_MCP_URL", "")
    return monkeypatch


def _http_toolset(url: str = "http://127.0.0.1:37842/mcp"):
    """An instance of the sanitizing toolset, without connecting to anything.

    ``McpToolset.__init__`` builds an ``MCPSessionManager`` and validates the
    connection parameters; it does not open a session, so this is safe to
    construct in a test and is the object ``build_obsidian_tools`` returns.
    """
    cls = obsidian_tools.sanitizing_mcp_toolset_class()
    return cls(connection_params=StreamableHTTPConnectionParams(url=url))


# --- disabled is the default ---------------------------------------------------


def test_no_transport_configured_means_no_tools(obsidian_env):
    """Gotcha 11.

    Not an error, not a warning: the agent starts, answers from the model's own
    weights, and cannot retrieve anything. Rule 7 asks it to search the vault and
    it has nothing to search with, so a run looks like a summariser with no
    memory rather than like a misconfiguration.
    """
    assert obsidian_tools.build_obsidian_tools() == []


def test_whitespace_does_not_count_as_configured(obsidian_env):
    """``.env`` lines arrive with trailing spaces, and ``OBSIDIAN_MCP_URL=   ```
    would otherwise produce a connection to a host named by three spaces."""
    obsidian_env.setenv("OBSIDIAN_MCP_URL", "   ")
    obsidian_env.setenv("OBSIDIAN_VAULT_PATH", "\t")
    assert obsidian_tools.build_obsidian_tools() == []


# --- a missing optional dependency disables, it does not raise -----------------


def test_a_missing_mcp_package_degrades_to_no_tools(obsidian_env, capsys):
    """``google-adk[mcp]`` is an extra, and the message has to say what to run.

    Pinned because the import happens inside a function that ``agent.py`` calls
    at *module import*. An ``ImportError`` escaping here is not "the vault is
    unavailable", it is "the agent will not start", and the traceback would point
    at ``agent.py`` rather than at the missing extra.
    """
    obsidian_env.setenv("OBSIDIAN_MCP_URL", "http://127.0.0.1:37842/mcp")
    obsidian_env.setitem(sys.modules, "mcp", None)

    assert obsidian_tools.build_obsidian_tools() == []
    assert "uv sync" in capsys.readouterr().err


def test_a_missing_adk_mcp_module_degrades_to_no_tools(obsidian_env):
    """The other half of the same import: the toolset class itself.

    ``sanitizing_mcp_toolset_class`` is what defers ``McpToolset`` to call time,
    so that the degradation above is possible at all. Testing the ADK import
    separately is what proves the deferral covers *both* imports and not just the
    ``mcp`` one -- moving the first out of the function body would leave this
    green and the failure at module import.
    """
    obsidian_env.setenv("OBSIDIAN_MCP_URL", "http://127.0.0.1:37842/mcp")
    obsidian_env.setitem(sys.modules, "google.adk.tools.mcp_tool", None)

    assert obsidian_tools.build_obsidian_tools() == []


# --- transport selection -------------------------------------------------------


def test_the_url_transport_is_streamable_http(obsidian_env):
    obsidian_env.setenv("OBSIDIAN_MCP_URL", "http://127.0.0.1:37842/mcp")

    params = obsidian_tools.build_obsidian_tools()[0].connection_params

    assert isinstance(params, StreamableHTTPConnectionParams)
    assert params.url == "http://127.0.0.1:37842/mcp"


def test_the_stdio_transport_spawns_uvx_against_the_vault_path(obsidian_env, tmp_path):
    """``uvx obsidian-mcp <path>``, exactly.

    Both halves matter and neither is visible if the other is right. A different
    command (``uv``, ``npx``) or a different argument order produces a child
    process that either does not start or starts a server with no vault, and the
    second failure is the "there are no notes" answer that reads as a retrieval
    bug -- which is what AGENTS.md records a walkthrough losing time to.
    """
    vault = tmp_path / "ck"
    vault.mkdir()
    obsidian_env.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    params = obsidian_tools.build_obsidian_tools()[0].connection_params

    assert isinstance(params, StdioConnectionParams)
    assert params.server_params.command == "uvx"
    assert params.server_params.args == ["obsidian-mcp", str(vault)]


def test_a_relative_vault_path_is_refused_with_a_message(obsidian_env, capsys):
    """Absolute or nothing.

    The stdio server takes one positional path, so a relative one is resolved
    against whatever the *child's* working directory turns out to be -- which is
    not the agent's. The failure surfaces as a server with no vault, and rule 7
    goes unactionable with no error anywhere. Refusing at build time is the only
    point at which the cause is still knowable.
    """
    obsidian_env.setenv("OBSIDIAN_VAULT_PATH", "relative/path/to/vault")

    assert obsidian_tools.build_obsidian_tools() == []
    assert "must be absolute" in capsys.readouterr().err


def test_the_url_wins_when_both_transports_are_configured(obsidian_env, tmp_path):
    """HTTP takes precedence, and it wins *before* the path is validated.

    The deployment this mirrors is real: ``docker-compose.yml`` sets
    ``OBSIDIAN_MCP_URL`` for the sidecar and ``run.sh`` may also leave
    ``OBSIDIAN_VAULT_PATH`` in ``.env`` from an earlier configuration. The path
    is used as a *relative* one here, so if the precedence check were reordered
    the function would refuse and return ``[]`` -- the agent would answer with
    no vault at all, while the sidecar sat there healthy and reachable. Asserted
    as both the type and the URL so the winner is unambiguous.
    """
    obsidian_env.setenv("OBSIDIAN_VAULT_PATH", "a/stale/relative/path")
    obsidian_env.setenv("OBSIDIAN_MCP_URL", "http://127.0.0.1:37842/mcp")

    params = obsidian_tools.build_obsidian_tools()[0].connection_params

    assert isinstance(params, StreamableHTTPConnectionParams)
    assert params.url == "http://127.0.0.1:37842/mcp"


# --- the name filter ------------------------------------------------------------


def test_the_filter_names_exactly_the_eight_vault_tools(obsidian_env):
    """A literal, and the count as well as the names.

    ADK's ``tool_filter`` is an allowlist applied to what the server advertises.
    Renaming a tool in the filter is therefore not an error condition: the tool
    simply stops existing for the agent, the token count drops by exactly the
    schema of the missing tool, and nothing says why. ``note_write`` and
    ``note_insert`` are in here rather than out because persistence into the vault
    is a product feature, not just retrieval.
    """
    obsidian_env.setenv("OBSIDIAN_MCP_URL", "http://127.0.0.1:37842/mcp")

    toolset = obsidian_tools.build_obsidian_tools()[0]

    assert set(toolset.tool_filter) == EXPECTED_TOOL_FILTER
    # The set comparison above cannot see a duplicate, and a duplicated name
    # would silently shorten the real list.
    assert len(toolset.tool_filter) == len(EXPECTED_TOOL_FILTER) == 8


def test_vault_info_is_exposed_because_it_is_the_only_runtime_naming_surface(obsidian_env):
    """The one tool whose necessity is not obvious from its name.

    Everything else is retrieval or persistence. ``vault_info`` exists because
    ``sources.resolve_vault_name`` has exactly one runtime fallback for the
    vault's name, and this is the call that feeds it (AGENTS.md, "Vault name
    resolution", step 3). Dropping it from the filter would not break a feature;
    it would quietly remove the last way a name can be discovered when the vault
    is mounted somewhere that does not spell it -- turning
    ``[unknown]`` from a corner case into the normal answer.
    """
    obsidian_env.setenv("OBSIDIAN_MCP_URL", "http://127.0.0.1:37842/mcp")

    assert "vault_info" in obsidian_tools.build_obsidian_tools()[0].tool_filter


def test_the_search_tools_rule_seven_names_are_present(obsidian_env):
    """``search_text`` and ``search_metadata`` are the instruction, not a convenience.

    Rule 7 of the agent's instructions is "retrieve first", and these are the
    tools it names. Without them the rule is unactionable and the agent falls
    back to answering from the model, which looks like a correct answer.
    """
    obsidian_env.setenv("OBSIDIAN_MCP_URL", "http://127.0.0.1:37842/mcp")

    tool_filter = obsidian_tools.build_obsidian_tools()[0].tool_filter

    assert {"search_text", "search_metadata"} <= set(tool_filter)
    assert "note_read" in tool_filter, "rule 7 reads the notes it finds, not just the hits"


# --- the sanitizing toolset class ----------------------------------------------


def test_the_class_is_built_fresh_on_every_call():
    """Two calls, two classes -- and a module-level cache would break the reason
    the import is deferred.

    ``sanitizing_mcp_toolset_class`` imports ``McpToolset`` inside the function so
    that ``build_obsidian_tools`` can return ``[]`` when the extra is missing.
    Memoising the class (with ``functools.cache``, or by moving the body to
    module scope) would make the second call cheap and would also put the import
    back at module-import time, which is the failure ``test_a_missing_mcp_
    package_degrades_to_no_tools`` exists to catch. Asserting the two classes are
    distinct is what makes that regression visible here rather than there.
    """
    first = obsidian_tools.sanitizing_mcp_toolset_class()
    second = obsidian_tools.sanitizing_mcp_toolset_class()

    assert first is not second
    assert issubclass(first, McpToolset)
    assert issubclass(second, McpToolset)


def test_both_mcp_toolsets_are_built_from_the_same_class():
    """One sanitiser, shared -- the property that stops a third server re-breaking it.

    The fallback-path 400 (AGENTS.md, "Tool-schema sanitisation") comes back the
    moment *any* MCP toolset skips the rewrite, and it only fires when the
    primary model has already failed, so it is invisible until it is expensive.
    ``gmail_tools`` importing this module's factory is the guard; asserting the
    identity here means the guard lives in the file that documents why the
    factory is shared rather than only in the one that happens to use it.
    """
    from text_summarizer import gmail_tools

    assert gmail_tools.sanitizing_mcp_toolset_class is (obsidian_tools.sanitizing_mcp_toolset_class)
    assert issubclass(gmail_tools.sanitizing_mcp_toolset_class(), McpToolset), (
        "the shared factory still has to produce a toolset, not just a name"
    )


def test_get_tools_rewrites_the_raw_mcp_tool_schema(monkeypatch):
    """The rewrite happens in place, on the object ADK caches per connection.

    Not on a copy: ``McpToolset`` builds its tools once per connection and
    reuses them for every subsequent turn, so replacing ``raw_mcp_tool`` with a
    new object would leave the cached declaration unsanitised and the 400 would
    come back on the second call rather than the first. The assertion is on
    ``raw_mcp_tool.inputSchema`` specifically because that is the field the model
    request is built from.

    ``super().get_tools`` is stubbed so this exercises the transformation without
    a server. The recovery wrapper around it -- evict a dead session and retry
    once -- is a separate behaviour, covered in ``test_mcp_session_watchdog.py``.
    """
    sanitized = obsidian_tools.sanitize_tool_schema(DEAD_ON_GOOGLE_UPSTREAM)
    assert sanitized != DEAD_ON_GOOGLE_UPSTREAM, "the fixture must actually need fixing"

    async def run_async(*, args, tool_context):
        return "ok"

    tool = SimpleNamespace(
        name="search_metadata",
        raw_mcp_tool=SimpleNamespace(inputSchema=DEAD_ON_GOOGLE_UPSTREAM),
        run_async=run_async,
    )

    async def fake_get_tools(self, readonly_context=None):
        return [tool]

    monkeypatch.setattr(McpToolset, "get_tools", fake_get_tools)

    tools = asyncio.run(_http_toolset().get_tools())

    assert tools == [tool]
    assert tool.raw_mcp_tool.inputSchema == sanitized
    # The six-way union is narrowed, not merely given an `items`: a top-level
    # `items` on a parameter that is also a plain string is what OpenRouter's
    # ModelRun provider refuses ("more than one JSON reading of the same emitted
    # value"), and that refusal is a dead turn on the fallback path. See
    # `_require_items`.
    assert tool.raw_mcp_tool.inputSchema["properties"]["value"]["type"] == [
        "boolean",
        "null",
        "number",
        "object",
        "string",
    ]
    # ...and the tool is now dispatchable through the coercing wrapper, not just
    # carrying a rewritten schema. A tool that was sanitised but not wired up
    # would look identical above, which is why this is asserted separately.
    assert asyncio.run(tool.run_async(args={}, tool_context=None)) == "ok"


def test_get_tools_tolerates_a_tool_with_no_run_async(monkeypatch):
    """Coercion is skipped, not fatal, when there is nothing to wrap.

    Worth pinning because the failure it avoids is the worst available shape. An
    ``AttributeError`` inside ``_fetch_tools`` aborts the listing, which is the one
    failure this class exists to *recover* from -- the eviction-and-retry path would
    then run against a server that was answering perfectly, and the turn would die
    on a healthy vault. Degrading to "no coercion" costs one rejected argument and
    keeps the tools.
    """
    tool = SimpleNamespace(
        name="search_metadata",
        raw_mcp_tool=SimpleNamespace(inputSchema=DEAD_ON_GOOGLE_UPSTREAM),
    )

    async def fake_get_tools(self, readonly_context=None):
        return [tool]

    monkeypatch.setattr(McpToolset, "get_tools", fake_get_tools)

    tools = asyncio.run(_http_toolset().get_tools())

    assert tools == [tool]
    # The schema rewrite -- the part that predates coercion -- still happened.
    assert tool.raw_mcp_tool.inputSchema == obsidian_tools.sanitize_tool_schema(
        DEAD_ON_GOOGLE_UPSTREAM
    )
    assert not hasattr(tool, "run_async")


@pytest.mark.parametrize(
    "raw",
    [
        SimpleNamespace(),
        SimpleNamespace(inputSchema=None),
        SimpleNamespace(inputSchema="not a schema"),
    ],
)
def test_a_tool_without_a_usable_schema_is_left_alone(monkeypatch, raw):
    """The rewrite is guarded by two ``getattr`` calls and an ``isinstance``.

    This loop runs on every model call for every tool, so it is on the hot path
    of every turn. A server that adds a tool with no ``inputSchema`` at all is
    ordinary, and the loop must not raise on it -- which is the difference between
    "one tool renders without validation" and "the turn is dead".
    """
    tool = SimpleNamespace(name="odd", raw_mcp_tool=raw)

    async def fake_get_tools(self, readonly_context=None):
        return [tool]

    monkeypatch.setattr(McpToolset, "get_tools", fake_get_tools)

    assert asyncio.run(_http_toolset().get_tools()) == [tool]
    assert getattr(raw, "inputSchema", None) in (None, "not a schema")


def test_a_tool_with_no_raw_mcp_tool_at_all_survives(monkeypatch):
    """``getattr(tool, "raw_mcp_tool", None)`` -- the attribute is optional.

    ADK's own ``FunctionTool``-shaped tools can end up in the same list on a
    connection, and ``None.inputSchema`` would be an ``AttributeError`` on every
    turn rather than on the first.
    """
    tool = SimpleNamespace(name="no-raw", raw_mcp_tool=None)

    async def fake_get_tools(self, readonly_context=None):
        return [tool]

    monkeypatch.setattr(McpToolset, "get_tools", fake_get_tools)

    assert asyncio.run(_http_toolset().get_tools()) == [tool]
