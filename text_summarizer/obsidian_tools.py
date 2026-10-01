"""Optional Obsidian vault access via MCP (filesystem or HTTP transport)."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

#: JSON-Schema keywords dropped before a tool schema is shown to a Gemini backend.
#: ``$schema``/``$defs`` are draft-2020-12 metadata: Gemini's own parser has no use
#: for them, and once ``$ref`` targets are inlined (see :func:`_inline_refs`) keeping
#: them only risks a backend tripping over an empty ``$defs`` container.
_DROPPED_KEYS = frozenset({"$schema", "$defs"})

#: Depth cap for ``$ref`` expansion, so a self-referential ``$defs`` cannot hang the
#: process. No schema a real MCP server emits comes close.
_MAX_REF_DEPTH = 10

#: ``items`` injected for an array-typed parameter that declares none. See
#: :func:`sanitize_tool_schema` for why the absence is fatal rather than merely lax.
_DEFAULT_ITEMS: dict[str, str] = {"type": "string"}


def _inline_refs(node: Any, defs: dict, depth: int = 0) -> Any:
    """Recursively replace ``{"$ref": "#/$defs/X"}`` with a copy of ``defs["X"]``.

    Sibling keys (``description`` and friends) win over the target's own values, which
    matches JSON-Schema 2020-12 semantics for keywords that are not mutually exclusive.
    Unknown refs are left untouched rather than dropped, so a schema is never silently
    weakened by a typo on the server side.
    """
    if isinstance(node, list):
        return [_inline_refs(v, defs, depth) for v in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/") and depth < _MAX_REF_DEPTH:
        target = defs.get(ref[len("#/$defs/") :])
        if isinstance(target, dict):
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            expanded = _inline_refs(target, defs, depth + 1)
            merged = {**expanded, **{k: _inline_refs(v, defs, depth) for k, v in siblings.items()}}
            return _require_items(merged)

    return {
        k: _inline_refs(v, defs, depth) for k, v in node.items() if k not in _DROPPED_KEYS
    }


def _require_items(node: Any) -> Any:
    """Give every array-typed subschema an ``items`` keyword, recursing into children.

    ``type`` may be a list (a union). ``array`` anywhere in that list means the
    parameter also accepts a JSON array, and an array schema without ``items`` is
    rejected outright by Gemini's request validator.
    """
    if isinstance(node, list):
        return [_require_items(v) for v in node]
    if not isinstance(node, dict):
        return node

    declared = node.get("type")
    is_array = declared == "array" or (
        isinstance(declared, list) and "array" in declared
    )
    if is_array and "items" not in node:
        # The original schema left element types unconstrained; the widest schema
        # that still validates is an empty subschema, so preserve that intent.
        node = {**node, "items": dict(_DEFAULT_ITEMS)}

    return {k: _require_items(v) for k, v in node.items()}


#: Sessions evicted because a call against them failed. Exported for the tests and
#: for `doctor`; a non-zero value after a fresh process start means the server
#: dropped a session, which it should not be doing to a client in constant use.
MCP_SESSION_RECOVERIES = 0


async def evict_pooled_session(manager: Any, headers: Any) -> bool:
    """Drop a dead MCP session from ADK's pool. Return True if one was dropped.

    **Why this is needed at all.** ``MCPSessionManager.create_session`` decides a
    pooled session is reusable when the loop matches, the streams are open, and
    the background task is alive. It probes the streams with
    ``_is_session_disconnected``, which reads *client-side* ``_closed`` flags. A
    session the **server** terminated looks perfectly healthy by that test: the
    local transport is still open, the local task is still running, and only the
    server has forgotten the session. So the entry is handed back out forever and
    every subsequent call returns ``Session terminated``.

    ``retry_on_errors`` on ``create_session`` does not help: it retries the whole
    method, the reuse check passes again, and it returns the same dead session.
    The pool has to be emptied explicitly, which is what this does.

    Every step is defensive. ``MCPSessionManager`` exposes all of this privately,
    so a rename upstream would raise ``AttributeError``; that must degrade to
    "could not heal" (and then a loud failure) rather than to a crash inside the
    recovery path itself.

    Prefers ``_cleanup_session`` so the transport is torn down properly, and falls
    back to ``_forget_session``, which only drops the bookkeeping entries.
    """
    key_for = getattr(manager, "_session_key_for", None)
    if key_for is None:
        return False
    try:
        key = key_for(headers or None)
    except Exception:  # pragma: no cover - defensive
        logger.exception("could not compute the MCP session key")
        return False

    sessions = getattr(manager, "_sessions", None)
    entry = sessions.get(key) if isinstance(sessions, dict) else None
    cleanup = getattr(manager, "_cleanup_session", None)
    forget = getattr(manager, "_forget_session", None)

    try:
        if entry is not None and cleanup is not None:
            _session, exit_stack, stored_loop = entry
            return await _run_cleanup(cleanup(key, exit_stack, stored_loop))
        if forget is not None:
            forget(key)
            return True
    except Exception:  # pragma: no cover - defensive
        logger.exception("failed to evict the dead MCP session %s", key)
        return False
    return False


async def _run_cleanup(awaitable: Any) -> bool:
    """Await ADK's ``_cleanup_session`` and report success."""
    try:
        await awaitable
        return True
    except Exception:  # pragma: no cover - defensive
        logger.exception("MCP session cleanup failed")
        return False


def sanitize_tool_schema(schema: Any) -> Any:

    """Return a copy of a JSON Schema that every Gemini backend in this project accepts.

    Gemini is reached three ways and each validates differently:

    * the primary ``GeminiModel`` sends ``parameters_json_schema`` verbatim;
    * the ``FallbackModel`` chain sends the same schema through LiteLLM to
      OpenRouter, which may route the request to a Google AI Studio upstream. That
      upstream first lowers the JSON Schema to a native ``Schema`` message, turning
      a ``type`` union into ``any_of`` -- and an ``any_of`` branch of type ``ARRAY``
      without ``items`` is a hard ``INVALID_ARGUMENT``.

    The failure mode is nasty because it only fires on the *fallback* path, i.e.
    exactly when the primary model already failed: the turn dies on a schema the
    primary path had accepted moments earlier. Normalising here keeps one schema
    working on every path.

    Two transformations, both no-ops for schemas that are already valid:

    1. ``$ref`` targets are inlined and ``$schema``/``$defs`` removed, so no backend
       has to resolve JSON-Schema pointers (Gemini reports
       ``reference to undefined schema`` for a dangling ``$ref``).
    2. Every array-typed subschema gains an ``items`` keyword when it lacks one.
    """
    if not isinstance(schema, dict):
        return schema

    defs = schema.get("$defs")
    inlined = _inline_refs(schema, defs if isinstance(defs, dict) else {})
    return _require_items(inlined)


def sanitizing_mcp_toolset_class():
    """Build the ``McpToolset`` subclass, deferring the ADK import to call time.

    Shared by every MCP-backed toolset in the project (``obsidian_tools`` and
    ``gmail_tools``) so a third server cannot reintroduce the fallback-path 400 by
    omission.

    The import is deliberately inside the function: ``google-adk[mcp]`` is optional
    and the ``build_*_tools`` helpers must keep degrading to ``[]`` without it.
    Defining a real class at module scope would make the dependency mandatory at
    import time.
    """
    from google.adk.tools.mcp_tool import McpToolset

    class SanitizingMcpToolset(McpToolset):
        """``McpToolset`` that repairs a dead session and normalises tool schemas.

        Two jobs, both in ``get_tools`` because that is the one method every turn
        runs before the model sees any tool:

        * **Recovery.** A server that drops the session leaves ADK pooling a
          session that looks alive to every check ADK makes, so the vault tools
          disappear for the rest of the process's life. A failed listing evicts the
          session and is retried once against a fresh one. See
          :func:`evict_pooled_session` for the full mechanism.
        * **Schema normalisation.** Every schema is rewritten so each Gemini
          backend accepts it; see :func:`sanitize_tool_schema`.

        Recovery is retried exactly once and the second failure is re-raised
        rather than swallowed. That is a deliberate behaviour change: a vault
        that is configured but unreachable used to produce a turn in which the
        agent confidently reported *"there are no notes on that topic"*, which is
        indistinguishable from a correct answer and sent this whole investigation
        down the wrong path. A failed turn is visible; a plausible lie is not.
        """

        async def _fetch_tools(self, readonly_context=None):
            tools = await super().get_tools(readonly_context)
            for tool in tools:
                raw = getattr(tool, "raw_mcp_tool", None)
                schema = getattr(raw, "inputSchema", None)
                if isinstance(schema, dict):
                    raw.inputSchema = sanitize_tool_schema(schema)
            return tools

        async def get_tools(self, readonly_context=None):
            try:
                return await self._fetch_tools(readonly_context)
            except Exception as exc:
                logger.warning(
                    "MCP tool listing failed (%s); evicting the session and retrying once",
                    exc,
                )

            global MCP_SESSION_RECOVERIES
            headers = await self._build_headers(readonly_context)
            evicted = await evict_pooled_session(self._mcp_session_manager, headers)
            if not evicted:
                logger.error(
                    "MCP session could not be evicted (ADK's internals changed?); "
                    "the vault will stay unreachable until this process restarts"
                )
            else:
                MCP_SESSION_RECOVERIES += 1

            try:
                return await self._fetch_tools(readonly_context)
            except Exception:
                logger.exception(
                    "MCP vault unreachable after a session reset. The agent will "
                    "not answer this turn: an empty toolset would make it report "
                    "an empty vault, which reads as a correct answer."
                )
                raise

    return SanitizingMcpToolset


def build_obsidian_tools() -> list:
    """Return an McpToolset list based on env vars, or [] when disabled.

    Set one of:
      OBSIDIAN_VAULT_PATH  — absolute path to the vault (stdio via uvx obsidian-mcp)
      OBSIDIAN_MCP_URL     — streamable HTTP URL, e.g. http://127.0.0.1:37842/mcp
    """
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    mcp_url = os.environ.get("OBSIDIAN_MCP_URL", "").strip()

    if not vault_path and not mcp_url:
        return []

    try:
        toolset_cls = sanitizing_mcp_toolset_class()
        from google.adk.tools.mcp_tool.mcp_session_manager import (
            StdioConnectionParams,
            StreamableHTTPConnectionParams,
        )
        from mcp import StdioServerParameters
    except ImportError:
        print(
            "Obsidian MCP env vars set but google-adk[mcp] not installed; "
            "run: uv sync",
            file=sys.stderr,
        )
        return []

    if mcp_url:
        connection_params = StreamableHTTPConnectionParams(url=mcp_url)
    else:
        if not os.path.isabs(vault_path):
            print(
                f"OBSIDIAN_VAULT_PATH must be absolute, got: {vault_path}",
                file=sys.stderr,
            )
            return []
        connection_params = StdioConnectionParams(
            server_params=StdioServerParameters(
                command="uvx",
                args=["obsidian-mcp", vault_path],
            ),
        )

    return [
        toolset_cls(
            connection_params=connection_params,
            tool_filter=[
                # vault_info reports {"vault_name": ..., "vault_path": ...} and is
                # the only runtime surface that can name the vault. The agent is
                # not required to call it; when a turn does, sources.py picks the
                # name up from the function_response as a resolution fallback.
                "vault_info",
                "vault_list",
                "note_read",
                "note_create",
                "note_write",
                "note_insert",
                "search_text",
                "search_metadata",
            ],
        )
    ]
