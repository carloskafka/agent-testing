"""Optional Obsidian vault access via MCP (filesystem or HTTP transport)."""

from __future__ import annotations

import os
from typing import Any

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
        """``McpToolset`` that normalises every tool schema for the Gemini backends.

        The schema is rewritten in place on the raw MCP tool, which ADK caches per
        connection, so the transform runs once per server rather than once per turn.
        It is idempotent, so re-running on an already-sanitised schema is a no-op.
        """

        async def get_tools(self, readonly_context=None):
            tools = await super().get_tools(readonly_context)
            for tool in tools:
                raw = getattr(tool, "raw_mcp_tool", None)
                schema = getattr(raw, "inputSchema", None)
                if isinstance(schema, dict):
                    raw.inputSchema = sanitize_tool_schema(schema)
            return tools

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
            'run: uv sync'
        )
        return []

    if mcp_url:
        connection_params = StreamableHTTPConnectionParams(url=mcp_url)
    else:
        if not os.path.isabs(vault_path):
            print(f"OBSIDIAN_VAULT_PATH must be absolute, got: {vault_path}")
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
