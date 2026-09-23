"""Optional Obsidian vault access via MCP (filesystem or HTTP transport)."""

from __future__ import annotations

import os


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
        from google.adk.tools.mcp_tool import McpToolset
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
        McpToolset(
            connection_params=connection_params,
            tool_filter=[
                "vault_list",
                "note_read",
                "note_create",
                "note_write",
                "note_append",
                "search_text",
                "search_metadata",
            ],
        )
    ]
