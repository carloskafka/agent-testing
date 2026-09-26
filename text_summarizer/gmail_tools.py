"""Optional read-only Gmail access via a stdio MCP server (this repo)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .obsidian_tools import sanitizing_mcp_toolset_class


def build_gmail_tools() -> list:
    """Return an McpToolset backed by the local Gmail MCP server, or [] when disabled.

    Enabled only when all three Google OAuth vars are set:
      GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN
    """
    required = [
        os.environ.get("GOOGLE_CLIENT_ID", "").strip(),
        os.environ.get("GOOGLE_CLIENT_SECRET", "").strip(),
        os.environ.get("GOOGLE_REFRESH_TOKEN", "").strip(),
    ]
    if not all(required):
        return []

    try:
        toolset_cls = sanitizing_mcp_toolset_class()
        from google.adk.tools.mcp_tool.mcp_session_manager import (
            StdioConnectionParams,
        )
        from mcp import StdioServerParameters
    except ImportError:
        print(
            "Gmail MCP env vars set but google-adk[mcp] not installed; "
            "run: uv sync"
        )
        return []

    server_path = Path(__file__).with_name("gmail_mcp_server.py")
    # The MCP stdio client only inherits a whitelist of env vars (HOME, PATH, ...)
    # by default, so the Google OAuth credentials must be forwarded explicitly or
    # the child server sees them as unset. `env` is merged over that whitelist.
    gmail_env = {
        "GOOGLE_CLIENT_ID": required[0],
        "GOOGLE_CLIENT_SECRET": required[1],
        "GOOGLE_REFRESH_TOKEN": required[2],
    }
    return [
        toolset_cls(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=[str(server_path)],
                    env=gmail_env,
                ),
            ),
            tool_filter=[
                "gmail_get_latest_messages",
                "gmail_search",
                "gmail_read",
                "gmail_get_thread",
            ],
        )
    ]