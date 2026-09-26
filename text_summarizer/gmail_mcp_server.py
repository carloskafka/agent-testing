"""Gmail MCP server (read-only) backed by the Gmail API.

The agent spawns this process over stdio (see ``gmail_tools.py``). It exposes a
small set of read-only tools so the agent can search, list, and read emails.

Configuration (all three must be set for the agent to load these tools):
  GOOGLE_CLIENT_ID       — OAuth 2.0 client id (Desktop app)
  GOOGLE_CLIENT_SECRET   — matching client secret
  GOOGLE_REFRESH_TOKEN   — minted once by ``gmail_oauth.py``
"""

from __future__ import annotations

import base64
import os
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from mcp.server.fastmcp import FastMCP

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_URI = "https://oauth2.googleapis.com/token"

mcp = FastMCP("gmail-mcp")

_service_cache = None


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is not set - Gmail tools unavailable")
    return value


def _service():
    """Lazily build a Gmail API client, refreshing the access token on demand."""
    global _service_cache
    if _service_cache is not None:
        return _service_cache
    creds = Credentials(
        token=None,
        refresh_token=_env("GOOGLE_REFRESH_TOKEN"),
        token_uri=TOKEN_URI,
        client_id=_env("GOOGLE_CLIENT_ID"),
        client_secret=_env("GOOGLE_CLIENT_SECRET"),
        scopes=SCOPES,
    )
    creds.refresh(Request())
    _service_cache = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return _service_cache


def _header(msg: dict[str, Any], name: str) -> str:
    for header in msg.get("payload", {}).get("headers", []):
        if str(header.get("name", "")).lower() == name.lower():
            return str(header.get("value", ""))
    return ""


def _decoded_body(payload: dict[str, Any]) -> str:
    """Walk a message payload (flat or multipart) and return its plain text."""
    body = payload.get("body") or {}
    data = body.get("data")
    if data:
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
        except Exception:
            return ""
    chunks = []
    for part in payload.get("parts") or []:
        mime = str(part.get("mimeType", ""))
        if mime.startswith("text/"):
            chunks.append(_decoded_body(part))
    return "\n".join(part for part in chunks if part)


def _summary(msg: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": msg.get("id"),
        "thread_id": msg.get("threadId"),
        "from": _header(msg, "From"),
        "to": _header(msg, "To"),
        "subject": _header(msg, "Subject"),
        "date": _header(msg, "Date"),
        "snippet": msg.get("snippet", ""),
    }


def _full(msg: dict[str, Any]) -> dict[str, Any]:
    result = _summary(msg)
    body = _decoded_body(msg.get("payload") or {})
    result["body"] = body or result["snippet"]
    return result


@mcp.tool()
def gmail_get_latest_messages(max_results: int = 5) -> list[dict[str, Any]]:
    """Return the N most recent messages from the user's Gmail inbox (metadata only)."""
    try:
        results = (
            _service()
            .users()
            .messages()
            .list(userId="me", maxResults=max(1, min(int(max_results), 50)))
            .execute()
        )
        return [
            _summary(
                _service()
                .users()
                .messages()
                .get(userId="me", id=msg["id"], format="metadata")
                .execute()
            )
            for msg in results.get("messages", [])
        ]
    except (HttpError, ValueError) as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def gmail_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the user's Gmail using a Gmail search query (e.g. 'from:jobs subject:offer is:unread')."""
    try:
        results = (
            _service()
            .users()
            .messages()
            .list(userId="me", q=query, maxResults=max(1, min(int(max_results), 50)))
            .execute()
        )
        return [
            _summary(
                _service()
                .users()
                .messages()
                .get(userId="me", id=msg["id"], format="metadata")
                .execute()
            )
            for msg in results.get("messages", [])
        ]
    except (HttpError, ValueError) as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def gmail_read(message_id: str) -> dict[str, Any]:
    """Read one full email (headers + body) by its message id."""
    try:
        msg = (
            _service()
            .users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return _full(msg)
    except HttpError as exc:
        return {"error": str(exc)}


@mcp.tool()
def gmail_get_thread(thread_id: str) -> list[dict[str, Any]]:
    """Return every message in an email thread by thread id, oldest first."""
    try:
        thread = (
            _service().users().threads().get(userId="me", id=thread_id).execute()
        )
        return [_full(m) for m in (thread.get("messages") or [])]
    except HttpError as exc:
        return [{"error": str(exc)}]


if __name__ == "__main__":
    mcp.run()