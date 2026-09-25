"""Initialize Langfuse tracing before any ADK runner starts.

No-op when LANGFUSE_PUBLIC_KEY is unset, so local runs without observability
still work.
"""

from __future__ import annotations

import os

_langfuse = None


def langfuse_client():
    """Return the shared Langfuse client, or None when observability is disabled."""
    return _langfuse


def setup_observability() -> None:
    global _langfuse
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return

    try:
        from langfuse import get_client
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    except ImportError:
        print(
            "Langfuse env vars set but packages missing; "
            'run: uv sync  (needs langfuse + openinference-instrumentation-google-adk)'
        )
        return

    langfuse = get_client()
    _langfuse = langfuse
    if not langfuse.auth_check():
        print(
            "Langfuse auth failed — check LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in .env"
        )
        return

    GoogleADKInstrumentor().instrument()
    print(f"Langfuse tracing enabled → {os.environ.get('LANGFUSE_BASE_URL', '')}")
