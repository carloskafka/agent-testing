"""Initialize Langfuse tracing before any ADK runner starts.

No-op when LANGFUSE_PUBLIC_KEY is unset, so local runs without observability
still work.
"""

from __future__ import annotations

import os

_langfuse = None

#: Name reported to Langfuse for the prompt behind every generation. Langfuse
#: stores it on the span as ``prompt_name``, which is the dimension the dashboards
#: group by, so without it the prompt breakdown stays empty.
DEFAULT_PROMPT_NAME = "text_summarizer"

#: Name reported for the trace wrapping one agent turn. Without it every trace lists
#: as a blank row in the UI, which is the first thing anyone looks at.
DEFAULT_TRACE_NAME = "text_summarizer"

# Langfuse's OTel ingestion maps these span attributes onto its own fields.
_PROMPT_NAME_ATTR = "langfuse.observation.prompt.name"
_PROMPT_VERSION_ATTR = "langfuse.observation.prompt.version"
_COMPLETION_START_ATTR = "langfuse.observation.completion_start_time"

# Trace-scoped attributes. Note that only ``langfuse.trace.*`` is namespaced --
# ``user.id``/``session.id`` are plain OTel semantic conventions that Langfuse maps
# onto the trace, so renaming them to ``langfuse.user.id`` would silently drop them.
_TRACE_NAME_ATTR = "langfuse.trace.name"
_USER_ID_ATTR = "user.id"
_SESSION_ID_ATTR = "session.id"


def prompt_name() -> str:
    """The prompt name to report, overridable via ``LANGFUSE_PROMPT_NAME``."""
    return os.environ.get("LANGFUSE_PROMPT_NAME", "").strip() or DEFAULT_PROMPT_NAME


def trace_name() -> str:
    """The trace name to report, overridable via ``LANGFUSE_TRACE_NAME``."""
    return os.environ.get("LANGFUSE_TRACE_NAME", "").strip() or DEFAULT_TRACE_NAME


def tag_trace_identity(callback_context, *_: object) -> None:
    """Name the current trace and attach ``userId``/``sessionId`` to it.

    Wired as the agent's ``before_agent_callback`` so it runs while the ``agent_run``
    span is open -- OTel silently drops attributes set on an ended span. Any span in
    the trace may carry these; Langfuse folds them onto the trace, so tagging the
    agent span is enough even though the runner's ``invocation`` span started first.

    ``userId``/``sessionId`` are what make Langfuse's "User consumption" widget and
    per-session filtering meaningful; without them every session is anonymous.

    The ``**_`` absorbs the remaining arguments ADK passes to a callback, and
    returning ``None`` leaves the turn untouched. Never raises: observability must
    not break the agent.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return

        attributes = {_TRACE_NAME_ATTR: trace_name()}

        session = getattr(callback_context, "session", None)
        session_id = getattr(session, "id", None)
        if session_id:
            attributes[_SESSION_ID_ATTR] = str(session_id)

        # CallbackContext.user_id is the authoritative value: it is the identity the
        # runner resolved for this turn, and it is set even for sessions that were
        # restored without a user_id on the session record.
        user_id = getattr(callback_context, "user_id", None) or getattr(
            session, "user_id", None
        )
        if user_id:
            attributes[_USER_ID_ATTR] = str(user_id)

        span.set_attributes(attributes)
    except Exception as exc:  # pragma: no cover
        print(f"[observability] trace identity tagging failed: {exc}")


def tag_current_span(**_: object) -> None:
    """Attach prompt identity to the currently active generation span.

    Wired directly as the agent's ``after_model_callback``; the ``**_`` absorbs the
    arguments ADK passes to a callback (``callback_context``, ``llm_response``) and
    returning ``None`` leaves the model's own response untouched.

    Must run while the span is still open -- OTel silently drops attributes set on
    an ended span. Never raises: observability must not break the agent.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        span.set_attribute(_PROMPT_NAME_ATTR, prompt_name())
    except Exception as exc:  # pragma: no cover
        print(f"[observability] prompt tagging failed: {exc}")


def langfuse_client():
    """Return the shared Langfuse client, or None when observability is disabled."""
    return _langfuse


def report_cache_outcome(*, enabled: bool, hit: bool, elapsed_ms: float) -> None:
    """Tag the current trace with the vault-cache outcome and score the lookup.

    Without this, a cache hit is only visible as the *absence* of a generation
    span, which cannot be filtered or aggregated. Emitting a tag plus two
    scores makes hit rate and lookup latency chartable, and makes cache-hit and
    live-LLM cohorts separable in the UI.

    Never raises: observability must not break the agent.
    """
    if _langfuse is None:
        return

    outcome = "disabled" if not enabled else ("hit" if hit else "miss")
    try:
        from langfuse import propagate_attributes

        with propagate_attributes(
            tags=[f"cache-{outcome}"],
            metadata={"cache.outcome": outcome, "cache.lookup_ms": f"{elapsed_ms:.3f}"},
        ):
            pass
    except Exception as exc:  # pragma: no cover
        print(f"[observability] cache tagging failed: {exc}")

    for name, value in (("cache.hit", 1.0 if hit else 0.0), ("cache.lookup_ms", round(elapsed_ms, 3))):
        try:
            _langfuse.score_current_trace(name=name, value=value, data_type="NUMERIC")
        except Exception as exc:  # pragma: no cover
            print(f"[observability] score '{name}' failed: {exc}")


def _auth_check_with_timeout(client, timeout: float) -> bool | None:
    """Run ``client.auth_check()`` under a timeout.

    Returns True/False on a definitive answer, or None when it could not be
    determined in time. ``auth_check`` is a blocking call to ``/api/public/projects``
    and is explicitly discouraged by the SDK for production code. A slow or wedged
    Langfuse must not silently disable tracing, so a timeout is reported as
    "unknown" and the caller proceeds to instrument anyway -- a genuinely invalid
    key then fails at export time, where the error is actionable, rather than at
    startup, where it looks like "observability is off".
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.auth_check)
        try:
            return bool(future.result(timeout=timeout))
        except FutureTimeout:
            future.cancel()
            return None
        except Exception as exc:
            print(f"[observability] auth check error ({type(exc).__name__}: {exc})")
            return False


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

    try:
        langfuse = get_client()
        try:
            timeout = float(os.environ.get("LANGFUSE_AUTH_CHECK_TIMEOUT", "5"))
        except ValueError:
            timeout = 5.0
        ok = _auth_check_with_timeout(langfuse, timeout)
        if ok is False:
            raise RuntimeError("auth_check() returned False")
        if ok is None:
            print(
                f"[observability] Langfuse auth check did not respond within "
                f"{timeout:g}s — instrumenting anyway. If traces do not appear, "
                "check that the Langfuse API is responsive."
            )
        GoogleADKInstrumentor().instrument()
    except Exception as exc:
        print(
            f"Langfuse unavailable ({type(exc).__name__}: {exc}) — continuing "
            "without observability. Check LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in .env"
        )
        return

    _langfuse = langfuse
    print(f"Langfuse tracing enabled → {os.environ.get('LANGFUSE_BASE_URL', '')}")
