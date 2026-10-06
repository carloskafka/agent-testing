"""The current date and time, as a tool the model can call.

Two gaps this closes, both reported live:

1. **The model does not know what day it is.** A system prompt cannot carry a
   date, because it is fixed when the prompt is written and the server keeps
   running. "Summarize today's news" therefore means whatever the model's
   training cutoff suggested, and a Gemini or OpenRouter fallback has no
   clock of its own to fall back on either.
2. **The vault cache returned the previous day's answer** for exactly that
   prompt. See :func:`second_brain.cache_key_text_for` -- that is the deeper
   defect, fixed separately, because it lives in the cache key rather than in the
   model's knowledge.

So this tool is the *complement* of that fix, not the fix: with the cache
day-scoped, a bare repeat re-asks the model every day, and the model still cannot
say which day it is. Having the clock lets it resolve "today" itself instead of
guessing, which is what the user actually asked for.

Deliberately cheap and unconditioned: no arguments, no configuration, no
network, no ``tool_context``. It cannot fail in a way that breaks a turn, so it
does not need the wrapping the vault writers have.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from google.adk.tools.function_tool import FunctionTool


def current_datetime(timezone_name: str = "local", tool_context: Any | None = None) -> str:
    """The current date and time as JSON. Use this whenever the answer depends on when it is.

    Call it before answering anything involving "today", "yesterday", "this week",
    "latest", "current", or a date range -- the model has no clock of its own and
    its training cutoff is not the present.

    Args:
        timezone_name: "local" for the server's timezone, "utc" for UTC. A named
            IANA zone such as "Europe/Lisbon" is accepted if the host has tzdata.
    """
    # Prefer session state timezone when caller asks for "local" and we have one.
    tz = timezone_name.strip().lower()
    chosen = timezone_name.strip()
    if tz in ("", "local") and tool_context is not None:
        try:
            state = getattr(tool_context, "state", None)
            stored = None
            if isinstance(state, dict):
                stored = state.get("user_timezone")
            else:
                try:
                    stored = state.get("user_timezone")
                except Exception:
                    stored = None
            if isinstance(stored, str) and stored.strip():
                chosen = stored.strip()
                tz = chosen.lower()
        except Exception:
            pass

    if tz in ("", "local"):
        moment = datetime.now().astimezone()
        zone = str(moment.tzinfo) if moment.tzinfo else "local"
    elif tz == "utc":
        moment = datetime.now(UTC)
        zone = "UTC"
    else:
        try:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

            zone = chosen
            moment = datetime.now(ZoneInfo(zone))
        except (ImportError, ZoneInfoNotFoundError, ValueError, KeyError):
            # A bad zone name is not worth failing a turn over: fall back to local
            # and say so, so the model can see what it actually got rather than
            # silently reasoning about the wrong day.
            moment = datetime.now().astimezone()
            zone = f"local (unrecognised timezone {timezone_name!r})"

    return json.dumps(
        {
            "iso": moment.isoformat(timespec="seconds"),
            "date": moment.date().isoformat(),
            "time": moment.strftime("%H:%M:%S"),
            "weekday": moment.strftime("%A"),
            "timezone": zone,
        },
        ensure_ascii=False,
    )


def build_clock_tools() -> list:
    """The clock tool, always.

    Unlike the other toolsets this one is *not* conditional. A missing clock is a
    silent wrong answer rather than a missing capability, and it costs no
    configuration, no key and no network -- so gating it would introduce the very
    kind of "worked yesterday, wrong today" behaviour it exists to prevent. Returned
    as a list anyway, so the caller spreads it the same way as the others.
    """
    return [FunctionTool(current_datetime)]
