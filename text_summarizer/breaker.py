"""Remember which provider accounts are spent, so a dead tier costs one probe.

**What this is for.** Session ``b6f09fca``, 2026-10-02. Gemini served eight calls of
a turn, the ninth returned 429, and the chain walked to OpenRouter — where every
entry answered ``free-models-per-day, remaining 0`` — and the turn **wrote its
notes and returned no answer at all**. There was nothing after the last tier.

The waste was never large. A 429 comes back in **0.23s** (measured), so skipping one
saves a fifth of a second. That is worth saying plainly, because the reason to build
this is not the 1.15s saved across six dead entries -- it is that a dead tier is
**indistinguishable from a slow one** in the trace. A turn that falls through four
providers and produces nothing looks exactly like a turn where every provider is
genuinely broken, and the two need different fixes.

**What it does not do, and why that matters more.**

* It does not rank models by speed. Ranking by latency optimises a signal that is
  not the bottleneck: a quota-exhausted account never runs the model at all. Worse,
  OpenRouter's published p50 disagrees with what this deployment actually measured
  by ~40x for one model (rank 91, measured median 18.9s), so a score built on it
  would be chasing noise.
* It does not reorder the chain between calls within a turn. That is the defect in
  ``model_chain``'s own docstring: Gemini validates the *whole* conversation, so a
  provider change mid-turn can leave a ``functionCall`` without a
  ``thought_signature`` and kill the next Gemini call with a 400. This module only
  ever removes an entry that has *already* proven it cannot serve, which is a
  strictly weaker claim and cannot splice a live conversation.
* It does not retry. ADK excludes 408 deliberately (a timeout does not say whether
  the request was processed), and a 429 is already in its retriable set.

**State is process-local and deliberately so.** It lives in a module-level dict
because the failure it prevents is a *turn* dying, and a turn runs in one process.
Persisting it would raise the questions of clock skew between containers, of what
happens when the reset passes while the process is down, and of whether a stale
"spent" mark is worse than no mark at all -- for a saving measured in tenths of a
second. The honest trade: a restarted container re-probes once, which costs the
0.23s this module exists to avoid.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

#: Fallback when a 429 carries no usable reset time: how long to consider the
#: account dead. Long enough not to re-probe on every turn, short enough that a
#: forgotten account recovers on its own. OpenRouter's daily budget resets at
#: 00:00 UTC, and the measured reset header pointed exactly there.
DEFAULT_COOLDOWN_S = 6 * 3600

#: A reset further out than this is treated as "unknown" rather than trusted.
#: It is a guard against a garbage header pinning an account off for a week, and it
#: costs nothing: the fallback cooldown applies instead.
MAX_COOLDOWN_S = 24 * 3600

#: Keyed by a provider account, not a model. **Deliberate, and load-bearing.**
#: OpenRouter's free budget is per *account* and shared across every ``:free`` model
#: on it -- measured, an account at 0/50 answers 429 for all of them. Keying by
#: model would leave five dead entries each probing, which is exactly the failure
#: mode. The value is the fingerprint of the key, never the key itself.
_SPENT: dict[str, float] = {}

_LOCK = threading.Lock()


@dataclass
class _BreakerState:
    """What one trip of the breaker looks like, for a readable ``repr``."""

    fingerprint: str
    until: float
    source: str
    trips: int = 0


def _fingerprint(key: str) -> str:
    """A stable, non-reversible label for a key.

    The breaker has to key on *which account*, so it needs something derived from
    the key -- but the key must never reach a log, a trace or a repr. An 8-char
    digest is enough to tell two accounts apart and useless to anyone who reads it.
    """
    import hashlib

    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def is_spent(key: str, *, now: float | None = None) -> bool:
    """Whether this account was recently proved exhausted.

    Reads do **not** delete the entry: the cooldown is a clock, not a one-shot
    latch, and a turn makes several calls per model, so clearing on read would
    re-probe within the same turn.
    """
    if not key:
        return False
    moment = time.monotonic() if now is None else now
    with _LOCK:
        until = _SPENT.get(_fingerprint(key))
        if until is None:
            return False
        if moment >= until:
            # Expired. Drop it so the dict cannot grow without bound across days.
            _SPENT.pop(_fingerprint(key), None)
            return False
        return True


def reset_at_from_error(exc: BaseException, *, now: float | None = None) -> float | None:
    """Seconds until the provider says this account resets, from a raised error.

    Reads ``X-RateLimit-Reset`` out of the provider body that LiteLLM embeds in its
    exception message. That is a string-scraping step and it is worth being honest
    about: **LiteLLM discards the response headers** -- measured, ``e.response.headers``
    came back empty on a real 429 -- while the JSON body, including its
    ``metadata.headers`` block, survives in ``str(exc)``. So the header is available
    in the one place it is not typed or structured.

    Returns ``None`` when there is nothing trustworthy to read, and the caller falls
    back to :data:`DEFAULT_COOLDOWN_S`. Never raises: this runs on the failure path,
    where an exception here would replace a 429 with a different, harder failure.
    """
    match = _reset_re.search(str(exc))
    if not match:
        return None
    try:
        epoch_ms = int(match.group(1))
    except ValueError:
        return None

    moment = time.time() if now is None else now
    # Guard both directions. A clock behind the provider's makes the delta negative,
    # which would otherwise read as "resets in the past" and trip the breaker for a
    # moment rather than for a day.
    delta = epoch_ms / 1000.0 - moment
    if delta <= 0:
        return None
    if delta > MAX_COOLDOWN_S:
        return None
    return delta


#: ``"X-RateLimit-Reset": "1790985600000"`` as it appears in an OpenRouter error body.
_reset_re = re.compile(r'\\?"?X-RateLimit-Reset\\?"?\s*:\s*\\?"?(\d{10,})')


def trip(key: str, exc: BaseException | None = None, *, now: float | None = None) -> float | None:
    """Mark an account exhausted. Returns the cooldown applied, or ``None`` if not.

    Idempotent per account: tripping an already-spent account extends it rather
    than adding a second entry, so a noisy caller cannot inflate the map.
    """
    if not key:
        return None
    moment = time.monotonic() if now is None else now
    # An exception is optional here: a caller may know an account is spent from the
    # quota headers alone, without one in hand. The fallback cooldown covers both
    # "no error object" and "error carried nothing useful".
    delta = (
        reset_at_from_error(exc, now=None if now is None else time.time())
        if exc is not None
        else None
    )
    cooldown = delta if delta is not None else DEFAULT_COOLDOWN_S

    fingerprint = _fingerprint(key)
    with _LOCK:
        previous = _SPENT.get(fingerprint)
        if previous is not None and previous > moment:
            # Two directions have to be guarded, and they fail oppositely, so both
            # are needed -- guarding one is half a guard.
            remaining = previous - moment
            # Two directions, failing oppositely, so both are guarded -- guarding
            # one is half a guard.
            #
            # `delta is None` -- no reset header this time, so nothing new was
            # learned, and a 6-hour guess must not *extend* a deadline the provider
            # said was two minutes away. Without this, one uninformative 429 kept an
            # account suppressed for 21600s after the provider said 120s (measured).
            #
            # `delta` present -- trust it, but never let it cut a known deadline
            # short: a provider reporting a shorter reset for the same account is
            # reporting an inconsistency, not a new fact.
            #
            # The conditional expression is lazy, so `max` is never handed the None.
            cooldown = remaining if delta is None else max(delta, remaining)
        _SPENT[fingerprint] = moment + cooldown
    return cooldown


def clear(key: str | None = None) -> None:
    """Forget one account, or all of them.

    The all-arms form is the control for measuring what the breaker is worth, the
    same way ``CACHE_ENABLED=false`` is the control for the cache.
    """
    with _LOCK:
        if key is None:
            _SPENT.clear()
        else:
            _SPENT.pop(_fingerprint(key), None)


def spent_accounts(*, now: float | None = None) -> list[_BreakerState]:
    """What is currently suppressed. For a diagnostic, never on the hot path."""
    moment = time.monotonic() if now is None else now
    with _LOCK:
        return [
            _BreakerState(fingerprint=fp, until=until, source="cooldown")
            for fp, until in _SPENT.items()
            if until > moment
        ]


def _coerce_body(message: str) -> dict | None:
    """Best-effort extraction of the provider's JSON error body.

    Split out and separately unused at runtime: it exists so the parsing *shape* can
    be tested against a captured body without constructing a live 429, which is
    otherwise very hard to arrange on demand.
    """
    tail = message[message.find('{"error"') :] if '{"error"' in message else ""
    if not tail:
        return None
    try:
        return json.loads(tail)
    except ValueError:
        return None
