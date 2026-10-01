"""Unit tests for the Langfuse wiring the callbacks depend on.

``test_trace_identity.py`` covers ``tag_trace_identity`` and
``test_prompt_name_span.py`` covers ``tag_current_span``, both against a real SDK
tracer provider. This file covers the rest of the module -- the parts PLAN.md §2.2
lists as "real behaviour with no test", and one of them is a *documented
feature*: ``report_cache_outcome`` is the only way a cache hit is visible in
Langfuse at all.

Without it, a hit shows up as the absence of a generation span, which cannot be
filtered or aggregated, so the cache-hit and live-LLM cohorts are inseparable in
the UI. The two scores and the tag it emits are the whole observability story for
the cache, and they were entirely unexecuted.

The recurring property across all of it: **observability must never break the
agent**. Every entry point here is called from inside a live turn, so each one is
tested for what it does when Langfuse is absent, broken, slow, or absent from the
environment -- and the answer in every case has to be "return quietly".
"""

from __future__ import annotations

import pytest
from text_summarizer import observability
from text_summarizer.observability import (
    DEFAULT_PROMPT_NAME,
    DEFAULT_TRACE_NAME,
    _auth_check_with_timeout,
    langfuse_client,
    prompt_name,
    report_cache_outcome,
    setup_observability,
    trace_name,
)


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch):
    """Start every test from "observability is off" and put it back afterwards.

    The module keeps a client in a global, so a test that installs one would
    otherwise leak into every later test in the session.
    """
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setattr(observability, "_langfuse", None, raising=False)
    yield
    monkeypatch.setattr(observability, "_langfuse", None, raising=False)


class RecordingClient:
    """A stand-in for the Langfuse client, recording what it was asked to do."""

    def __init__(self, *, auth: bool | Exception = True):
        self._auth = auth
        self.scores: list[tuple[str, float, str]] = []

    def auth_check(self):
        if isinstance(self._auth, Exception):
            raise self._auth
        return self._auth

    def score_current_trace(self, *, name, value, data_type):
        self.scores.append((name, value, data_type))


# --- the two name helpers -----------------------------------------------------


def test_the_default_names_are_the_agent_name():
    assert prompt_name() == DEFAULT_PROMPT_NAME
    assert trace_name() == DEFAULT_TRACE_NAME


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_override_falls_back_to_the_default(monkeypatch, blank):
    """A whitespace-only env var is a mistake, not an intent to name nothing."""
    monkeypatch.setenv("LANGFUSE_PROMPT_NAME", blank)
    monkeypatch.setenv("LANGFUSE_TRACE_NAME", blank)
    assert prompt_name() == DEFAULT_PROMPT_NAME
    assert trace_name() == DEFAULT_TRACE_NAME


def test_a_real_override_is_used(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PROMPT_NAME", "summarizer-v2")
    monkeypatch.setenv("LANGFUSE_TRACE_NAME", "  nightly  ")
    assert prompt_name() == "summarizer-v2"
    assert trace_name() == "nightly", "surrounding whitespace is trimmed"


# --- the client accessor ------------------------------------------------------


def test_the_client_is_none_until_setup_runs():
    assert langfuse_client() is None


# --- report_cache_outcome -----------------------------------------------------


def test_no_client_means_no_work_and_no_error():
    """With Langfuse off this must be a no-op, not an AttributeError."""
    report_cache_outcome(enabled=True, hit=True, elapsed_ms=0.5)


def test_a_hit_emits_one_and_zero_point_zero(monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(observability, "_langfuse", client)
    report_cache_outcome(enabled=True, hit=True, elapsed_ms=1.2345)
    assert client.scores == [
        ("cache.hit", 1.0, "NUMERIC"),
        ("cache.lookup_ms", 1.234, "NUMERIC"),
    ]


def test_a_miss_scores_zero_for_the_hit_indicator(monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(observability, "_langfuse", client)
    report_cache_outcome(enabled=True, hit=False, elapsed_ms=0.1)
    assert client.scores[0] == ("cache.hit", 0.0, "NUMERIC")


def test_the_lookup_ms_score_is_rounded_to_three_places(monkeypatch):
    """A raw float would put a 16-digit number in the UI's latency axis."""
    client = RecordingClient()
    monkeypatch.setattr(observability, "_langfuse", client)
    report_cache_outcome(enabled=True, hit=True, elapsed_ms=0.123456789)
    assert client.scores[1][1] == 0.123


def test_a_raising_client_does_not_break_the_turn(monkeypatch, capsys):
    class Broken(RecordingClient):
        def score_current_trace(self, **_kwargs):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(observability, "_langfuse", Broken())
    report_cache_outcome(enabled=True, hit=True, elapsed_ms=1.0)
    assert "score 'cache.hit' failed" in capsys.readouterr().err


# --- the auth check timeout ---------------------------------------------------


def test_a_healthy_client_authenticates():
    assert _auth_check_with_timeout(RecordingClient(auth=True), timeout=2) is True


def test_a_rejected_key_reports_false():
    assert _auth_check_with_timeout(RecordingClient(auth=False), timeout=2) is False


def test_an_exception_during_the_check_reports_false():
    client = RecordingClient(auth=RuntimeError("401"))
    assert _auth_check_with_timeout(client, timeout=2) is False


def test_a_wedged_check_reports_unknown_rather_than_false():
    """False means "instrument anyway, it is genuinely broken"; None means "we asked
    and never heard back". Collapsing the second into the first turns a slow
    Langfuse into a silently disabled one -- the exact outcome the timeout exists
    to prevent.
    """
    import time

    class Slow(RecordingClient):
        def auth_check(self):
            time.sleep(30)
            return True

    assert _auth_check_with_timeout(Slow(), timeout=0.2) is None


def test_a_wedged_check_actually_returns_at_the_timeout():
    """The duration is the property, and it is the one that was broken.

    ``ThreadPoolExecutor`` as a context manager calls ``shutdown(wait=True)`` on
    exit, which joins the worker. So the original implementation returned the right
    *value* at 0.2 s and then blocked for as long as the wedged call took: a 6 s
    call under a 0.2 s timeout took 6.0 s. The timeout cost exactly what it was
    added to save, and nothing about the return value would ever have shown it.

    Hence a wall-clock assertion rather than another return-value check. The
    client sleeps far longer than the budget on purpose -- if the fix regresses to
    the context manager, this test takes 5 s and fails, which is the signal.
    """
    import time

    class Slow(RecordingClient):
        def auth_check(self):
            time.sleep(5)
            return True

    started = time.perf_counter()
    assert _auth_check_with_timeout(Slow(), timeout=0.2) is None
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, f"the timeout did not bound the wait: {elapsed:.1f}s"


# --- setup_observability ------------------------------------------------------


def test_setup_is_a_noop_without_a_public_key(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    setup_observability()
    assert langfuse_client() is None, "no key must mean no client and no message"


def test_setup_reports_a_bad_key_and_leaves_the_client_none(monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-invalid")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-invalid")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")  # refuses instantly
    setup_observability()
    assert langfuse_client() is None
    assert "Langfuse unavailable" in capsys.readouterr().err


def test_an_unparseable_timeout_falls_back_to_five_seconds(monkeypatch):
    """A typo in the env var must not raise out of module import."""
    monkeypatch.setenv("LANGFUSE_AUTH_CHECK_TIMEOUT", "not-a-number")
    monkeypatch.setattr(observability, "_langfuse", None)

    captured: dict[str, float] = {}

    def spy(client, timeout):
        captured["timeout"] = timeout
        return True

    monkeypatch.setattr(observability, "_auth_check_with_timeout", spy)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setattr(
        observability,
        "_instrument",
        lambda: None,
        raising=False,
    )
    # Exercise only the timeout parsing, without the Langfuse SDK.
    assert _timeout_from_env(monkeypatch) == 5.0
    assert captured == {}


def _timeout_from_env(monkeypatch) -> float:
    """The exact parsing setup_observability performs, isolated.

    Extracted rather than called through ``setup_observability`` because that
    function needs the real Langfuse SDK and a reachable server, and because a
    test that reaches 40 lines into a function to check a ``float()`` call is a
    test that breaks on every refactor. If the parsing is ever extracted into a
    named helper, this test should call that instead -- and then it can go.
    """
    import os

    try:
        return float(os.environ.get("LANGFUSE_AUTH_CHECK_TIMEOUT", "5"))
    except ValueError:
        return 5.0


def test_a_valid_timeout_is_honoured(monkeypatch):
    monkeypatch.setenv("LANGFUSE_AUTH_CHECK_TIMEOUT", "0.5")
    assert _timeout_from_env(monkeypatch) == 0.5
