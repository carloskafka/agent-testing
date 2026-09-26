"""Unit tests for the Langfuse trace-identity ``before_agent_callback``.

The callback's whole job is to set OTel attributes on the *currently open* span
before the turn runs, so the assertions here are about the attributes and the
span they land on -- not about Langfuse, which is absent in the unit tests.

Two failure modes are pinned down because both are silent in Langfuse:

* a missing/ended span means the attributes are dropped without error;
* a wrong attribute *name* is accepted by OTel and then ignored by Langfuse's
  ingestion, which is why ``user.id``/``session.id`` are used rather than the
  ``langfuse.``-prefixed names a reader would expect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from text_summarizer import agent as agent_module
from text_summarizer.observability import (
    _SESSION_ID_ATTR,
    _TRACE_NAME_ATTR,
    _USER_ID_ATTR,
    DEFAULT_TRACE_NAME,
    tag_trace_identity,
    trace_name,
)


@pytest.fixture
def captured_spans():
    """Collect spans from a real SDK tracer provider.

    A real provider is the point: ``set_attributes`` on a non-recording span is a
    no-op, which is exactly the bug a mock would hide.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Not set globally: the callback reads whatever tracer is current in the
    # test's own context, which keeps the fixture from leaking into other tests.
    return exporter, provider


def _context(*, user_id="u-1", session_id="s-1", session_user_id=None):
    return SimpleNamespace(
        user_id=user_id,
        session=SimpleNamespace(id=session_id, user_id=session_user_id),
    )


def test_tags_name_user_and_session(captured_spans):
    exporter, provider = captured_spans

    with provider.get_tracer("test").start_as_current_span("agent_run"):
        tag_trace_identity(_context(), None)

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes[_TRACE_NAME_ATTR] == DEFAULT_TRACE_NAME
    assert attributes[_USER_ID_ATTR] == "u-1"
    assert attributes[_SESSION_ID_ATTR] == "s-1"


def test_agent_uses_the_callback(captured_spans):
    """Guard the wiring, not just the function -- a renamed hook fails silently."""
    assert agent_module.root_agent.before_agent_callback is tag_trace_identity


def test_extra_callback_arguments_are_absorbed():
    """ADK passes more than one argument; a strict signature would raise."""
    assert tag_trace_identity(_context(), None) is None


def test_absent_identity_is_omitted_not_emptied(captured_spans):
    """A blank userId/sessionId would filter worse than no value at all."""
    exporter, provider = captured_spans

    with provider.get_tracer("test").start_as_current_span("agent_run"):
        tag_trace_identity(_context(user_id=None, session_id=None), None)

    attributes = exporter.get_finished_spans()[0].attributes
    assert _USER_ID_ATTR not in attributes
    assert _SESSION_ID_ATTR not in attributes


def test_falls_back_to_session_user_id(captured_spans):
    exporter, provider = captured_spans

    with provider.get_tracer("test").start_as_current_span("agent_run"):
        tag_trace_identity(_context(user_id=None, session_user_id="from-session"), None)

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes[_USER_ID_ATTR] == "from-session"


def test_no_active_span_is_a_no_op():
    """Outside any span the callback must return quietly, not raise."""
    assert trace.get_current_span() is trace.INVALID_SPAN
    assert tag_trace_identity(_context(), None) is None


def test_broken_context_does_not_raise(captured_spans):
    """Observability must never be the reason a turn fails."""

    class Hostile:
        @property
        def user_id(self):
            raise RuntimeError("boom")

        @property
        def session(self):
            raise RuntimeError("boom")

    _, provider = captured_spans
    with provider.get_tracer("test").start_as_current_span("agent_run"):
        assert tag_trace_identity(Hostile(), None) is None


def test_trace_name_is_overridable(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACE_NAME", "  custom-name  ")
    assert trace_name() == "custom-name"


def test_blank_trace_name_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACE_NAME", "   ")
    assert trace_name() == DEFAULT_TRACE_NAME
