"""The generation span must carry ``prompt_name`` for the dashboards' prompt split.

Langfuse stores the prompt name on the span as ``langfuse.observation.prompt.name``
and the dashboards group by it, so without this attribute the prompt breakdown is
permanently empty. Setting it is only possible while the generation span is still
open, which is what makes it an ``after_model_callback`` and not a
``before_model_callback`` -- OTel silently drops attributes set on an ended span.

This runs a real ``LlmAgent`` through a real ``InMemoryRunner`` with a stub
``BaseLlm`` and an in-memory span exporter, so it is offline and costs no quota.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from text_summarizer import observability

PROMPT_NAME_ATTR = "langfuse.observation.prompt.name"


class _StubLlm(BaseLlm):
    """A BaseLlm that returns one canned answer and reports a model_version."""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="- stub answer")]),
            finish_reason=types.FinishReason.STOP,
            model_version=self.model,
        )


def _capture_spans(agent: LlmAgent, message: str = "Summarize: hello world") -> list:
    """Run the agent once and return every span the exporter recorded."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = trace.get_tracer_provider()
    trace._TRACER_PROVIDER = provider  # noqa: SLF001 - test-local provider swap
    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        GoogleADKInstrumentor().instrument(tracer_provider=provider)

        async def _go():
            runner = InMemoryRunner(agent=agent, app_name="prompt_test")
            session = await runner.session_service.create_session(
                app_name="prompt_test", user_id="u1"
            )
            async for _ in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=message)]
                ),
            ):
                pass

        asyncio.run(_go())
    finally:
        GoogleADKInstrumentor().uninstrument()
        trace._TRACER_PROVIDER = previous  # noqa: SLF001
    return exporter.get_finished_spans()


def _agent() -> LlmAgent:
    return LlmAgent(
        name="text_summarizer",
        model=_StubLlm(model="stub-model-1"),
        instruction="Summarize.",
        after_model_callback=observability.tag_current_span,
    )


def test_generation_span_carries_prompt_name():
    """The attribute must land on a real generation span, not just be callable."""
    spans = _capture_spans(_agent())
    tagged = [s for s in spans if PROMPT_NAME_ATTR in (s.attributes or {})]
    assert tagged, (
        f"no span carried {PROMPT_NAME_ATTR}; "
        f"spans seen: {[s.name for s in spans]}"
    )
    assert all(
        s.attributes[PROMPT_NAME_ATTR] == observability.DEFAULT_PROMPT_NAME
        for s in tagged
    )


def test_prompt_name_env_override(monkeypatch):
    """LANGFUSE_PROMPT_NAME overrides the default without touching code."""
    monkeypatch.setenv("LANGFUSE_PROMPT_NAME", "custom-prompt")
    assert observability.prompt_name() == "custom-prompt"

    monkeypatch.setenv("LANGFUSE_PROMPT_NAME", "   ")
    assert observability.prompt_name() == observability.DEFAULT_PROMPT_NAME


def test_tag_current_span_is_a_noop_without_a_recording_span():
    """Must never raise, even with no active span (cache-hit / untraced paths)."""
    observability.tag_current_span()
