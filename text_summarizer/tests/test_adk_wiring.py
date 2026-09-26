"""In-process ADK wiring test: proves the provenance path end to end, offline.

The unit tests in ``test_sources.py`` and ``test_agent_callback.py`` cover the
renderer and the callback in isolation. This file runs a *real* ``LlmAgent``
through a *real* ``InMemoryRunner`` with a stub ``BaseLlm``, so it verifies the
three ADK facts the whole feature rests on, at zero API cost:

1. ``Event.model_version`` really carries the backend that served the call
   (ADK merges non-``None`` ``LlmResponse`` fields into the model event).
2. Content returned from ``after_agent_callback`` really does become the
   agent's emitted response.
3. A ``FunctionTool`` really does receive ``tool_context`` -- which is how
   ``save_summary_to_second_brain`` reads the live model at write time.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import AsyncGenerator

import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from text_summarizer import agent as agent_module
from text_summarizer.sources import MODEL_TOKEN, SOURCES_HEADING, VAULT_TOKEN
from text_summarizer.second_brain import note_provenance

SERVED_MODEL = "gemini-3.5-flash-lite"

MODEL_ANSWER = (
    "- Dogs are domesticated mammals valued for loyalty and companionship.\n"
    "- Dogs come in many breeds with varying size, color, and temperament.\n"
    "- Dogs are social animals that thrive on human and canine interaction.\n"
    "\n"
    f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: same topic\n"
    "\n"
    'Saved to the second brain as "Dogs Summary".'
)


class _StubLlm(BaseLlm):
    """A BaseLlm that returns one canned answer and reports a model_version."""

    answer: str = MODEL_ANSWER
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.answer)]),
            finish_reason=types.FinishReason.STOP,
            model_version=self.model,
        )


class _ToolCallingStubLlm(BaseLlm):
    """Calls one tool on the first turn, then answers -- like the real agent."""

    tool_name: str = "probe"
    tool_args: dict = {"label": "from-stub"}
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls == 1:
            part = types.Part(
                function_call=types.FunctionCall(
                    name=self.tool_name, args=dict(self.tool_args), id="call-1"
                )
            )
        else:
            part = types.Part(text="- stub answer")
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            finish_reason=types.FinishReason.STOP,
            model_version=self.model,
        )


class _RepeatingToolCallingStubLlm(BaseLlm):
    """Calls the tool on every turn, not just the first.

    ``_ToolCallingStubLlm`` only calls on its first invocation, which is right for
    single-turn tests but would hide a second turn's write entirely.
    """

    tool_name: str = "probe"
    tool_args: dict = {"label": "from-stub"}
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls % 2 == 1:
            part = types.Part(
                function_call=types.FunctionCall(
                    name=self.tool_name, args=dict(self.tool_args), id=f"call-{self.calls}"
                )
            )
        else:
            part = types.Part(text="- stub answer")
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            finish_reason=types.FinishReason.STOP,
            model_version=self.model,
        )


def _run(agent: LlmAgent, message: str) -> list:
    """Drive the agent once through InMemoryRunner and return the yielded events."""

    async def _go():
        runner = InMemoryRunner(agent=agent, app_name="wiring_test")
        session = await runner.session_service.create_session(
            app_name="wiring_test", user_id="u1"
        )
        return [
            event
            async for event in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=message)]),
            )
        ]

    return asyncio.run(_go())


def _point_vault_at(monkeypatch, tmp_path, name: str = "ck") -> str:
    """Repoint the vault at a temp dir.

    ``second_brain.VAULT_ROOT`` is read at import time, so the env var alone
    would not move it; both are set.
    """
    from text_summarizer import second_brain

    vault = tmp_path / name
    monkeypatch.setenv("SECOND_BRAIN_VAULT", str(vault))
    monkeypatch.setattr(second_brain, "VAULT_ROOT", str(vault))
    return str(vault)


def _final_model_text(events) -> str:
    texts = [
        "".join(p.text for p in (e.content.parts or []) if getattr(p, "text", None))
        for e in events
        if e.author == "text_summarizer" and e.content and e.content.parts
    ]
    return texts[-1] if texts else ""


def _all_text(events) -> str:
    """Every piece of text on every event, tool payloads included."""
    return "".join(
        p.text
        for e in events
        for p in ((e.content.parts if e.content else None) or [])
        if getattr(p, "text", None)
    )


def test_after_agent_callback_output_becomes_the_response(monkeypatch):
    """Fact 2: the callback's return value is what the caller ends up seeing."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    agent = LlmAgent(
        name="text_summarizer",
        model=_StubLlm(model=SERVED_MODEL),
        after_agent_callback=agent_module.report_scores_after_agent,
    )
    events = _run(agent, "Summarize: dogs are great.")
    final = _final_model_text(events)
    assert SOURCES_HEADING in final
    assert f"[obsidian][ck][{SERVED_MODEL}][[Dogs Overview]]: same topic" in final
    assert VAULT_TOKEN not in final and MODEL_TOKEN not in final
    # The raw model text is still on the preceding event, unmodified.
    assert any(MODEL_TOKEN in _final_model_text([e]) for e in events)


def test_model_version_reaches_the_event(monkeypatch):
    """Fact 1: the served model is readable off the session's model event."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    agent = LlmAgent(name="text_summarizer", model=_StubLlm(model=SERVED_MODEL))
    events = _run(agent, "Summarize: dogs are great.")
    with_model = [e for e in events if e.model_version == SERVED_MODEL]
    assert with_model, [e.model_version for e in events]


def test_fallback_model_name_is_what_gets_rendered(monkeypatch):
    """Whatever the backend reports is what lands in the block -- no lookup table."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    served = "openrouter/qwen/qwen3.8-27b:free"
    agent = LlmAgent(
        name="text_summarizer",
        model=_StubLlm(model=served),
        after_agent_callback=agent_module.report_scores_after_agent,
    )
    final = _final_model_text(_run(agent, "Summarize: dogs are great."))
    assert f"[obsidian][ck][{served}]" in final
    assert SERVED_MODEL not in final


def test_function_tool_receives_tool_context_with_live_provenance(monkeypatch):
    """Fact 3: inside a tool call, the served model is resolvable."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    seen: dict = {}

    def probe(label: str, tool_context=None) -> str:
        seen.update(note_provenance(tool_context))
        return f"ok {label}"

    tool = agent_module.FunctionTool(probe)
    # `tool_context` is framework-supplied and must not reach the model's schema.
    schema = tool._get_declaration().parameters_json_schema or {}
    assert list(schema.get("properties", {})) == ["label"]

    agent = LlmAgent(
        name="text_summarizer",
        model=_ToolCallingStubLlm(model=SERVED_MODEL, tool_name="probe"),
        tools=[tool],
    )
    _run(agent, "Summarize: dogs are great.")
    assert seen.get("generated_by_model") == SERVED_MODEL
    assert seen.get("generated_in_vault") == "ck"


def test_note_provenance_from_a_live_session(monkeypatch):
    """The same helper, fed a real session, resolves the served model."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    agent = LlmAgent(name="text_summarizer", model=_StubLlm(model=SERVED_MODEL))
    events = _run(agent, "Summarize: dogs are great.")
    provenance = note_provenance(
        type("Ctx", (), {"session": type("S", (), {"events": events})()})()
    )
    assert provenance["generated_by_model"] == SERVED_MODEL
    assert provenance["generated_in_vault"] == "ck"


def test_cache_hit_replays_the_stored_note_verbatim(monkeypatch, tmp_path):
    """A fingerprint match short-circuits the model and the callback."""
    from text_summarizer.second_brain import source_fingerprint

    vault = _point_vault_at(monkeypatch, tmp_path)
    prompt = "Summarize: an already-known text."
    stored = "- Known summary line one.\n\n## From the vault\n[[Old Note]]"
    brain = os.path.join(vault, "Second Brain")
    os.makedirs(brain, exist_ok=True)
    with open(os.path.join(brain, "2026-09-25 - known.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "---\nsource_fingerprint: "
            + source_fingerprint(prompt)
            + "\n---\n\n"
            + stored
            + "\n\n## Related\n- [[Dogs]]\n"
        )
    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("VAULT_NAME", "ck")

    agent = LlmAgent(
        name="text_summarizer",
        model=_StubLlm(model=SERVED_MODEL, answer="SHOULD NOT BE CALLED"),
        before_model_callback=agent_module.cache_hit_before_model,
        after_agent_callback=agent_module.report_scores_after_agent,
    )
    events = _run(agent, prompt)
    final = _final_model_text(events)
    assert final.strip() == stored
    assert SOURCES_HEADING not in final
    every_text = "".join(
        p.text
        for e in events
        for p in ((e.content.parts if e.content else None) or [])
        if getattr(p, "text", None)
    )
    assert "SHOULD NOT BE CALLED" not in every_text


def test_saved_note_records_provenance_in_frontmatter(monkeypatch, tmp_path):
    from text_summarizer import second_brain

    vault = _point_vault_at(monkeypatch, tmp_path)
    monkeypatch.setenv("VAULT_NAME", "ck")

    # Direct call without a tool context: the served model cannot be resolved,
    # so the field is omitted rather than invented.
    second_brain.save_summary_to_second_brain(
        title="Provenance Probe", summary_content="- x", topics="Dogs"
    )
    notes = list((__import__("pathlib").Path(vault) / "Second Brain").glob("*.md"))
    assert notes, "note was not written"
    body = notes[0].read_text(encoding="utf-8")
    assert 'generated_in_vault: "ck"' in body
    assert "generated_by_model" not in body


def _persisting_agent(**kwargs) -> LlmAgent:
    """An agent that calls the real save tool on turn 1, then answers.

    ``model=...`` overrides the stub, so a test can watch for a model call that
    should never happen.
    """
    from text_summarizer import second_brain

    kwargs.setdefault(
        "model",
        _ToolCallingStubLlm(
            model=SERVED_MODEL,
            tool_name="save_summary_to_second_brain",
            tool_args={
                "title": "Cache Key Probe",
                "summary_content": "- the stored answer",
                "topics": "Dogs",
            },
        ),
    )
    return LlmAgent(
        name="text_summarizer",
        tools=[agent_module.FunctionTool(second_brain.save_summary_to_second_brain)],
        **kwargs,
    )


def _repeating_persisting_agent(**kwargs) -> LlmAgent:
    """As :func:`_persisting_agent`, but persists on every turn."""
    from text_summarizer import second_brain

    kwargs.setdefault(
        "model",
        _RepeatingToolCallingStubLlm(
            model=SERVED_MODEL,
            tool_name="save_summary_to_second_brain",
            tool_args={
                "title": "Cache Key Probe",
                "summary_content": "- the stored answer",
                "topics": "Dogs",
            },
        ),
    )
    return LlmAgent(
        name="text_summarizer",
        tools=[agent_module.FunctionTool(second_brain.save_summary_to_second_brain)],
        **kwargs,
    )


def test_written_fingerprint_is_the_key_the_cache_looks_up(monkeypatch, tmp_path):
    """The invariant that was broken: write key == lookup key.

    The note used to be fingerprinted from a model-supplied ``source_text`` while
    the cache looked the turn up by the fingerprint of the *user's prompt*. The
    model paraphrased that argument differently on every run, so the two could
    never agree and the cache never hit -- once per turn, at the cost of the whole
    turn. This walks the real path (agent -> tool -> note -> lookup) and asserts
    the stored fingerprint is the one a repeat of the same prompt searches for.
    """
    from text_summarizer.second_brain import find_cached_summary, source_fingerprint

    _point_vault_at(monkeypatch, tmp_path)
    monkeypatch.setenv("VAULT_NAME", "ck")
    monkeypatch.setenv("CACHE_ENABLED", "true")

    prompt = "Summarize: dogs are great."
    _run(_persisting_agent(), prompt)

    # A repeat of the same prompt, in a brand new session, must find the note.
    assert find_cached_summary(prompt) is not None
    # ...and the key it used is the prompt's own fingerprint, not anything the
    # model produced along the way.
    brain = tmp_path / "ck" / "Second Brain"
    stored = [
        p.read_text(encoding="utf-8")
        for p in brain.glob("*.md")
        if "source_fingerprint" in p.read_text(encoding="utf-8")
    ]
    assert stored, "no note recorded a fingerprint"
    assert f"source_fingerprint: {source_fingerprint(prompt)}" in stored[0]


def test_second_identical_turn_is_served_from_the_vault(monkeypatch, tmp_path):
    """End to end: the second identical prompt costs zero model calls.

    This is the user-visible bug -- the same question sent in two new chats took
    the same ~12s both times. The stub model would answer with a sentinel, so any
    call reaching it is visible in the output.
    """
    _point_vault_at(monkeypatch, tmp_path)
    monkeypatch.setenv("VAULT_NAME", "ck")
    monkeypatch.setenv("CACHE_ENABLED", "true")

    prompt = "Summarize: dogs are great."
    first = _run(
        _persisting_agent(before_model_callback=agent_module.cache_hit_before_model),
        prompt,
    )
    assert "MUST NOT RUN ON THE SECOND TURN" not in _all_text(first)

    second = _run(
        _persisting_agent(
            model=_StubLlm(
                model=SERVED_MODEL, answer="MUST NOT RUN ON THE SECOND TURN"
            ),
            before_model_callback=agent_module.cache_hit_before_model,
        ),
        prompt,
    )
    assert "MUST NOT RUN ON THE SECOND TURN" not in _all_text(second)
    # The stored note is replayed instead.
    assert "- the stored answer" in _all_text(second)


def test_cache_is_not_consulted_again_after_the_tool_writes(monkeypatch, tmp_path):
    """A cache hit on the *second* model call would truncate the turn.

    The agent persists its note from inside the turn, so by the follow-up model
    call -- the one that writes the actual answer -- the fingerprint for the
    current prompt is already on disk. Replaying it there would return the note
    instead of the answer. The stub answers "- stub answer" on that call, so the
    sentinel distinguishes the two outcomes.
    """
    _point_vault_at(monkeypatch, tmp_path)
    monkeypatch.setenv("VAULT_NAME", "ck")
    monkeypatch.setenv("CACHE_ENABLED", "true")

    events = _run(
        _persisting_agent(before_model_callback=agent_module.cache_hit_before_model),
        "Summarize: dogs are great.",
    )
    text = _all_text(events)
    assert "- stub answer" in text, "follow-up model call was skipped by the cache"
    assert "- the stored answer" not in text, "the note was replayed instead"


def test_turn_after_a_cached_turn_is_not_short_circuited(monkeypatch, tmp_path):
    """A new user message in the same session must reach the model.

    Guards the positional check in ``_is_first_model_call_this_turn``: prior turns'
    events stay on the session, so a naive "does the session contain a model event"
    test would wrongly disable the cache for every turn after the first.
    """
    from text_summarizer.second_brain import source_fingerprint

    _point_vault_at(monkeypatch, tmp_path)
    monkeypatch.setenv("VAULT_NAME", "ck")
    monkeypatch.setenv("CACHE_ENABLED", "true")

    async def _go():
        runner = InMemoryRunner(
            agent=_repeating_persisting_agent(
                before_model_callback=agent_module.cache_hit_before_model
            ),
            app_name="wiring_test",
        )
        session = await runner.session_service.create_session(
            app_name="wiring_test", user_id="u1"
        )
        first = [
            e
            async for e in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="Summarize: dogs are great.")]
                ),
            )
        ]
        second = [
            e
            async for e in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="Now summarize the cats.")]
                ),
            )
        ]
        return first, second

    first, second = asyncio.run(_go())
    # Turn 1 wrote a note for its own prompt; the stub's answer reached the caller.
    assert "- stub answer" in _all_text(first)
    # Turn 2 is a different prompt, so it is a miss: the model runs and the tool
    # persists a note keyed on the new prompt. If the positional check wrongly
    # disabled the cache after the first turn, turn 2 would replay turn 1's note
    # and no new fingerprint would appear.
    assert "- stub answer" in _all_text(second)
    brain = tmp_path / "ck" / "Second Brain"
    bodies = "\n".join(p.read_text(encoding="utf-8") for p in brain.glob("*.md"))
    assert f"source_fingerprint: {source_fingerprint('Now summarize the cats.')}" in bodies


def test_tool_written_note_records_the_live_served_model(monkeypatch, tmp_path):
    """The full path: agent -> tool -> note frontmatter, with the real model."""
    from text_summarizer import second_brain

    vault = _point_vault_at(monkeypatch, tmp_path)
    monkeypatch.setenv("VAULT_NAME", "ck")
    agent = LlmAgent(
        name="text_summarizer",
        model=_ToolCallingStubLlm(
            model=SERVED_MODEL,
            tool_name="save_summary_to_second_brain",
            tool_args={
                "title": "Provenance Probe",
                "summary_content": "- dogs are great",
                "topics": "Dogs",
            },
        ),
        tools=[agent_module.FunctionTool(second_brain.save_summary_to_second_brain)],
    )
    _run(agent, "Summarize: dogs are great.")
    notes = list((__import__("pathlib").Path(vault) / "Second Brain").glob("*.md"))
    assert notes, "note was not written"
    body = notes[0].read_text(encoding="utf-8")
    assert f'generated_by_model: "{SERVED_MODEL}"' in body
    assert 'generated_in_vault: "ck"' in body


@pytest.mark.parametrize("root", ["/", "", "."])
def test_vault_root_without_a_usable_basename(root):
    from text_summarizer.sources import resolve_vault_name

    identity = resolve_vault_name(env={}, vault_root=root)
    assert identity.name in {"vault", os.path.basename(os.path.abspath(root))} or (
        identity.name == "unknown"
    )
