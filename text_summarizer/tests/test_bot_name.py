"""The bold name stamped at the head of every answer.

Same family as ``test_sources.py`` (the renderer in isolation),
``test_agent_callback.py`` (the callbacks) and ``test_adk_wiring.py`` (a real
``InMemoryRunner``, real events). What is pinned here:

* the label is **derived** from the agent's own name, so there is one source of
  truth, and ``BOT_NAME`` overrides it while ``BOT_NAME=`` turns it off;
* the stamp is emitted exactly once, at the top, and the ``**Sources**`` block
  still comes last (instruction rule 7 puts the source lines at the very end);
* **nothing about a response carrying a function call is touched.** That is the
  load-bearing test in this file -- see ``_response_with_text`` in ``agent.py``:
  ``_finalize_model_response_event`` replaces the event's ``content`` wholesale,
  so a callback that returned a fresh single-part ``Content`` used to *delete*
  the ``function_call`` and the agent silently stopped calling tools for that
  turn;
* a cache hit carries the name too, and the stored note is still byte-identical;
* the metrics never see it.

Everything here runs offline against a stub ``BaseLlm``.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest
from _helpers import (
    SERVED_MODEL,
    _agent_texts,
    _all_text,
    _final_model_text,
    _point_vault_at,
    _StubLlm,
)
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from text_summarizer import agent as agent_module
from text_summarizer.second_brain import source_fingerprint
from text_summarizer.sources import render_sources

ANSWER_WITH_SOURCES = (
    "- Dogs are loyal companions.\n"
    "- Dogs vary widely in size.\n"
    "\n"
    "[obsidian][@@ADK_VAULT@@][@@ADK_MODEL@@][[Dogs Overview]]: same topic\n"
    "\n"
    'Saved to the second brain as "Dogs Summary".'
)
PLAIN_ANSWER = "- Dogs are loyal companions.\n- Dogs vary widely in size."


class _TextAndToolStubLlm(BaseLlm):
    """Emits text *and* a function call in one response, then answers.

    The shape that made the content-clobbering bug reachable: one response
    carrying both. A real model does this ("I'll look that up." then the call),
    and rule 7's "list them at the very END of your answer" makes it rare rather
    than impossible. It calls only on the first turn so the runner terminates.
    """

    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls == 1:
            parts = [
                types.Part(text="Let me check the vault."),
                types.Part(
                    function_call=types.FunctionCall(name="probe", args={"label": "x"}, id="call-1")
                ),
            ]
        else:
            parts = [types.Part(text="- the final answer")]
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            finish_reason=types.FinishReason.STOP,
            model_version=self.model,
        )


def _probe(label: str) -> str:
    """A real function so the runner has something to dispatch and record."""
    return f"probe saw {label}"


def _context(events=None):
    """The smallest ``CallbackContext`` stand-in the callback actually reads."""
    return SimpleNamespace(session=SimpleNamespace(events=list(events or [])))


def _response(*parts, partial: bool = False) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=list(parts)),
        partial=partial,
        model_version=SERVED_MODEL,
    )


def _text(response: LlmResponse) -> str:
    return "".join(p.text for p in (response.content.parts or []) if getattr(p, "text", None))


def _run(agent: LlmAgent, message: str) -> list:
    async def _go():
        runner = InMemoryRunner(agent=agent, app_name="bot_name_test")
        session = await runner.session_service.create_session(
            app_name="bot_name_test", user_id="u1"
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


def _text_only_agent(answer: str) -> LlmAgent:
    return LlmAgent(
        name=agent_module.AGENT_NAME,
        model=_StubLlm(model=SERVED_MODEL, answer=answer),
        after_model_callback=agent_module.render_sources_after_model,
    )


# --- the label -------------------------------------------------------------------


def test_the_default_label_is_readable_and_derived_from_the_agent_name(monkeypatch):
    """``text_summarizer`` is presented as ``Text Summarizer Agent``.

    Derived, not hardcoded: the point is that there is one source of truth, so
    the label cannot drift from ``LlmAgent(name=...)``.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    assert agent_module.bot_name() == "Text Summarizer Agent"
    assert agent_module.bot_name_line() == "**Text Summarizer Agent**"


def test_renaming_the_agent_changes_the_label_with_no_second_edit(monkeypatch):
    """The proof that the label is derived rather than a literal in the code."""
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    monkeypatch.setattr(agent_module, "AGENT_NAME", "research_digest")
    assert agent_module.bot_name() == "Research Digest Agent"


def test_bot_name_overrides_the_label(monkeypatch):
    monkeypatch.setenv(agent_module.BOT_NAME_ENV, "Second Brain")
    assert agent_module.bot_name_line() == "**Second Brain**"


def test_bot_name_set_to_empty_turns_the_stamp_off(monkeypatch):
    """The control an eval comparison needs, like ``CACHE_ENABLED=false``.

    Set-and-empty must differ from unset, which is why the resolver reads
    ``os.environ.get(...) is None`` rather than a falsy check.
    """
    monkeypatch.setenv(agent_module.BOT_NAME_ENV, "")
    assert agent_module.bot_name_line() == ""
    assert agent_module.prepend_bot_name(PLAIN_ANSWER) == PLAIN_ANSWER


def test_the_stamp_is_idempotent(monkeypatch):
    """A second pass must not produce a second name line.

    A visible duplicate of exactly the kind ``AGENTS.md`` gotcha 16 exists to
    prevent, and the same text can legitimately pass through twice.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    once = agent_module.prepend_bot_name(PLAIN_ANSWER)
    assert agent_module.prepend_bot_name(once) == once
    assert once.count("**Text Summarizer Agent**") == 1


def test_strip_only_removes_the_stamp_this_module_writes(monkeypatch):
    """A bold word the model started with is not the stamp, and stays put."""
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    stamped = agent_module.prepend_bot_name(PLAIN_ANSWER)
    assert agent_module.strip_bot_name(stamped) == PLAIN_ANSWER
    assert agent_module.strip_bot_name("**Some Other Bold**\n- x") == ("**Some Other Bold**\n- x")
    assert agent_module.strip_bot_name(PLAIN_ANSWER) == PLAIN_ANSWER


def test_blank_text_is_left_alone(monkeypatch):
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    for value in ("", "   \n "):
        assert agent_module.prepend_bot_name(value) == value


# --- the callback ----------------------------------------------------------------


def test_the_name_is_the_first_line_and_the_sources_block_stays_last(monkeypatch, tmp_path):
    """Both steps, in the documented order, in one response."""
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    _point_vault_at(monkeypatch, tmp_path / "ck", name="ck", modules=("agent",))
    out = agent_module.render_sources_after_model(
        _context(), _response(types.Part(text=ANSWER_WITH_SOURCES))
    )
    final = _text(out)
    assert final.splitlines()[0] == "**Text Summarizer Agent**"
    assert final.splitlines()[1] == ""
    assert final.splitlines()[2] == "- Dogs are loyal companions."
    assert final.rstrip().splitlines()[-1] == 'Saved to the second brain as "Dogs Summary".'
    assert "**Sources**" in final
    assert final.index("**Text Summarizer Agent**") < final.index("**Sources**")
    assert "[@@ADK_VAULT@@]" not in final


def test_a_response_with_no_sources_block_still_gets_the_name(monkeypatch, tmp_path):
    """The uncited case is the one a three-entry callback list would have passed.

    It is asserted because it is the case that hides the ``_stop_on_truthy``
    bug: the Sources renderer returns ``None`` here, so a third list entry would
    have run -- and been the only thing keeping the feature alive.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    _point_vault_at(monkeypatch, tmp_path / "ck", name="ck", modules=("agent",))
    out = agent_module.render_sources_after_model(
        _context(), _response(types.Part(text=PLAIN_ANSWER))
    )
    assert _text(out) == f"**Text Summarizer Agent**\n\n{PLAIN_ANSWER}"


def test_nothing_to_do_returns_none(monkeypatch, tmp_path):
    """No text, a streamed chunk, and the feature off are all pure no-ops.

    Returning a response for any of these would replace the model's own event
    for no reason, and the chunk case would put a whole-answer stamp in the
    middle of a stream.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    _point_vault_at(monkeypatch, tmp_path / "ck", name="ck", modules=("agent",))
    blank = _response(types.Part(text="   "))
    assert agent_module.render_sources_after_model(_context(), blank) is None
    chunk = _response(types.Part(text="- half a sen"), partial=True)
    assert agent_module.render_sources_after_model(_context(), chunk) is None

    monkeypatch.setenv(agent_module.BOT_NAME_ENV, "")
    unchanged = _response(types.Part(text=PLAIN_ANSWER))
    assert agent_module.render_sources_after_model(_context(), unchanged) is None


def test_a_response_carrying_a_function_call_is_left_untouched(monkeypatch, tmp_path):
    """The regression that matters most in this file.

    ``_finalize_model_response_event`` replaces the event's ``content`` with
    whatever the callback returns, so stamping a response that also asks for a
    tool would delete the ``function_call`` -- and the flow dispatches tools on
    exactly that field (``base_llm_flow.py:858``). The agent would stop calling
    tools for that turn, with nothing in the trace to say why.

    Asserted on the *function call surviving*, not on the text, because the
    text was never the thing at risk.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    _point_vault_at(monkeypatch, tmp_path / "ck", name="ck", modules=("agent",))
    original = _response(
        types.Part(text="Let me check the vault."),
        types.Part(function_call=types.FunctionCall(name="probe", args={}, id="call-1")),
    )
    assert agent_module.render_sources_after_model(_context(), original) is None


def test_the_text_part_is_substituted_in_place_and_other_parts_survive(monkeypatch, tmp_path):
    """The second line of defence, independent of the guard above.

    ``_response_with_text`` must not rebuild the response from a single text
    part: correctness should not rest on the caller having remembered to skip
    function-call responses.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    _point_vault_at(monkeypatch, tmp_path / "ck", name="ck", modules=("agent",))
    call = types.Part(function_call=types.FunctionCall(name="probe", args={}, id="c1"))
    out = agent_module._response_with_text(
        _response(types.Part(text="old"), call, types.Part(text="tail")), "new"
    )
    assert out.get_function_calls(), "the function call part was dropped"
    assert _text(out) == "new"


def test_render_sources_is_still_idempotent_with_the_name_present(monkeypatch, tmp_path):
    """A second render pass must reproduce the text byte for byte."""
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    vault = _point_vault_at(monkeypatch, tmp_path / "ck", name="ck", modules=("agent",))
    once = agent_module.render_sources_after_model(
        _context(), _response(types.Part(text=ANSWER_WITH_SOURCES))
    )
    twice = render_sources(_text(once), vault_name="ck", model_name=SERVED_MODEL, vault_root=vault)
    assert twice == _text(once)
    assert twice.startswith("**Text Summarizer Agent**")


# --- a real runner ---------------------------------------------------------------


def test_a_real_turn_puts_the_name_first_and_the_answer_on_one_event(monkeypatch, tmp_path):
    """End to end through ``InMemoryRunner``, against the event *count*.

    Event count, not text: when the answer was emitted twice (gotcha 16) the
    text was right on both copies and the last event -- what a Langfuse trace
    reports -- was correct. Asserting on text alone is the blind spot that hid
    it.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    _point_vault_at(monkeypatch, tmp_path / "ck", name="ck")
    events = _run(_text_only_agent(ANSWER_WITH_SOURCES), "Summarize: dogs are great.")
    texts = _agent_texts(events)
    assert len(texts) == 1, f"the answer was emitted {len(texts)} times: {texts}"
    assert texts[0].splitlines()[0] == "**Text Summarizer Agent**"
    assert _all_text(events).count("**Text Summarizer Agent**") == 1


def test_a_tool_calling_turn_still_runs_its_tool(monkeypatch, tmp_path):
    """The end-to-end version of the §4 regression, and of the duplication guard.

    Two things have to hold on this turn: the tool is dispatched (which is only
    true if the callback left the event's ``function_call`` intact), and exactly
    one message carries the name -- the interim "Let me check the vault." is not
    the answer, so stamping it would print the name twice in a single turn, the
    duplication gotcha 16 exists to prevent.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    _point_vault_at(monkeypatch, tmp_path / "ck", name="ck", modules=("agent",))
    agent = LlmAgent(
        name=agent_module.AGENT_NAME,
        model=_TextAndToolStubLlm(model=SERVED_MODEL),
        tools=[_probe],
        after_model_callback=agent_module.render_sources_after_model,
    )
    events = _run(agent, "Summarize: dogs are great.")
    assert any(_tool_named(e, "probe") for e in events), "the tool was never dispatched"
    texts = _agent_texts(events)
    stamped = [t for t in texts if t.startswith("**Text Summarizer Agent**")]
    assert len(stamped) == 1, f"the name appeared on {len(stamped)} messages: {texts}"
    assert "Let me check the vault." in texts, "the interim text was altered"


def _tool_named(event, name: str) -> bool:
    for part in (event.content.parts if event.content else []) or []:
        response = getattr(part, "function_response", None)
        if response is not None and getattr(response, "name", None) == name:
            return True
    return False


# --- cache hits and notes --------------------------------------------------------


def test_a_cache_hit_carries_the_name_and_leaves_the_note_untouched(monkeypatch, tmp_path):
    """Stamped on the emitted copy; the stored note is never rewritten.

    A hit short-circuits in ``before_model_callback``, before the model runs, so
    no ``after_model_callback`` fires -- which makes this the only place the
    stamp can be added on that path.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    monkeypatch.setenv("CACHE_ENABLED", "true")
    vault = _point_vault_at(monkeypatch, tmp_path)
    prompt = "Summarize: an already-known text."
    stored = "- Known summary line one.\n\n## From the vault\n[[Old Note]]"
    brain = os.path.join(vault, "Second Brain")
    os.makedirs(brain, exist_ok=True)
    path = os.path.join(brain, "2026-09-25 - known.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "---\nsource_fingerprint: "
            + source_fingerprint(prompt)
            + "\n---\n\n"
            + stored
            + "\n\n## Related\n- [[Dogs]]\n"
        )
    before = pathlib.Path(path).read_text(encoding="utf-8")

    agent = LlmAgent(
        name=agent_module.AGENT_NAME,
        model=_StubLlm(model=SERVED_MODEL, answer="SHOULD NOT BE CALLED"),
        before_model_callback=agent_module.cache_hit_before_model,
        after_model_callback=agent_module.render_sources_after_model,
    )
    events = _run(agent, prompt)
    final = _final_model_text(events)
    assert final == f"**Text Summarizer Agent**\n\n{stored}"
    assert "SHOULD NOT BE CALLED" not in _all_text(events)
    assert pathlib.Path(path).read_text(encoding="utf-8") == before


# --- metrics ---------------------------------------------------------------------


def test_the_metrics_never_see_the_name(monkeypatch):
    """A presentation artefact must not move a score.

    ``source_overlap``/``fidelity`` compare the response against the *user's*
    words; ``Text Summarizer Agent`` is not one of them, so an unstripped stamp
    is a small permanent downward bias. ``response_match_score`` is the same
    argument: it is the number the eval loop compares instruction edits on.
    """
    monkeypatch.delenv(agent_module.BOT_NAME_ENV, raising=False)
    emitted = agent_module.prepend_bot_name(
        "- Dogs are loyal.\n- Dogs are social.", name="Text Summarizer Agent"
    )
    user = "Summarize: dogs are loyal and social animals."
    scored = agent_module.strip_bot_name(emitted)
    assert scored == "- Dogs are loyal.\n- Dogs are social."
    plain = agent_module._score_generation(user, scored)
    stamped = agent_module._score_generation(user, emitted)
    assert plain == stamped


def test_bot_name_off_leaves_the_answer_byte_identical_to_the_models(monkeypatch, tmp_path):
    """The off switch is a true no-op, which is what makes it a usable control.

    With the stamp off and nothing to substitute, the callback returns ``None``
    -- the model's own event is delivered untouched. There is no half-on state
    where the response is rebuilt to no purpose.
    """
    monkeypatch.setenv(agent_module.BOT_NAME_ENV, "")
    _point_vault_at(monkeypatch, tmp_path / "ck", name="ck", modules=("agent",))
    original = _response(types.Part(text=PLAIN_ANSWER))
    assert agent_module.render_sources_after_model(_context(), original) is None
    events = _run(_text_only_agent(PLAIN_ANSWER), "Summarize: dogs are great.")
    assert _agent_texts(events) == [PLAIN_ANSWER]


@pytest.mark.parametrize("name", ["Text Summarizer Agent", "Second Brain"])
def test_strip_and_prepend_round_trip_for_any_label(monkeypatch, name):
    monkeypatch.setenv(agent_module.BOT_NAME_ENV, name)
    stamped = agent_module.prepend_bot_name(PLAIN_ANSWER)
    assert stamped.splitlines()[0] == f"**{name}**"
    assert agent_module.strip_bot_name(stamped) == PLAIN_ANSWER
