"""``ask_user``: the gate, the pause, and the resume.

**The bar being defended is "ask only when it changes the answer".** An agent
that asks about everything is worse than one that never asks -- every question
costs a turn and trains the user to type "yes" reflexively -- so most of this file
is about questions that are *refused*, and the single most important test here is
``test_a_question_with_an_obvious_default_is_never_asked``.

The pause itself is an ADK primitive rather than anything this code implements,
which means the parts worth testing are the two ends of it: that
``request_confirmation`` really produces the ``adk_request_confirmation`` event the
bundled dev UI knows how to render, and that the answer comes back as data on the
tool call it interrupted. Both are verified here against a real runner, because a
mechanism that does not survive contact with the UI is not a mechanism.

Everything here is offline: a stub ``BaseLlm``, ``InMemoryRunner``, no quota.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from text_summarizer import ask_user as mod
from text_summarizer.agent import root_agent
from text_summarizer.ask_user import ask_user, worth_asking

# --- the gate -----------------------------------------------------------------


def test_a_question_with_an_obvious_default_is_never_asked():
    """The behaviour the whole module exists to protect.

    One option is not a question. This is the most common shape by a wide margin
    -- the agent finds one cinema, or one payment method -- and it must cost
    nothing at all.
    """
    allowed, reason = worth_asking(
        options=["Cinemark Osasco"],
        default="Cinemark Osasco",
        consequence="the only cinema in town",
    )
    assert allowed is False
    assert "no decision" in reason


def test_an_answer_that_changes_nothing_is_not_worth_a_turn():
    """A question with no consequence is the agent asking whether it is sure."""
    allowed, reason = worth_asking(
        options=["yes", "no"],
        default="yes",
        consequence="   ",
    )
    assert allowed is False
    assert "would not change anything" in reason


def test_a_real_fork_is_allowed():
    allowed, _ = worth_asking(
        options=["Cinemark Osasco", "Kinoplex Osasco"],
        default="Cinemark Osasco",
        consequence="every session time belongs to one cinema or the other",
    )
    assert allowed is True


def test_options_differing_only_in_case_are_one_option():
    """"Cinemark" and "cinemark" are one cinema, and offering both reads as a bug."""
    allowed, _ = worth_asking(
        options=["Cinemark", "cinemark"],
        default="Cinemark",
        consequence="different session times",
    )
    assert allowed is False


def test_too_many_options_is_refused_rather_than_asked():
    """Past MAX_OPTIONS this stops being a decision and becomes a form."""
    options = [f"Cinema {i}" for i in range(mod.MAX_OPTIONS + 1)]
    allowed, reason = worth_asking(
        options=options,
        default=options[0],
        consequence="different session times",
    )
    assert allowed is False
    assert "too many" in reason


def test_a_default_outside_the_options_is_refused():
    """There would be nothing to fall back to, so the turn could not finish."""
    allowed, reason = worth_asking(
        options=["Cinemark", "Kinoplex"],
        default="Some Other Cinema",
        consequence="different session times",
    )
    assert allowed is False
    assert "nothing to fall back to" in reason


# --- the tool ----------------------------------------------------------------


def test_the_gate_is_enforced_in_code_not_in_the_prompt():
    """The model is told when to ask; it is not trusted to decide it matters.

    A stub ``tool_context`` records the confirmation rather than raising, so this
    asserts the tool refused *before* reaching ADK. If the gate were only a rule
    in the instruction, this test could not exist.
    """
    asked: list = []

    class _Ctx:
        function_call_id = "fc1"
        tool_confirmation = None

        def request_confirmation(self, **kwargs):
            asked.append(kwargs)

    result = ask_user(
        question="Which cinema?",
        options=["only one"],
        default="only one",
        consequence="the only cinema",
        tool_context=_Ctx(),
    )
    assert result["status"] == "not_asked"
    assert asked == [], "the tool paused the turn despite refusing to ask"


def test_a_real_question_pauses_the_turn():
    asked: list = []

    class _Ctx:
        function_call_id = "fc1"
        tool_confirmation = None

        def request_confirmation(self, **kwargs):
            asked.append(kwargs)

    result = ask_user(
        question="Which cinema?",
        options=["Cinemark Osasco", "Kinoplex Osasco"],
        default="Cinemark Osasco",
        consequence="every session time belongs to one cinema or the other",
        tool_context=_Ctx(),
    )
    assert result["status"] == "asked"
    assert asked, "no confirmation was requested, so the turn did not pause"
    # The payload is what comes back on resume, so it must carry the options.
    assert asked[0]["payload"]["options"] == ["Cinemark Osasco", "Kinoplex Osasco"]


def test_without_a_turn_context_it_answers_rather_than_stalling():
    """A tool called with no framework to pause on must still finish.

    The alternative is returning a question the caller cannot act on, which is a
    dead turn wearing the costume of a helpful one.
    """
    result = ask_user(
        question="Which cinema?",
        options=["Cinemark", "Kinoplex"],
        default="Kinoplex",
        consequence="different session times",
        tool_context=None,
    )
    assert result["status"] == "answered"
    assert result["choice"] == "Kinoplex"
    assert result["answered_by"] == "default_no_turn_context"


def test_malformed_arguments_are_returned_as_data():
    """The model can read an error and correct it; raising would kill the turn."""
    result = ask_user(
        question="Which?",
        options="Cinemark",  # not a list
        default="Cinemark",
        consequence="x",
    )
    assert "error" in result
    assert "list" in result["error"]


# --- the pause and resume, against a real runner ------------------------------


class _AskThenAnswer(BaseLlm):
    """Calls ``ask_user`` once, then answers -- like the real agent would."""

    model: str = "stub-model"
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls == 1:
            part = types.Part(
                function_call=types.FunctionCall(
                    name="ask_user",
                    args={
                        "question": "Which cinema?",
                        "options": ["Cinemark Osasco", "Kinoplex Osasco"],
                        "default": "Cinemark Osasco",
                        "consequence": "every session time belongs to one cinema",
                    },
                    id="fc1",
                )
            )
        else:
            part = types.Part(text="sessions for Kinoplex")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _run_agent():
    session_service = InMemorySessionService()
    agent = LlmAgent(name="a", model=_AskThenAnswer(), tools=[ask_user])
    runner = InMemoryRunner(agent=agent, app_name="a")
    runner.session_service = session_service
    return session_service, runner


def _collect(runner, session_service, **kwargs):
    async def _go():
        out = []
        async for event in runner.run_async(**kwargs):
            out.append(event)
        return out

    return asyncio.run(_go())


def _setup():
    session_service, runner = _run_agent()
    session_id = "s1"
    asyncio.run(
        session_service.create_session(app_name="a", user_id="u", session_id=session_id)
    )
    return session_service, runner, session_id


def test_the_pause_emits_the_event_the_dev_ui_renders():
    """``request_confirmation`` must produce ``adk_request_confirmation``.

    This is the load-bearing ADK fact, and it is checked against the runner
    rather than read off a docstring: the bundled dev UI keys its confirmation
    form on ``functionCall.name === "adk_request_confirmation"`` and renders
    nothing for any other name. A pause the UI cannot draw is a turn that hangs
    with no way to answer it.
    """
    session_service, runner, session_id = _setup()
    events = _collect(
        runner,
        session_service,
        user_id="u",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    )

    confirmations = [
        part.function_call
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_call
        and part.function_call.name == "adk_request_confirmation"
    ]
    assert confirmations, "no adk_request_confirmation event: the UI would not render a prompt"

    call = confirmations[0]
    # The UI reads `args.originalFunctionCall` to show what is being confirmed,
    # so its absence would leave the prompt with nothing to describe.
    original = call.args.get("originalFunctionCall")
    assert original, "originalFunctionCall is missing; the UI has nothing to show"
    assert original["name"] == "ask_user"
    assert original["args"]["options"] == ["Cinemark Osasco", "Kinoplex Osasco"]

    # `long_running_tool_ids` is how ADK marks the turn as awaiting input.
    long_running = [
        getattr(event, "long_running_tool_ids", None) for event in events
    ]
    assert any(ids and call.id in ids for ids in long_running), (
        "the confirmation event is not marked long-running, so the runner would "
        "not treat the turn as interrupted"
    )


def test_the_answer_comes_back_as_data_and_the_turn_continues():
    """The whole point: ask, receive the choice, keep going.

    Resumed with the wire shape the bundled dev UI actually sends -- read out of
    ``main-*.js``, where ``onSend`` posts ``{confirmed, payload}`` against the
    confirmation call's own id.
    """
    session_service, runner, session_id = _setup()
    events = _collect(
        runner,
        session_service,
        user_id="u",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    )

    confirmation = next(
        part.function_call
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_call
        and part.function_call.name == "adk_request_confirmation"
    )
    invocation_id = events[-1].invocation_id

    resumed = _collect(
        runner,
        session_service,
        user_id="u",
        session_id=session_id,
        invocation_id=invocation_id,
        new_message=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=confirmation.id,
                        name="adk_request_confirmation",
                        response={
                            "confirmed": True,
                            "payload": {"choice": "Kinoplex Osasco"},
                        },
                    )
                )
            ],
        ),
    )

    # The tool saw the choice, and the agent carried on to a real answer.
    tool_results = [
        part.function_response.response
        for event in resumed
        for part in (event.content.parts if event.content else []) or []
        if part.function_response
    ]
    assert any(
        isinstance(r, dict) and r.get("choice") == "Kinoplex Osasco"
        for r in tool_results
    ), f"the choice never reached the tool: {tool_results}"

    texts = [
        part.text
        for event in resumed
        for part in (event.content.parts if event.content else []) or []
        if part.text
    ]
    assert any("Kinoplex" in t for t in texts), (
        f"the turn did not continue past the question: {texts}"
    )


def test_an_answer_we_never_offered_falls_back_to_the_default():
    """The resumed payload is client-supplied, so it cannot be trusted to be one
    of our options -- and inventing a third value is worse than the default."""

    class _Confirmed:
        confirmed = True
        payload = {"choice": "Cine Palace (not offered)"}

    class _Ctx:
        function_call_id = "fc1"
        tool_confirmation = _Confirmed()

    result = ask_user(
        question="Which cinema?",
        options=["Cinemark Osasco", "Kinoplex Osasco"],
        default="Cinemark Osasco",
        consequence="different session times",
        tool_context=_Ctx(),
    )
    assert result["choice"] == "Cinemark Osasco"
    assert result["answered_by"] == "default_unrecognised_answer"


def test_declining_falls_back_to_the_default():
    class _Declined:
        confirmed = False
        payload = {}

    class _Ctx:
        function_call_id = "fc1"
        tool_confirmation = _Declined()

    result = ask_user(
        question="Which cinema?",
        options=["Cinemark Osasco", "Kinoplex Osasco"],
        default="Cinemark Osasco",
        consequence="different session times",
        tool_context=_Ctx(),
    )
    assert result["choice"] == "Cinemark Osasco"
    assert result["answered_by"] == "default_declined"


def test_the_ui_sending_the_options_list_takes_the_first():
    """The bundled UI posts the whole list unless the user types one.

    Read off ``onSend`` in ``main-*.js``: ``JSON.parse(payload)`` falls back to
    ``originalFunctionCall.args``, so a user who hits send without typing produces
    a payload with no ``choice`` in it. Treating that as "no answer" would make
    the form silently useless.
    """

    class _WholeList:
        confirmed = True
        payload = {"options": ["Cinemark Osasco", "Kinoplex Osasco"], "default": "x"}

    class _Ctx:
        function_call_id = "fc1"
        tool_confirmation = _WholeList()

    result = ask_user(
        question="Which cinema?",
        options=["Cinemark Osasco", "Kinoplex Osasco"],
        default="Cinemark Osasco",
        consequence="different session times",
        tool_context=_Ctx(),
    )
    assert result["choice"] == "Cinemark Osasco"
    assert result["answered_by"] == "user"


# --- wiring -------------------------------------------------------------------


def test_the_tool_is_wired_into_the_agent():
    """Otherwise the whole feature is dead code with a passing test suite."""
    from google.adk.tools import FunctionTool

    assert any(
        isinstance(t, FunctionTool) and t.func is ask_user for t in root_agent.tools
    ), "ask_user is not in root_agent.tools"


def test_the_instruction_explains_when_to_ask_and_when_not_to():
    """Both halves, because a rule that only says "ask" produces the worst agent.

    The "do NOT" clause is load-bearing: without it the model reads a permission
    as an instruction, and an agent that asks about everything costs a turn every
    time and trains the user to answer without reading.
    """
    instruction = root_agent.instruction
    assert "ask_user" in instruction
    assert "Do NOT ask when a sensible default is obvious" in instruction


def test_ask_user_is_absent_from_the_optimizer_s_required_rules():
    """Rule 15 is not in ``REQUIRED_RULES``, and should not be -- stated as a test.

    Same reasoning as rule 10: the guard protects rules whose loss no score can
    see. Rule 15's absence moves neither eval criterion, so requiring it would
    reject rewrites for no gain while adding a rule number to maintain.
    """
    from text_summarizer.auto_optimize import REQUIRED_RULES, missing_required_rules

    assert 15 not in REQUIRED_RULES
    # A rewrite that keeps every guarded rule but drops 15 is still compliant --
    # which is the point. (Dropping *all* rules is of course not: asserted here so
    # the guard cannot be quietly broken while making this test pass.)
    kept = "\n".join(f"{n}. rule {n}" for n in REQUIRED_RULES)
    assert missing_required_rules(kept) == [], "a compliant rewrite was rejected"
    # And the guard still bites: dropping a *guarded* rule is reported.
    assert missing_required_rules(kept.replace("7. rule 7", "")) == [7]