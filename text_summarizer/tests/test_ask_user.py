"""``ask_user``: the gate, the pause, and the resume.

**The bar being defended is "ask only when it changes the answer".** An agent
that asks about everything is worse than one that never asks -- every question
costs a turn and trains the user to type "yes" reflexively -- so most of this file
is about questions that are *refused*, and the single most important test here is
``test_a_question_with_an_obvious_default_is_never_asked``.

The pause itself is an ADK primitive rather than anything this code implements,
which means the parts worth testing are the two ends of it: that the turn really
stops, and that the answer comes back as data on the tool call it interrupted.
Both are verified here against a real runner, because a mechanism that does not
survive contact with the runner is not a mechanism.

**The runner these tests use is the resumable ``App``, not a bare ``LlmAgent``,
and that is not a detail.** Requesting a confirmation does not pause a turn: ADK
emits the ``adk_request_confirmation`` event and, with no resumability config,
carries straight on to the next model call -- so the agent asks a question and
answers it itself, and the user is handed a form for a decision already made.
That is not a hypothesis; it is what session
``f0842db4-9d42-4911-8b34-255a3731021f`` did, with the confirmation event and the
next model call eleven milliseconds apart. ``text_summarizer.app`` is what
``adk web`` loads (``AgentLoader`` checks ``app`` before ``root_agent``), so the
tests below drive that, and ``test_a_bare_agent_would_not_pause`` pins the
difference so the wrapper cannot be dropped silently.

Everything here is offline: a stub ``BaseLlm``, ``InMemoryRunner``, no quota.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from text_summarizer import ask_user as mod
from text_summarizer.agent import root_agent
from text_summarizer.ask_user import _clean, ask_user, worth_asking

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


def test_a_long_list_is_still_a_question():
    """The cap is gone, and this is what holds it gone.

    There used to be a five-option ceiling, on the reasoning that past a handful a
    question stops being a decision. That reasoning was about the agent's
    judgement and it produced the worst outcome available for a real list: asked
    *"all the films showing tonight"*, ``ask_user`` **refused the question**, took
    its own ``default``, and answered with one film. It looked like an answer.

    So the gate is now on the *shape* of the question, not its length. What a long
    list costs is a taller card; what the cap cost was a wrong answer.

    ``MAX_OPTION_CHARS`` still bounds each label, which is the bound that was doing
    the real work.
    """
    options = [f"Film {i}" for i in range(11)]
    allowed, reason = worth_asking(
        options=options,
        default=options[0],
        consequence="each one is a film the user might actually want to see",
    )
    assert allowed is True, reason
    assert len(options) == 11

    # And it reaches the UI rather than being answered for the user: an
    # eleven-button card is the whole point, so the option list is not truncated
    # anywhere between the model and the screen.
    cleaned = _clean(
        question="Which film?",
        options=options,
        default=options[0],
        consequence="different showtimes",
    )
    assert cleaned["allowed"] is True
    assert cleaned["options"] == options


def test_a_label_is_still_bounded():
    """What does the bounding, now that the count is not.

    An option is a label in a chat bubble, not a paragraph. Without this, a model
    that pastes a page into an option produces a card nobody can read and a
    default that cannot be matched back.
    """
    long_label = "A" * (mod.MAX_OPTION_CHARS + 200)

    asked: list = []

    class _Ctx:
        function_call_id = "fc1"
        tool_confirmation = None

        def request_confirmation(self, **kwargs):
            asked.append(kwargs)

    result = ask_user(
        question="Which?",
        options=[long_label, "B"],
        default="B",
        consequence="different showtimes",
        tool_context=_Ctx(),
    )
    assert result["status"] == "asked"
    assert asked, "no confirmation requested, so nothing was drawn"
    assert len(result["options"][0]) == mod.MAX_OPTION_CHARS
    assert result["options"][1] == "B"
    # Truncated on the way out, so the label the user reads is the bounded one.
    assert asked[0]["payload"]["options"][0] == "A" * mod.MAX_OPTION_CHARS


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


def _run_agent(*, resumable: bool = True):
    """A runner shaped like the deployed one.

    ``resumable=True`` builds the ``App`` that ``text_summarizer`` exports and
    ``adk web`` loads. ``resumable=False`` builds the bare ``LlmAgent``, which is
    what the runner would have been before the wrapper existed -- and which does
    not pause. It is here so the difference can be *demonstrated* rather than
    asserted from a docstring.
    """
    session_service = InMemorySessionService()
    agent = LlmAgent(name="a", model=_AskThenAnswer(), tools=[ask_user])
    if resumable:
        app = App(
            name="a",
            root_agent=agent,
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
        runner = InMemoryRunner(app=app, app_name="a")
    else:
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


def _setup(*, resumable: bool = True):
    session_service, runner = _run_agent(resumable=resumable)
    session_id = "s1"
    asyncio.run(
        session_service.create_session(app_name="a", user_id="u", session_id=session_id)
    )
    return session_service, runner, session_id


def _first_run(**setup_kwargs):
    session_service, runner, session_id = _setup(**setup_kwargs)
    return session_service, runner, session_id, _collect(
        runner,
        session_service,
        user_id="u",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    )


def _answer_text(events) -> list[str]:
    return [
        part.text
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.text
    ]


def _confirmation_call(events):
    return next(
        part.function_call
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_call
        and part.function_call.name == "adk_request_confirmation"
    )


def test_the_turn_stops_at_the_question_and_waits_for_the_user():
    """The load-bearing test in this file, and the one that was missing.

    ``ask_user`` exists to make the agent *stop*. Emitting a confirmation is not
    stopping: ADK builds the event, marks it long-running, and -- with no
    resumability config -- goes on to the next model call in the same invocation.
    The agent then answers its own question, and the user is shown a form for a
    decision that has already been taken.

    Asserting the *event* exists, as an earlier version of this test did, cannot
    catch that: the event is emitted in both cases, so such a test passes against
    the broken behaviour and the suite reports a working feature while the feature
    does not work. What differs is whether the agent goes on to answer, so that is
    what this asserts.
    """
    _, _, _, events = _first_run()

    confirmations = [
        part.function_call
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_call
        and part.function_call.name == "adk_request_confirmation"
    ]
    assert confirmations, (
        "no adk_request_confirmation event: the UI would not render a prompt at all"
    )

    call = confirmations[0]
    # The UI reads `args.originalFunctionCall` to show what is being confirmed,
    # so its absence would leave the prompt with nothing to describe.
    original = call.args.get("originalFunctionCall")
    assert original, "originalFunctionCall is missing; the UI has nothing to show"
    assert original["name"] == "ask_user"
    assert original["args"]["options"] == ["Cinemark Osasco", "Kinoplex Osasco"]

    # `long_running_tool_ids` is how ADK marks the request as awaiting input.
    long_running = [
        getattr(event, "long_running_tool_ids", None) for event in events
    ]
    assert any(ids and call.id in ids for ids in long_running), (
        "the confirmation event is not marked long-running, so the runner would "
        "not treat the turn as interrupted"
    )

    # ...and the thing that actually matters.
    assert _answer_text(events) == [], (
        "the agent kept going after asking: "
        f"{_answer_text(events)}. It answered its own question and the user was "
        "never given a chance to answer theirs."
    )


def test_a_bare_agent_would_not_pause():
    """Why the ``App`` wrapper exists, demonstrated rather than asserted.

    Without this, dropping ``resumability_config`` is a silent regression: every
    other test in this file still passes, because the confirmation event is still
    emitted and the resume still works. Only the pause stops happening -- the agent
    quietly answers its own question again. So the control is asserted here, on the
    exact same stub model and the exact same runner.
    """
    _, _, _, events = _first_run(resumable=False)

    assert any(
        part.function_call
        and part.function_call.name == "adk_request_confirmation"
        for event in events
        for part in (event.content.parts if event.content else []) or []
    ), "control is wrong: the bare runner did not even emit the confirmation"

    assert _answer_text(events), (
        "the bare runner paused too, so resumability is not what causes the pause "
        "and the App wrapper in text_summarizer/__init__.py is load-bearing for no "
        "reason -- revisit that claim rather than deleting it"
    )


def test_the_answer_comes_back_as_data_and_the_turn_continues():
    """The whole point: ask, receive the choice, keep going.

    Driven with the payload an option button in the patched dev UI sends:
    ``{confirmed: true, payload: {choice: "..."}}`` against the confirmation call's
    own id. ``patch-adk-devui-confirm.py`` is what produces that button, and the
    two are pinned to each other in ``test_adk_devui_confirm_patch.py``.
    """
    session_service, runner, session_id, first = _first_run()
    confirmation = _confirmation_call(first)
    invocation_id = first[-1].invocation_id

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

    texts = _answer_text(resumed)
    assert any("Kinoplex" in t for t in texts), (
        f"the turn did not continue past the question: {texts}"
    )


def test_the_stock_ui_submit_resumes_as_declined_not_as_a_choice():
    """The payload the **unpatched** dev UI sends, and what it must not become.

    The bundled UI prefills its payload textarea with
    ``JSON.stringify(originalFunctionCall.args)`` and posts it untouched, so the
    payload that comes back is the tool's own arguments -- ``question``,
    ``options``, ``default``, ``consequence`` -- with no ``choice`` in it. An
    earlier version of this file fed ``{choice: ...}`` by hand and called that the
    wire shape the UI sends; it is not, and nothing in the suite could tell.

    So the real shape is pinned here. It must resolve to ``default_declined`` --
    the user pressed Submit without ticking Confirm, which is a decline -- and
    critically it must **not** resolve to ``answered_by: "user"``, because
    ``ask_user`` reads ``payload.options[0]`` as a choice when no ``choice`` is
    present. That fallback is right for a client that sends the options list as
    its answer, and wrong for one that sends it as an unedited prefilled form:
    this is the second, and taking the first option would silently answer the
    question on the user's behalf while reporting that they had.
    """
    session_service, runner, session_id, first = _first_run()
    confirmation = _confirmation_call(first)

    resumed = _collect(
        runner,
        session_service,
        user_id="u",
        session_id=session_id,
        invocation_id=first[-1].invocation_id,
        new_message=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=confirmation.id,
                        name="adk_request_confirmation",
                        response={
                            "confirmed": False,
                            "payload": {
                                "question": "Which cinema?",
                                "options": ["Cinemark Osasco", "Kinoplex Osasco"],
                                "default": "Cinemark Osasco",
                                "consequence": "every session time belongs to one "
                                "cinema or the other",
                            },
                        },
                    )
                )
            ],
        ),
    )

    answers = [
        part.function_response.response
        for event in resumed
        for part in (event.content.parts if event.content else []) or []
        if part.function_response and part.function_response.name == "ask_user"
    ]
    assert answers, "ask_user was not re-executed on resume"
    assert answers[0]["answered_by"] == "default_declined", (
        f"the stock UI's submit must read as a decline, not as a choice: {answers[0]}"
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


def test_echoing_the_options_list_is_not_taken_as_a_choice():
    """The ``payload.options`` shape is *our* data, not the user's answer.

    The bundled dev UI prefills its payload textarea with the tool's own
    arguments, so an unedited submit posts ``{question, options, default,
    consequence}`` straight back. An earlier version of this file read
    ``options[0]`` out of that and reported ``answered_by: "user"`` -- inventing an
    answer and attributing it to the person who was never asked. The whole point
    of ``answered_by`` is that it distinguishes *the user chose* from *a default
    was used*, so a shape we cannot interpret has to land on the default.

    This is the control for that: it fails if the fallback is ever reinstated.
    """

    class _EchoedOptions:
        confirmed = True
        payload = {"options": ["Cinemark Osasco", "Kinoplex Osasco"], "default": "x"}

    class _Ctx:
        function_call_id = "fc1"
        tool_confirmation = _EchoedOptions()

    result = ask_user(
        question="Which cinema?",
        options=["Cinemark Osasco", "Kinoplex Osasco"],
        default="Cinemark Osasco",
        consequence="different session times",
        tool_context=_Ctx(),
    )
    assert result["answered_by"] == "default_unrecognised_answer"
    assert result["choice"] == "Cinemark Osasco"


def test_an_option_button_payload_is_taken_as_the_choice():
    """The shape the patched dev UI sends when a button is clicked.

    The counterpart to the test above: the honest shape must still work, or
    "never invent an answer" would have been met by refusing to accept one.
    """

    class _ButtonClicked:
        confirmed = True
        payload = {"choice": "Kinoplex Osasco"}

    class _Ctx:
        function_call_id = "fc1"
        tool_confirmation = _ButtonClicked()

    result = ask_user(
        question="Which cinema?",
        options=["Cinemark Osasco", "Kinoplex Osasco"],
        default="Cinemark Osasco",
        consequence="different session times",
        tool_context=_Ctx(),
    )
    assert result["answered_by"] == "user"
    assert result["choice"] == "Kinoplex Osasco"


# --- wiring -------------------------------------------------------------------


def test_the_tool_is_wired_into_the_agent():
    """Otherwise the whole feature is dead code with a passing test suite."""
    from google.adk.tools import FunctionTool

    assert any(
        isinstance(t, FunctionTool) and t.func is ask_user for t in root_agent.tools
    ), "ask_user is not in root_agent.tools"


def test_the_package_exports_a_resumable_app_wrapping_the_same_agent():
    """The ``App`` is what makes the pause happen, so its absence is the bug.

    ``adk web`` resolves an agent through ``AgentLoader``, which checks ``app``
    before ``root_agent`` -- so this is not decoration on an unused export, it is
    the object the deployment runs. And ``root_agent`` has to stay the same
    object, because ``adk eval`` and ``adk run`` still take that path and a
    wrapped copy would quietly fork the two surfaces.
    """
    from text_summarizer import app

    assert app is not None, (
        "text_summarizer exports no App; adk web falls back to root_agent and "
        "every ask_user turn answers its own question again"
    )
    assert app.root_agent is root_agent
    assert app.resumability_config is not None
    assert app.resumability_config.is_resumable is True


def test_adk_web_actually_loads_the_resumable_app():
    """The loader's ``app``-before-``root_agent`` preference, end to end.

    The export test above proves the object exists; this proves the thing that
    consumes it picks it up. Those are different claims, and only one of them is
    about production -- ``adk eval`` uses ``root_agent`` and would never notice.
    An ADK release that reorders that check would leave the export intact and
    this test would be the only thing that noticed.
    """
    import text_summarizer
    from google.adk.cli.utils.agent_loader import AgentLoader

    agents_dir = str(Path(text_summarizer.__file__).resolve().parents[1])
    loaded = AgentLoader(agents_dir=agents_dir).load_agent("text_summarizer")

    assert isinstance(loaded, App), (
        f"adk web would load a {type(loaded).__name__}, not the resumable App; "
        "confirmations will be emitted but the turn will not pause"
    )
    assert loaded.resumability_config is not None
    assert loaded.resumability_config.is_resumable is True


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