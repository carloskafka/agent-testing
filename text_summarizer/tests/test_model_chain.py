"""A conversation must not be offered to a model that will reject its history.

The live failure this covers is in ``model_chain``'s module docstring: session
``bc4de8b5-535f-41ce-b35b-775b297df039``, turn *"What did you learn yesterday?"*,
four spans deep, dead on a Gemini 400 complaining about a ``functionCall`` that
Nemotron had produced two calls earlier.

What makes it worth a test file rather than a comment is that **every part of the
guard could be present and the bug would still be live**:

- the predicate could look at the wrong thing and never fire;
- the selection could be made once at construction rather than per call, so it
  would be right for the first model call of every turn and wrong for the rest --
  which is precisely the shape that passes a test driving a single call;
- the sub-chain could be built but never actually *used*, because the wrapper
  delegated to the full chain anyway.

So the tests here drive a **sequence of calls sharing one growing conversation**,
which is the only shape in which the defect exists. One call cannot show it: a
single call has no foreign history to protect.

Two conventions worth stating, because both are easy to get backwards:

**The stub that fails is the primary.** ``HistorySafeFallbackModel``'s entire
purpose is to keep a signature-requiring model out of a foreign conversation, so a
test whose primary *accepts* the history would pass against code that never
selects anything. :class:`RejectingGemini` raises the real
``INVALID_ARGUMENT`` / 400 pair on every call, which is what makes an unguarded
chain fail loudly here rather than quietly.

**The assertion is on which models were tried, not on the outcome.** A stub that
returns an error is not enough on its own: the same 400 propagates whether the
guard fired and the fallback then served the turn, or whether nothing was
attempted at all. :func:`_calls` is the record the tests read, so "Gemini was
never asked" is asserted directly.
"""

from __future__ import annotations

import asyncio

import pytest
from google.adk.models import BaseLlm, LlmCapabilities, LlmRequest, LlmResponse
from google.genai import types
from google.genai.errors import ClientError
from text_summarizer.model_chain import (
    HistorySafeFallbackModel,
    history_needs_signed_calls,
    signature_free_chain,
)

#: The message the live turn died on, trimmed. Matched rather than matched-
#: loosely so a stub that stopped being faithful to the real error fails here
#: instead of quietly becoming a model that tolerates what Gemini rejects.
MISSING_SIGNATURE = (
    "Function call is missing a thought_signature in functionCall parts."
)

PRIMARY = "gemini-test-primary"
BACKUP_A = "openrouter/backup-a:free"
BACKUP_B = "openrouter/backup-b:free"


class FailingModel(BaseLlm):
    """A backend that records the request and then fails with a given status.

    Both statuses this repo has seen live are reproduced by the same stub, because
    they are the two halves of the failure and they behave differently:

    * **503** -- what Gemini returned on the live turn's second call, and what sent
      it to the fallback tier. Retriable, so the chain moves on and the turn
      completes. This is what a *healthy* fallback looks like.
    * **400** -- what Gemini returned on the third call, having been handed a
      history containing Nemotron's unsigned ``functionCall``. **Not** retriable,
      so it propagates and the turn dies.

    Both are raised as ``ClientError``, which carries ``code`` -- the attribute
    ``FallbackModel._status_code`` reads to decide whether to move on. The stub
    therefore reproduces the *decision* as well as the error, which is what makes
    "the guard fired" distinguishable from "the guard is absent".
    """

    calls: list[object] = []
    status: int = 503

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        self.calls.append(llm_request)
        _attempted.append(self.model)
        message = (
            MISSING_SIGNATURE
            if self.status == 400
            else "The model is overloaded. Please try again later."
        )
        raise ClientError(
            self.status,
            {"error": {"code": self.status, "message": message, "status": "UNAVAILABLE" if self.status != 400 else "INVALID_ARGUMENT"}},
        )
        # Unreachable, and load-bearing: a `raise` alone leaves this a coroutine,
        # which FallbackModel cannot drive (`async for` over a coroutine is a
        # TypeError). The `yield` is what makes the function an async generator.
        yield

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)


class RecordingModel(BaseLlm):
    """A backend that answers, and remembers that it was asked."""

    calls: list[object] = []
    answer: str = "ok"

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        self.calls.append(llm_request)
        _attempted.append(self.model)
        # A `yield` rather than a `return`: BaseLlm.generate_content_async is an
        # async *generator*, and ADK drives it with `async for`.
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.answer)]),
            model_version=self.model,
        )

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)


#: Every model asked, in order, across the whole test. Appended to by each stub
#: as it is called, which is why it is a flat list rather than a per-stub flag:
#: the live failure was three *successive* calls on one turn, and a record that
#: could only say "was the primary used at all" would collapse those three into
#: one and could not tell the fixed code from the broken code.
_attempted: list[str] = []


def _calls() -> list[str]:
    """The names of the models that were actually asked, in order.

    Read off the stubs rather than off a return value, because the failure mode
    this file exists to catch produces no return value at all: an unguarded
    chain raises, and a chain that selected the wrong sub-chain raises on a
    different model. What was *attempted* is the only thing that distinguishes
    "the guard worked" from "the guard is absent and nothing noticed".
    """
    return list(_attempted)


@pytest.fixture(autouse=True)
def _fresh_stubs():
    """Reset the attempted-call log between tests.

    Autouse because every test here asserts on what was attempted, and a record
    left over from a previous test would make an ordering-dependent suite that
    fails in a way nobody can reproduce.
    """
    _attempted.clear()
    yield
    _attempted.clear()


def _chain(primary_status: int = 503, **kwargs) -> HistorySafeFallbackModel:
    """A chain shaped like the deployed one: Gemini, then two backups.

    Args:
        primary_status: What the primary fails with. 503 (the default) is the
            transient overload that moves the chain on to a backup; 400 is the
            non-retriable rejection that ends a turn, and is what a conversation
            containing an unsigned call provokes on the live path.
    """
    primary = FailingModel(model=PRIMARY, calls=[], status=primary_status)
    backups = [
        RecordingModel(model=BACKUP_A, calls=[]),
        RecordingModel(model=BACKUP_B, calls=[]),
    ]
    defaults = {
        "models": [primary, *backups],
        "unsigned_history_models": backups,
    }
    return HistorySafeFallbackModel(**{**defaults, **kwargs})


# --- the predicate -------------------------------------------------------------


def test_a_conversation_with_no_function_calls_is_clean():
    """The common case: a first model call has no history at all.

    If this returned true, every ordinary turn would be handed to the fallback
    tier and the primary would be used only when that failed -- inverting the
    chain's order, which is a routing decision rather than a formatting one.
    """
    request = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="hi")])])

    assert history_needs_signed_calls(request) is False


def test_a_gemini_function_call_is_not_dirty():
    """A signed ``functionCall`` is exactly what Gemini expects to see.

    The first call of the live turn produced one, and the second call went
    straight back to Gemini and worked. So a signed call must not cost the
    primary its place -- otherwise the guard would degrade every turn the moment
    the model used a tool, which is most of them.
    """
    request = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part(text="hi")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="current_datetime", args={}),
                        thought_signature=b"signed",
                    )
                ],
            ),
        ]
    )

    assert history_needs_signed_calls(request) is False


def test_an_unsigned_function_call_is_dirty():
    """The live shape: Nemotron's ``read_day_digest`` call, no signature.

    This is the whole defect in one assertion. A model that does not support
    ``thought_signature`` cannot produce one, so any tool call from the fallback
    tier makes the conversation unreadable to the primary.
    """
    request = LlmRequest(
        contents=[
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="read_day_digest", args={"day": "2026-09-30"}),
                    )
                ],
            )
        ]
    )

    assert history_needs_signed_calls(request) is True


def test_an_empty_signature_counts_as_absent():
    """``b""`` is not a signature, and it is what a provider sends when it has the
    field but nothing to put in it.

    A truthiness check written as ``is not None`` would treat this as signed and
    let the conversation through to a primary that rejects it -- the same dead
    turn, one field-width further away.
    """
    request = LlmRequest(
        contents=[
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="read_day_digest", args={}),
                        thought_signature=b"",
                    )
                ],
            )
        ]
    )

    assert history_needs_signed_calls(request) is True


def test_a_plain_text_part_is_never_a_signature_bearing_one():
    """The narrowness, pinned.

    Text parts have no ``thought_signature`` either, so a predicate that checked
    "any part without a signature" would call every conversation foreign. This
    is the assertion that keeps the guard from being *so* conservative that the
    primary stops being used.
    """
    request = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part(text="a long question with no tools")]),
            types.Content(role="model", parts=[types.Part(text="an answer")]),
        ]
    )

    assert history_needs_signed_calls(request) is False


# --- the selection -------------------------------------------------------------


def test_a_clean_conversation_still_starts_at_the_primary():
    """The guard must not cost the ordinary path anything.

    Gemini is asked first and, here, fails -- so this also pins that the backup
    tier is reached on a clean conversation. A guard that skipped the primary
    outright would pass every other test in this file while quietly making the
    free tier the model that answers every turn.
    """
    chain = _chain()
    request = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="hi")])])

    asyncio.run(_drain(chain, request))

    assert _calls() == [PRIMARY, BACKUP_A]


def test_an_unsigned_history_never_reaches_the_primary():
    """The fix, asserted on what was attempted rather than on what came back.

    The live turn's third call went to Gemini and died there. Here the same
    conversation is offered to the same chain and Gemini is never asked, so the
    first backup serves the turn.
    """
    chain = _chain()
    request = _unsigned_history_request()

    asyncio.run(_drain(chain, request))

    assert _calls() == [BACKUP_A], "a foreign conversation was offered to the primary"


def test_the_whole_live_turn_now_completes():
    """The regression test proper: the four spans of session ``bc4de8b5``, replayed.

    Gemini answers call one with a *signed* function call, then 503s on call two
    and a backup serves it with an *unsigned* one -- and call three must not go
    back to Gemini. Driven as three separate ``LlmRequest`` objects sharing one
    growing conversation, because that is the shape the flow produces:
    ``base_llm_flow._run_one_step_async`` builds a fresh request per step.

    The three-call structure is the point, and a single-call version of this test
    would pass against the unfixed code.
    """
    chain = _chain()

    # Call 1: clean history, Gemini answers with a signed call.
    first = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="what did you learn?")])])
    asyncio.run(_drain(chain, first))
    after_first = _calls()

    # Call 2: Gemini 503s in production; here RejectingGemini's 400 is enough,
    # because both are non-retriable-to-Gemini and both land us on a backup.
    signed = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="current_datetime", args={}),
                thought_signature=b"signed",
            )
        ],
    )
    second = LlmRequest(
        contents=[
            *first.contents,
            signed,
            types.Content(role="user", parts=[types.Part(text='{"iso": "2026-10-01T23:36:53+00:00"}')]),
        ]
    )
    asyncio.run(_drain(chain, second))
    after_second = _calls()

    # Call 3: the backup's unsigned call is in the history. This is the call that
    # raised "Function call is missing a thought_signature" live.
    third = LlmRequest(
        contents=[
            *second.contents,
            types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name="read_day_digest", args={"day": "2026-09-30"}))],
            ),
            types.Content(role="user", parts=[types.Part(text='{"note_count": 17}')]),
        ]
    )
    response = asyncio.run(_drain(chain, third))
    third_call = _calls()[len(after_second) :]

    assert after_first == [PRIMARY, BACKUP_A], "call 1: a clean history skipped the primary"
    assert after_second[len(after_first) :] == [PRIMARY, BACKUP_A], "call 2: 503 moved to a backup"
    assert third_call == [BACKUP_A], "call 3: an unsigned history went back to the primary"
    assert response.model_version == BACKUP_A


def test_the_selection_is_per_call_and_not_made_once():
    """Why the earlier two tests are not redundant.

    A wrapper that chose its chain once -- at construction, or on the first call,
    or by caching the first answer's model -- would pass
    :func:`test_a_clean_conversation_still_starts_at_the_primary` and then fail
    here. Both requests go to the *same instance*, which is the only way this can
    be observed: the wrapper is shared by every concurrent turn.
    """
    chain = _chain()
    clean = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="hi")])])

    asyncio.run(_drain(chain, clean))
    asyncio.run(_drain(chain, _unsigned_history_request()))

    assert _calls() == [PRIMARY, BACKUP_A, BACKUP_A]


def test_the_response_is_the_delegate_response_not_a_rewritten_one():
    """``model_version`` must survive the delegation.

    The ``**Sources**`` block reads this field off the response to name the
    backend that actually answered, and a wrapper that rebuilt the response would
    rename every source line to itself -- a silent provenance regression in a
    function whose job is routing. Asserted on the delegate's own name rather than
    the wrapper's, which is what makes it a provenance test and not a mock test.
    """
    chain = _chain()
    request = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="hi")])])

    response = asyncio.run(_drain(chain, request))

    assert response.model_version == BACKUP_A
    assert response.model_version != chain.model


# --- the construction invariants ----------------------------------------------


def test_a_sub_chain_that_is_not_a_sub_sequence_is_rejected():
    """A misconfiguration must fail at import, not on a live turn.

    An entry outside ``models`` could never be reached in either direction, so the
    guard would silently never select it -- or, worse, a reordering would quietly
    change which model is allowed to serve a foreign conversation. Either is a
    routing change made invisible.
    """
    backups = [
        RecordingModel(model=BACKUP_A, calls=[]),
        RecordingModel(model=BACKUP_B, calls=[]),
    ]

    with pytest.raises(ValueError, match="sub-sequence"):
        HistorySafeFallbackModel(
            models=[FailingModel(model=PRIMARY, calls=[]), *backups],
            unsigned_history_models=list(reversed(backups)),
        )


def test_a_model_name_cannot_be_set_directly():
    """``model`` is derived, like ``FallbackModel``'s.

    ADK reads it to name the span and to fill ``LlmRequest.model``. A wrapper that
    named itself would appear in every trace as a model that does not exist, and
    the mismatch is invisible in the output.
    """
    backups = [RecordingModel(model=BACKUP_A, calls=[])]

    with pytest.raises(ValueError, match="cannot be set directly"):
        HistorySafeFallbackModel(
            models=[FailingModel(model=PRIMARY, calls=[]), *backups],
            unsigned_history_models=backups,
            model="something-else",
        )


def test_the_wrapper_names_itself_after_the_primary():
    """The other half of the derived field: set from ``models[0]``.

    The trace should show Gemini for a Gemini turn even though a backup served
    it -- which is what makes the mismatch between the span name and
    ``model_version`` legible instead of confusing.
    """
    chain = _chain()

    assert chain.model == PRIMARY


def test_capabilities_come_from_the_primary_not_the_sub_chain():
    """The request is built before any call, so the primary defines the shape.

    Reading them off the sub-chain would make the answer depend on whose turn it
    is -- a request built for Gemini capabilities on the first call and for
    LiteLLM's on the third, which is a request the flow never re-derives.
    """
    chain = _chain()

    assert chain.capabilities == LlmCapabilities(output_schema_and_tools=True)


# --- signature_free_chain ------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["gemini-3.5-flash-lite", "gemini:gemini-3.5-flash-lite", "gemini/gemini-3.5-flash-lite", "GEMINI-3.5-flash-lite"],
)
def test_gemini_in_any_spelling_is_excluded(name):
    """The requirement belongs to the API, not to one model name.

    ADK's registry accepts ``gemini:`` and ``gemini/`` prefixes for the same
    model, and the name is compared case-insensitively because the chain is
    written by hand. A spelling left out here would reintroduce the live defect
    the first time somebody used it.
    """
    assert signature_free_chain([name, BACKUP_A]) == [BACKUP_A]


@pytest.mark.parametrize(
    "name",
    ["openrouter/nvidia/nemotron:free", "qwen/qwen3.8-27b:free", "openrouter/google/gemma:free", "gpt-oss-120b"],
)
def test_a_non_gemini_entry_is_kept(name):
    """The direction of the default matters as much as the matching.

    An entry with no case here is treated as signature-free, so it stays eligible
    for every conversation rather than being excluded from all of them. Getting
    this backwards would make the fallback tier unreachable and silently reduce
    the chain to a single model.
    """
    assert signature_free_chain(["gemini-test-primary", name]) == [name]


def test_the_order_of_the_surviving_entries_is_preserved():
    """Failover order is a routing decision, not a cosmetic one.

    ``[gemini, gemma, qwen, nvidia]`` must become ``[gemma, qwen, nvidia]``: the
    same sequence of attempts, minus the one entry that cannot serve the
    conversation. Sorting or deduplicating here would change which model answers
    a turn where more than one is reachable.
    """
    chain = ["gemini-test-primary", BACKUP_A, BACKUP_B, "openrouter/third:free"]

    assert signature_free_chain(chain) == [BACKUP_A, BACKUP_B, "openrouter/third:free"]


def test_a_chain_of_only_gemini_models_excludes_everything():
    """The degenerate case is empty, and the constructor is what rejects it.

    Asserted as "empty" rather than as an exception because
    :func:`signature_free_chain` is a filter and does not know what it is for; the
    refusal belongs to the model, which does.
    """
    assert signature_free_chain(["gemini-a", "gemini-b"]) == []


# --- helpers -------------------------------------------------------------------


def _unsigned_history_request() -> LlmRequest:
    """The live turn's third request: one unsigned ``functionCall`` in the history."""
    return LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part(text="what did you learn?")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="current_datetime", args={}),
                        thought_signature=b"signed",
                    )
                ],
            ),
            types.Content(role="user", parts=[types.Part(text='{"date": "2026-10-01"}')]),
            types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name="read_day_digest", args={"day": "2026-09-30"}))],
            ),
            types.Content(role="user", parts=[types.Part(text='{"note_count": 17}')]),
        ]
    )


async def _drain(chain: HistorySafeFallbackModel, request: LlmRequest) -> LlmResponse:
    """Run one model call to completion and return the response.

    Drains the async generator rather than taking the first item: a chain that
    falls over yields nothing from the primary and one item from the backup, and
    a test that stopped at the first ``yield`` would hang on the primary's error.
    """
    last = None
    async for response in chain.generate_content_async(request):
        last = response
    assert last is not None, "the chain yielded nothing at all"
    return last