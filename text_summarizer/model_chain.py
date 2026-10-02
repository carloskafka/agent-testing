"""A model chain that will not hand one provider's history to another.

Found live, on session ``bc4de8b5-535f-41ce-b35b-775b297df039``. The turn was
*"What did you learn yesterday?"*, and it died 24.8s in with no answer at all:

.. code-block:: text

    call_llm  7.2s  gemini-3.5-flash-lite              -> current_datetime   (signed)
    TOOL      0.0s                                    -> 2026-10-01T23:36:53Z
    call_llm  5.0s  nvidia/nemotron-3.5-lightning:free -> read_day_digest    (UNSIGNED)
    TOOL      0.0s                                    -> 17 notes, 36 topics
    call_llm  0.5s  gemini-3.5-flash-lite              -> 400 INVALID_ARGUMENT

The 400 is the whole story:

.. code-block:: text

    Function call is missing a thought_signature in functionCall parts.
    ... function call `default_api:read_day_digest`, position 5.

**Gemini validates the entire conversation on every call**, including turns it
did not take part in, so the third call rejected the *second* one. Nemotron
reached that call because Gemini returned 503 and both ``:free`` models ahead of
it returned 429 -- that is, the fallback chain working exactly as designed.

Three properties of ADK turn that into a dead turn rather than a slow one:

1. **The chain restarts at the primary every call.**
   ``FallbackModel.generate_content_async`` is ``for index, entry in
   enumerate(self.models)`` with no memory of who answered last, so a history
   that one provider produced is always offered to the primary again first.
2. **400 is not retriable.** ``DEFAULT_STATUS_CODES`` is ``{429, 500, 502, 503,
   504}``, so ``_should_fall_back`` rejects it and the error propagates -- the
   chain never even reaches nemotron again, which would have accepted the very
   same history without complaint.
3. **There is no turn identity on ``LlmRequest`` to pin against.** This is the
   part that decided the design. The obvious fix is to remember which entry
   served the first ``call_llm`` and start the rest of the turn there -- but
   ``base_llm_flow._run_one_step_async`` builds ``LlmRequest()`` *inside* the
   step, and ``run_async`` loops steps, so **every model call of a turn gets a
   fresh request object**. A pin keyed on the request would silently never pin,
   which is the same shape as the cache's ``_first_model_call_of_invocation``
   defect this repo already hit once (AGENTS.md gotcha 12): a guard that reads
   true for the wrong reason and passes every unit test.

So the pin is not stored, it is **derived from the payload**. The question a
turn actually needs answered is "may this provider see this conversation?", and
the conversation is right there in ``llm_request.contents``. That makes the
guard stateless, which is the strongest property available: there is no turn
boundary to get wrong, so a resumed session, two concurrent turns and a live
connection are all the same code path, and there is no state to leak between
them.

The invariant, stated once:

    **A model that requires ``thought_signature`` may only be offered a
    conversation in which every ``functionCall`` part carries one.**

Read the other way: once a foreign model has spoken, Gemini is out for the rest
of the conversation. The fallback tier still fails over freely -- gemma and qwen
accept an unsigned history, measured -- so the resilience is kept and only the
one splice that Gemini rejects is prevented.

**The fallback itself is not disabled, only made safe.** The primary still fails
transiently and the chain still moves on; that is the whole point of the tier,
and the trace still names whichever backend answered. What cannot happen now is
a *second* provider being handed the first one's history. On the live chain that
was rare and total: 4 OpenRouter generations in the three days to 2026-10-01, of
which exactly one ended a turn.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from google.adk.models import (
    BaseLlm,
    FallbackModel,
    LlmCapabilities,
    LlmRequest,
    LlmResponse,
)
from pydantic import Field, PrivateAttr, model_validator


def _entry_name(entry: str | BaseLlm) -> str:
    """The model name of a chain entry.

    Mirrors ``google.adk.models._fallback_model._model_name`` rather than
    importing it: that symbol is private and an ADK rename would break the import
    for no gain. The pair of forms is ADK's, not ours -- the Gemini primary is a
    bare string resolved through the LLM registry, every fallback is an
    already-constructed model.
    """
    return entry if isinstance(entry, str) else entry.model


def history_needs_signed_calls(llm_request: LlmRequest) -> bool:
    """Whether ``llm_request`` carries a ``functionCall`` part with no signature.

    The predicate the whole module exists for, and deliberately narrow: it asks
    about *function call* parts only. A plain text part has no signature either,
    and counting those would make every conversation look foreign and hand every
    turn to the fallback tier.

    An empty signature counts as absent. ``b""`` is what an upstream that
    supports the field but has nothing to put in it produces, and Gemini is not
    satisfied by it.

    Args:
        llm_request: The request about to be sent. Its ``contents`` are the
            whole conversation, so no separate history needs assembling.

    Returns:
        True if any ``functionCall`` part lacks a ``thought_signature``.
    """
    for content in llm_request.contents:
        for part in content.parts or ():
            if getattr(part, "function_call", None) is None:
                continue
            if not getattr(part, "thought_signature", None):
                return True
    return False


class HistorySafeFallbackModel(BaseLlm):
    """A ``FallbackModel`` chain that never re-offers a foreign history.

    Wraps two chains rather than reimplementing failover, because
    ``FallbackModel``'s loop is worth keeping: it snapshots and rolls back the
    request between attempts (``_RequestSnapshot``), restores ``llm_request.model``
    per delegate, and refuses to fail over mid-stream once a response has been
    yielded. Re-deriving that is a copy that will drift.

    The two chains share their delegate instances, so this costs no second client
    and no second Gemini resolution. Selection happens per call, from the
    conversation, with nothing stored between calls.

    Attributes:
        models: The full chain, primary first, exactly as it would have been
            handed to ``FallbackModel``.
        unsigned_history_models: The sub-chain allowed once the conversation
            carries an unsigned ``functionCall`` -- in practice the chain with
            every signature-requiring entry removed. Must be a sub-sequence of
            :attr:`models`, which is checked at construction rather than
            discovered on a live turn.
    """

    #: Derived from :attr:`models` in :meth:`_check_shape`, so it defaults to the
    #: empty string rather than being required. ``FallbackModel`` declares the same
    #: field the same way, for the same reason: ``BaseLlm`` requires it, and
    #: deriving it means the caller cannot set it inconsistently.
    model: str = ""

    models: list[str | BaseLlm] = Field(min_length=1)
    """The full chain. Mirrors ``FallbackModel.models`` so the shape is readable."""

    unsigned_history_models: list[str | BaseLlm] = Field(min_length=1)
    """The chain allowed once :func:`history_needs_signed_calls` is true."""

    _full: FallbackModel | None = PrivateAttr(default=None)
    _unsigned: FallbackModel | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_shape(self) -> HistorySafeFallbackModel:
        """Derive :attr:`model` and reject a sub-chain that is not a sub-chain.

        ``BaseLlm`` declares ``model`` and ADK reads it everywhere -- to name the
        span, to fill ``LlmRequest.model`` -- so it cannot be left unset. Taking
        it from the primary is what ``FallbackModel`` does with the same field,
        and a hand-built wrapper that named itself something else would show up
        in every trace as a model that does not exist.

        The sub-sequence check is here because the alternative is a silent
        misconfiguration: an ``unsigned_history_models`` naming a model outside
        the chain would simply never be selected, or worse, would quietly widen
        what the fallback tier is allowed to serve.
        """
        if not self.model:
            self.model = _entry_name(self.models[0])
        elif self.model != _entry_name(self.models[0]):
            raise ValueError(
                "HistorySafeFallbackModel.model is derived from the first entry"
                f" of `models` and cannot be set directly: got {self.model!r},"
                f" expected {_entry_name(self.models[0])!r}."
            )

        full = [_entry_name(entry) for entry in self.models]
        allowed = [_entry_name(entry) for entry in self.unsigned_history_models]
        if len(allowed) > len(full) or not any(
            full[start : start + len(allowed)] == allowed
            for start in range(len(full) - len(allowed) + 1)
        ):
            raise ValueError(
                "`unsigned_history_models` must be a sub-sequence of `models`,"
                f" in the same order: {allowed} is not a contiguous run of {full}."
                " The two chains share delegates, so an entry that is not in"
                " `models` could never be reached in either direction."
            )
        return self

    def _chain(self, llm_request: LlmRequest) -> FallbackModel:
        """The chain this conversation may be served by.

        Built on first use and cached, because ``FallbackModel`` resolves its own
        delegates lazily and rebuilding would drop that cache. The two entries are
        separate objects even when the sub-chain is the whole chain (which is the
        case for a chain with no signature-requiring model in it), so the choice
        is made once per call rather than being a property of the instance.
        """
        if history_needs_signed_calls(llm_request):
            if self._unsigned is None:
                self._unsigned = FallbackModel(models=list(self.unsigned_history_models))
            return self._unsigned
        if self._full is None:
            self._full = FallbackModel(models=list(self.models))
        return self._full

    @property
    def capabilities(self) -> LlmCapabilities:
        """The full chain's capabilities.

        The request is built before any call is made, so what it was built for is
        the *primary's* capabilities whatever ends up serving it -- the same
        reasoning ``FallbackModel.capabilities`` gives. Read off the full chain
        rather than the sub-chain, since a sub-chain that starts at the fallback
        tier would otherwise change the answer depending on whose turn it is.
        """
        if self._full is None:
            self._full = FallbackModel(models=list(self.models))
        return self._full.capabilities

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Yields the selected chain's response, unmodified.

        Delegates rather than wraps, so the response object is the delegate's own
        and ``model_version`` still names whichever backend actually answered.
        That field is what the ``**Sources**`` block reads (see ``sources.py``),
        so re-wrapping the response here would rename every source line to this
        wrapper -- a silent regression in the provenance feature, in a function
        whose name suggests it only routes.

        The ``llm_request`` handed down is the wrapper's own, not a copy.
        ``FallbackModel`` writes to it between attempts (``llm_request.model =
        delegate.model``) and restores the parts a failed attempt edited, and the
        flow expects to see those writes on the object it passed in.
        """
        async for response in self._chain(llm_request).generate_content_async(
            llm_request, stream
        ):
            yield response

    @asynccontextmanager
    async def connect(self, llm_request: LlmRequest) -> AsyncIterator[Any]:
        """Opens a live connection through the same selection.

        The connection type is ADK's private ``BaseLlmConnection``, so it is left
        as ``Any`` rather than imported from a private module.

        Unused in this deployment -- ``adk web`` serves ordinary turns and the
        dev UI posts ``streaming: false`` -- but a live connection carries a
        conversation too, and leaving ``BaseLlm.connect`` to raise would make
        this wrapper a worse model than the chain it wraps for any deployment
        that does use one.
        """
        async with self._chain(llm_request).connect(llm_request) as connection:
            yield connection


def signature_free_chain(entries: list[str | BaseLlm]) -> list[str | BaseLlm]:
    """``entries`` with every Gemini entry removed, order preserved.

    Exists so the caller states the exclusion rather than assembling it, and so
    the set of models that *require* ``thought_signature`` is one named list
    instead of a condition scattered across two branches of ``get_model``.

    A name is judged by its provider prefix, not by a hard-coded model list,
    because the requirement belongs to the API rather than to a model name that
    can be retired. ``gemini:`` and ``gemini/`` are both spellings ADK's registry
    accepts for the same model, so both are matched.

    Args:
        entries: The full chain.

    Returns:
        The same entries without the signature-requiring ones. May be empty when
        every entry requires signatures, which the constructor rejects -- a
        conversation with an unsigned call has to go somewhere, and failing to
        build the model is the honest response to there being nowhere.
    """
    return [entry for entry in entries if not _requires_thought_signature(_entry_name(entry))]


def _requires_thought_signature(model_name: str) -> bool:
    """Whether a model rejects a ``functionCall`` part carrying no signature.

    The Gemini API is the only one measured to, and this is the one place that
    assumption is written down.

    Only names resolving to the **native** Gemini API count. ADK's registry
    accepts a ``gemini:`` or ``gemini/`` provider prefix for the same model, so
    the prefix is stripped before the check -- otherwise the guard would miss the
    two spellings and reintroduce the live defect the first time somebody used
    one.

    A model reached *through another provider* does not count, even when its name
    says Gemini: ``openrouter/google/gemini-...`` is LiteLLM's request shape, and
    it is not the API that raised the 400. Judging by the ADK prefix rather than
    by the word "gemini" is what keeps that distinction, and it puts an unrecognised
    name on the signature-free side -- the safe direction, since such an entry
    stays eligible for every conversation instead of being excluded from all of
    them.
    """
    lowered = model_name.lower()
    for prefix in ("gemini:", "gemini/"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    return lowered == "gemini" or lowered.startswith("gemini-")


__all__ = [
    "HistorySafeFallbackModel",
    "history_needs_signed_calls",
    "signature_free_chain",
]