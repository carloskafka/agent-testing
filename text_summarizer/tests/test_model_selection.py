"""Which backend the agent can be served by, and in what order it will try them.

``get_model()`` has no tests, and it is the reason the whole ``**Sources**``
provenance feature exists. That feature works by reading
``LlmResponse.model_version`` off the response -- and the value there is
"whichever backend in the chain actually answered", which is unknowable when the
chain is *built*. So the chain is assembled from three independent inputs
(``MODEL_PROVIDER``, ``MODEL_ALIAS``, the hard-coded ``OPENROUTER_MODELS`` table),
and a mistake in any of them does not fail: the agent still runs, it just answers
from somewhere nobody expected, and the rendered source line names that somewhere.

That is the failure mode these tests are shaped around. Nothing here makes a
network call -- ``LiteLlm`` is a configuration object and ``FallbackModel`` a
container -- so the assertions are entirely about the *chain shape*: which entry
is primary, which are reachable as fallbacks, and that a model is never both.

**The import-time read is itself part of the contract.** ``MODEL_PROVIDER`` and
``MODEL_ALIAS`` are resolved into module globals at import, so
``monkeypatch.setenv`` after the fact changes nothing. ``test_the_env_var_alone_
does_not_move_the_chain`` pins that, because the obvious way to write these tests
-- set the env var, call the function -- would otherwise pass by accident on the
one machine whose exported variable happened to match.

The free-tier invariant (every name ends in ``:free``) is pinned as a test rather
than left to a comment because it is the *only* thing making the fallback tier
free, and it is enforced by a list comprehension in one function that only ever
runs for the values it is given.
"""

from __future__ import annotations

import pytest
from google.adk.models.lite_llm import LiteLlm
from text_summarizer import agent as agent_module
from text_summarizer.model_chain import HistorySafeFallbackModel

#: The prefix that tells LiteLLM which provider to route to. Without it a free
#: OpenRouter model name is looked up against the default provider, and the
#: fallback tier becomes a second, differently-broken attempt at the primary.
OPENROUTER_PREFIX = "openrouter/"

#: The eight filter names are pinned in ``test_obsidian_toolset.py``; this is the
#: alias table they are unrelated to, spelled out here so a reordering is visible.
EXPECTED_ALIASES = ("gemma", "qwen", "nvidia")


def _chain_names(model) -> list[str]:
    """The model name of each entry in a ``FallbackModel``, in order.

    Entries are heterogeneous by design: the Gemini primary is a bare ``str``
    (ADK resolves a string through the LLM registry) while every fallback is an
    already-constructed ``LiteLlm``. ADK's own dispatch normalises that pair in
    ``google.adk.models._fallback_model._model_name``; this mirrors it so the
    assertions read as the model names a user would recognise, without the test
    importing a private symbol and breaking on an ADK rename.
    """
    return [entry if isinstance(entry, str) else entry.model for entry in model.models]


# --- the table's own invariant -------------------------------------------------


def test_every_alias_resolves_to_a_free_model():
    """The invariant the whole fallback tier rests on.

    ``_free_openrouter_models`` filters on ``:free`` and the rest of the chain
    assumes what survives that filter is what you want. One paid name typed into
    this table would be *silently dropped* rather than used -- the agent would
    quietly have one fewer fallback and nobody would see a cost. A name with a
    ``:free`` suffix that is not actually free is the same problem inverted, and
    is not something this test can see either; the suffix is the contract.
    """
    assert agent_module.OPENROUTER_MODELS, "the table must not be emptied"
    for alias, name in agent_module.OPENROUTER_MODELS.items():
        assert name.endswith(":free"), f"{alias} -> {name} is not a free-tier model"


def test_the_aliases_are_the_three_documented_families():
    """Pinned as a tuple because the *order* is the tie-break order.

    With no ``MODEL_ALIAS`` set, ``_free_openrouter_models()[0]`` becomes the
    primary backend, so dict order is a routing decision rather than a cosmetic
    one. A new alias inserted at the front silently changes which model answers
    an ``openrouter`` run.
    """
    assert tuple(agent_module.OPENROUTER_MODELS) == EXPECTED_ALIASES


def test_no_two_aliases_name_the_same_model():
    """A duplicate would put the same model in the chain twice.

    ``get_model`` removes the primary from the fallback list by name, so a
    duplicated entry is filtered out rather than retried -- harmless for
    correctness, but it means the chain is shorter than the table looks and the
    first alias and the second are indistinguishable in the trace.
    """
    names = list(agent_module.OPENROUTER_MODELS.values())
    assert len(set(names)) == len(names), "two aliases point at the same model"


def test_the_free_list_is_the_whole_table():
    """Which links the invariant above to the chain below.

    Today these are equal because every entry is free. They stop being equal the
    moment a paid model is added, and that is the moment a reader needs to be
    able to see from this test that the paid one is not going to be used as a
    fallback -- rather than having to reason about the comprehension inside
    ``_free_openrouter_models``.
    """
    assert agent_module._free_openrouter_models() == list(agent_module.OPENROUTER_MODELS.values())


# --- the gemini path (the default) --------------------------------------------


def test_the_default_provider_was_pinned_at_import(monkeypatch):
    """``conftest.py`` sets ``MODEL_PROVIDER=gemini`` before the first import.

    Asserted rather than assumed because the *next* three tests are only
    meaningful against that default: an export left in a developer's shell would
    otherwise run them against the openrouter path and the suite would report
    green for the wrong code.
    """
    assert agent_module.MODEL_PROVIDER == "gemini"


def test_gemini_leads_and_the_free_models_follow(monkeypatch):
    """The chain is ``[gemini, *free]``, in that order.

    Order is the whole behaviour: a ``FallbackModel`` tries each entry in
    sequence, so putting a free OpenRouter model first would hand every ordinary
    turn to a tier whose reliability is not under our control (AGENTS.md known gap
    7) and use the primary only when that fails.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")

    names = _chain_names(agent_module.get_model())

    assert names[0] == agent_module.GEMINI_MODEL
    assert names[1:] == [OPENROUTER_PREFIX + n for n in agent_module._free_openrouter_models()]


def test_the_gemini_entry_is_a_name_not_a_litellm(monkeypatch):
    """ADK resolves a bare string through its own registry, which is the cheap path.

    Wrapping it in ``LiteLlm`` instead would be a behavioural change with no
    visible symptom: a model name that has to be looked up in a registry
    configured with the Gemini key, versus one already bound to a LiteLLM client.
    The two entry kinds in the same chain are the shape, so each is checked where
    it belongs.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")

    entries = agent_module.get_model().models

    assert isinstance(entries[0], str)
    assert all(isinstance(entry, LiteLlm) for entry in entries[1:])


def test_the_fallback_list_never_repeats_the_primary(monkeypatch):
    """A retried request is a wasted request, and a duplicate reads as two tries.

    The dedup is by name, so a fallback list that happened to contain the
    primary string would not be caught by a length check.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")
    names = _chain_names(agent_module.get_model())

    assert names.count(agent_module.GEMINI_MODEL) == 1
    assert len(set(names)) == len(names)


# --- the openrouter path -------------------------------------------------------


def test_the_aliased_model_leads_under_openrouter(monkeypatch):
    """``MODEL_ALIAS`` is the whole point of choosing the provider explicitly.

    A person who sets ``MODEL_PROVIDER=openrouter`` is not asking for "some free
    model", they are asking for the one they named; putting Gemini first would
    answer from a provider they deliberately left.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "openrouter")
    monkeypatch.setattr(agent_module, "MODEL_ALIAS", "nvidia")

    names = _chain_names(agent_module.get_model())

    assert names[0] == OPENROUTER_PREFIX + agent_module.OPENROUTER_MODELS["nvidia"]


def test_the_other_free_models_remain_reachable_as_fallbacks(monkeypatch):
    """Choosing an alias narrows the primary, not the chain.

    The free tier is the unreliable part (AGENTS.md gap 7), so the chain has to
    keep the other two behind the chosen one or ``MODEL_ALIAS`` would trade a
    routing decision for a single point of failure.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "openrouter")
    monkeypatch.setattr(agent_module, "MODEL_ALIAS", "gemma")

    names = _chain_names(agent_module.get_model())
    chosen = agent_module.OPENROUTER_MODELS["gemma"]

    assert names[0].endswith(chosen)
    assert sorted(n.removeprefix(OPENROUTER_PREFIX) for n in names) == sorted(
        agent_module.OPENROUTER_MODELS.values()
    )
    assert chosen not in names[1:], "the primary must not also be a fallback"


@pytest.mark.parametrize("alias", ["", "  ", "llama", "GEMMA", "gemma-4"])
def test_an_absent_or_unknown_alias_falls_back_to_the_first_free_model(monkeypatch, alias):
    """An unknown alias must not raise, and must not be a silent typo detector.

    ``OPENROUTER_MODELS.get(alias) or _free_openrouter_models()[0]`` treats every
    one of these the same way. The four spellings are here because they fail for
    different reasons -- empty, whitespace, wrong name, right name in the wrong
    case -- and a change that made, say, the case-sensitive one raise would
    otherwise be invisible.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "openrouter")
    monkeypatch.setattr(agent_module, "MODEL_ALIAS", alias)

    names = _chain_names(agent_module.get_model())

    assert names[0] == OPENROUTER_PREFIX + agent_module._free_openrouter_models()[0]


def test_no_entry_under_openrouter_is_a_bare_gemini_name(monkeypatch):
    """Every entry is routed through the OpenRouter prefix.

    This is the one place a *silent* misconfiguration is possible: a name like
    ``qwen/qwen3.8-27b:free`` without the prefix is still a valid-looking string,
    and LiteLLM would resolve it against whichever provider it defaults to.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "openrouter")
    monkeypatch.setattr(agent_module, "MODEL_ALIAS", "")

    names = _chain_names(agent_module.get_model())

    assert all(name.startswith(OPENROUTER_PREFIX) for name in names)
    assert agent_module.GEMINI_MODEL not in names, (
        "the default provider's model has no place in an explicit-openrouter chain"
    )


# --- what is built vs what is read --------------------------------------------


def test_both_providers_return_a_fallback_model(monkeypatch):
    """Never a bare model, on either path.

    This is the load-bearing assertion for provenance. The rendered source line
    reads ``LlmResponse.model_version``, and a single-model agent reports the
    name the *caller asked for*, not the name that served the call. The chain is
    what makes "which backend answered" a question with more than one answer --
    and therefore the reason the substitution has to happen in code.

    ``HistorySafeFallbackModel`` rather than ``FallbackModel`` since it replaced
    it as what ``get_model()`` returns. The chain is still a chain: the wrapper
    holds the same entries and delegates the failover to a real ``FallbackModel``
    per conversation, so "which backend answered" is unchanged.
    """
    for provider in ("gemini", "openrouter"):
        monkeypatch.setattr(agent_module, "MODEL_PROVIDER", provider)
        model = agent_module.get_model()
        assert isinstance(model, HistorySafeFallbackModel), provider
        assert len(model.models) >= 2, f"{provider} chain has no fallback at all"


def test_the_agent_runs_on_a_chain_not_on_a_single_model():
    """The wiring, not the builder.

    ``agent.py`` calls ``get_model()`` once at import and hands the result to
    ``root_agent``. If that were ever changed to a single backend, every chain
    test above would still pass while the deployed agent had lost its fallback
    and rendered a single name for every answer.
    """
    assert isinstance(agent_module.root_agent.model, HistorySafeFallbackModel)


# --- the history restriction ---------------------------------------------------


def test_gemini_is_excluded_from_the_unsigned_history_chain(monkeypatch):
    """The exclusion the wrapper exists for, asserted on the built chain.

    Under the default provider the chain leads with Gemini, so this is the case
    that would go wrong: a Gemini ``functionCall`` reached from a non-Gemini turn
    is rejected with a 400 that is not in ``FallbackModel``'s retriable set, and
    the turn dies instead of falling back (see ``model_chain``'s module
    docstring). Checking it here rather than only in ``test_model_chain`` keeps
    it next to the chain-shape assertions it is a property of.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")
    chain = agent_module.get_model()
    allowed = [m if isinstance(m, str) else m.model for m in chain.unsigned_history_models]

    assert agent_module.GEMINI_MODEL not in allowed
    assert allowed, "an empty sub-chain has nowhere to send a foreign conversation"


def test_the_openrouter_path_excludes_nothing(monkeypatch):
    """No Gemini in the chain, so nothing to exclude -- and the wrapper still does it.

    Vacuous by construction rather than by special case, which is the point: the
    same wrapper is returned on both paths, so a Gemini entry added to the
    openrouter chain later would be filtered without a second code path to
    remember to update.

    ``monkeypatch.setattr`` rather than relying on the imported default, because
    the assertion is about the *openrouter* chain and the two tests above it are
    about the gemini one. Asserting the wrong branch's chain would pass for the
    wrong reason.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "openrouter")
    chain = agent_module.get_model()

    def names(entries):
        return [m if isinstance(m, str) else m.model for m in entries]

    assert names(chain.unsigned_history_models) == names(chain.models)


def test_the_env_var_alone_does_not_move_the_chain(monkeypatch):
    """``MODEL_ALIAS`` is read at import; setting it afterwards is a no-op.

    Pinned because this is the trap in writing any of the tests above. Reading
    the env var inside ``get_model`` instead of at import would be a harmless
    refactor on its own, and this asserts the current timing on purpose: the two
    other places that read a module global resolved at import
    (``CACHE_ENABLED``, ``second_brain.VAULT_ROOT``) have each been a live bug,
    and the reason to know which way this one goes is that a test relying on
    ``setenv`` would be testing nothing.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "openrouter")
    before = _chain_names(agent_module.get_model())

    monkeypatch.setenv("MODEL_ALIAS", "nvidia")
    assert _chain_names(agent_module.get_model()) == before

    monkeypatch.setattr(agent_module, "MODEL_ALIAS", "nvidia")
    assert _chain_names(agent_module.get_model())[0].endswith(
        agent_module.OPENROUTER_MODELS["nvidia"]
    )


def test_the_openrouter_wrapper_is_where_the_prefix_is_added(monkeypatch):
    """Tested at the factory, because the prefix is the factory's only job.

    ``_openrouter_llm`` is the single point where a bare model name becomes a
    provider-routed one, and it is used by four call sites in the two branches
    above. Asserting it here means a future provider gets the prefix the same way
    the existing ones do, instead of by a reader copying the nearest example.
    """
    llm = agent_module._openrouter_llm("some/model:free")

    assert isinstance(llm, LiteLlm)
    assert llm.model == "openrouter/some/model:free"
