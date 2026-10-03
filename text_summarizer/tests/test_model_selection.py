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

import pathlib
import re

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
EXPECTED_ALIASES = ("lfm", "ling", "dots", "gemma", "qwen", "nvidia")


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

    The order is measured, not alphabetical. Benchmarked over the 17 free models
    OpenRouter lists on 2026-10-02, every entry confirmed to emit tool calls:

        lfm-2.5-2.6b  median 0.45s   <- first, and the default alias
        ling-3.0      median 1.17s
        dots-3        median 2.46s
        gemma / qwen  no figure: 0/50 for the whole window
        nemotron      median 18.9s, p90 87.8s, 429 on half its calls

    So the previous incumbent is now *last*, and the two models with no
    measurement at all sit in the middle rather than leading -- unknown is a
    different risk from slow, and neither should displace a known-good tier.
    """
    assert tuple(agent_module.OPENROUTER_MODELS) == EXPECTED_ALIASES


def test_the_default_alias_is_the_measured_fastest():
    """The *unset* default must be the fastest model, and ``.env.example`` must agree.

    Two separate files set the value independently, which is exactly the shape of a
    drift bug: the table could lead with ``lfm`` while a shipped ``.env`` still
    said ``gemma``, and every deployment would quietly run the 429'd model. So
    both are pinned here.

    ``conftest.py`` blanks ``MODEL_ALIAS``, so at import ``get_model`` takes the
    ``_free_openrouter_models()[0]`` branch -- which is precisely the deployment
    shape this asserts about.
    """
    assert agent_module.MODEL_ALIAS == "", (
        "conftest blanks MODEL_ALIAS; if that changes, this test is asserting the "
        "wrong thing and the unset branch is no longer being exercised"
    )
    fastest = agent_module._free_openrouter_models()[0]
    assert fastest.startswith("liquid/lfm-2.5-2.6b"), (
        f"with MODEL_ALIAS unset the chain starts at {fastest!r}, not the measured fastest"
    )

    example = pathlib.Path(__file__).resolve().parents[2] / ".env.example"
    configured = re.search(
        r"^MODEL_ALIAS=(\S*)", example.read_text(encoding="utf-8"), re.M
    )
    assert configured and configured.group(1) == "lfm", (
        "MODEL_ALIAS in .env.example does not name the fastest measured model"
    )


def test_nemotron_is_no_longer_the_default_free_model():
    """The specific regression this change fixes, kept as a test.

    Nemotron was measured at a median of 18.9s with a p90 of 87.8s, and it 429'd
    on half the calls that were attempted. It stays in the table because
    ``MODEL_ALIAS=nvidia`` is a documented setting, but a fresh deployment must
    not be routed to it first.
    """
    free = agent_module._free_openrouter_models()
    assert not free[0].startswith("nvidia/")
    assert free[-1].startswith("nvidia/"), (
        "the slowest measured model should be the last OpenRouter entry, not removed "
        "or promoted"
    )


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


def test_openrouter_leads_when_no_zen_key_and_gemini_is_last(monkeypatch):
    """The chain is ``[*free, gemini]`` -- Gemini last, not first.

    **This inverts the order this branch had since the Zen tier existed**, on
    2026-10-03, and the previous order is recorded here because it was a
    deliberate decision that has now been deliberately reversed:

    > "putting a free OpenRouter model first would hand every ordinary turn to a
    > tier whose reliability is not under our control (AGENTS.md known gap 7)"

    That reasoning still holds and is the reason Gemini is now the *last* entry
    rather than being dropped. OpenCode's Zen model leads instead, which is a
    configured preference rather than a free-tier default; with no Zen key the
    free OpenRouter models take that position.

    Asserted against a *configured* key, because the chain legitimately differs
    when none is set: with no OpenRouter key and no Zen key there is nowhere for an
    unsigned conversation to go, so a signature-free placeholder is appended rather
    than leaving the chain unbuildable. That path has its own test below.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 40)
    monkeypatch.setenv("OPENCODE_API_KEY", "")

    names = _chain_names(agent_module.get_model())

    assert names[-1] == agent_module.GEMINI_MODEL, "Gemini is the floor, not the default"
    assert names[:-1] == [
        OPENROUTER_PREFIX + n for n in agent_module._free_openrouter_models()
    ]


def test_zen_leads_when_it_is_configured(monkeypatch):
    """The order asked for on 2026-10-03: OpenCode, then OpenRouter, then Gemini.

    The reordering is not cosmetic. A ``FallbackModel`` tries entries in sequence,
    so the first entry serves every ordinary turn and the rest exist only for when
    it fails. Measured before the change on session
    ``068f3e5c-3ef2-458e-8926-bbf491312252``: Gemini served all nine generations
    because it was first and it did not fail -- which is the whole reason nothing
    else was ever reached.

    The cost is stated rather than discovered later: every turn now goes to a
    *free* endpoint, which is known gap 7 -- `gemma:free` and its neighbours 429
    often enough that a quota error was as likely to end in a rate-limit error as
    in a served turn. Gemini last is what keeps that from being fatal.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 40)
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")

    names = _chain_names(agent_module.get_model())

    assert names[0] == f"hosted_vllm/{agent_module.OPENCODE_MODEL}"
    assert names[1:-1] == [
        OPENROUTER_PREFIX + n for n in agent_module._free_openrouter_models()
    ]
    assert names[-1] == agent_module.GEMINI_MODEL


def test_the_gemini_entry_is_a_name_not_a_litellm(monkeypatch):
    """ADK resolves a bare string through its own registry, which is the cheap path.

    Wrapping it in ``LiteLlm`` instead would be a behavioural change with no
    visible symptom: a model name that has to be looked up in a registry
    configured with the Gemini key, versus one already bound to a LiteLLM client.
    The two entry kinds in the same chain are the shape, so each is checked where
    it belongs.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("OPENCODE_API_KEY", "")

    entries = agent_module.get_model().models

    # Gemini is the last entry since the 2026-10-03 reorder, so "first" here would
    # be wrong -- and an assertion left pointing at index 0 would silently start
    # checking a LiteLlm entry and pass for the wrong reason.
    assert isinstance(entries[-1], str)
    assert all(isinstance(entry, LiteLlm) for entry in entries[:-1])


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


# --- multiple accounts, and the Zen tier --------------------------------------
#
# Found on 2026-10-02, session `b6f09fca`: Gemini served eight calls of a turn, the
# ninth 429'd, the chain fell through to OpenRouter, and that account's free tier
# was already spent -- so the turn wrote its notes and **returned no answer at
# all**. The quota was the trigger; the missing second account is the defect.


def test_several_keys_give_several_times_the_budget(monkeypatch):
    """Two accounts means two independent 50/day pools, and the chain has both.

    The product is deliberately **per key, not per model**: an account's daily
    budget is shared across all of that account's models, so `key x model` pairs
    would spend the same 50 several times and look like headroom.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("OPENCODE_API_KEY", "")
    monkeypatch.setenv(
        "OPENROUTER_API_KEY", f"sk-or-v1-{'a' * 40};sk-or-v1-{'b' * 40}"
    )

    names = _chain_names(agent_module.get_model())

    free = list(agent_module._free_openrouter_models())
    assert names[-1] == agent_module.GEMINI_MODEL
    # Every model reachable from the first key, then every model from the second.
    assert names[:-1] == [OPENROUTER_PREFIX + n for n in free] * 2


def test_the_zen_tier_is_added_only_when_a_key_is_configured(monkeypatch):
    """Unconfigured, the chain is byte-for-byte what it was before Zen existed."""
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 40)
    monkeypatch.setenv("OPENCODE_API_KEY", "")

    assert agent_module.opencode_enabled() is False
    assert not any("space-bunny" in n for n in _chain_names(agent_module.get_model()))

    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")
    assert agent_module.opencode_enabled() is True
    names = _chain_names(agent_module.get_model())
    # First, since the 2026-10-03 reorder -- it serves every ordinary turn.
    assert names[0] == f"hosted_vllm/{agent_module.OPENCODE_MODEL}"


def test_zen_is_reachable_when_it_is_the_only_tier(monkeypatch):
    """No OpenRouter key is a supported configuration, not a broken one.

    Worth pinning because the alternative is a crash: a Gemini-only chain leaves
    nowhere for an *unsigned* conversation to go, and `HistorySafeFallbackModel`
    rejects an empty signature-free list at construction -- so the agent would
    fail to import rather than merely losing its fallback.
    """
    monkeypatch.setattr(agent_module, "MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")

    names = _chain_names(agent_module.get_model())
    # Zen first, Gemini last. Both entries are still required: dropping Gemini
    # would leave the chain ending on a signature-free model for a conversation
    # that *does* carry signatures, which is the history guard's whole subject.
    assert names == [
        f"hosted_vllm/{agent_module.OPENCODE_MODEL}",
        agent_module.GEMINI_MODEL,
    ]


# --- ';'-separated keys --------------------------------------------------------


def test_keys_split_on_a_semicolon_and_are_trimmed():
    """The separator is a list, so the number of accounts is not capped by a name."""
    keys = agent_module._split_keys(" sk-or-v1-aaa ; sk-or-v1-bbb ")
    assert keys == ["sk-or-v1-aaa", "sk-or-v1-bbb"]


def test_an_empty_entry_is_not_a_key():
    """`a;;b` and a trailing `;` are typing, not configuration."""
    assert agent_module._split_keys("sk-or-v1-aaa;;sk-or-v1-bbb;") == [
        "sk-or-v1-aaa",
        "sk-or-v1-bbb",
    ]
    assert agent_module._split_keys("") == []
    assert agent_module._split_keys(None) == []


def test_the_same_key_twice_is_one_key():
    """Otherwise it doubles a chain entry without adding a single request."""
    assert agent_module._split_keys("sk-or-v1-aaa;sk-or-v1-aaa") == ["sk-or-v1-aaa"]


def test_a_malformed_entry_is_dropped_and_reported(capsys):
    """Loudly, because the alternative is a 401 much later with no clue why.

    Only the *shape* of the rejected value is printed. It is a credential, and a
    log line is exactly where credentials end up.
    """
    keys = agent_module._split_keys(
        "sk-or-v1-" + "a" * 40 + ";garbage-not-a-key", pattern=agent_module._OPENROUTER_KEY_RE
    )
    assert keys == ["sk-or-v1-" + "a" * 40]

    err = capsys.readouterr().err
    assert "malformed" in err
    # The rejected value itself must not appear.
    assert "garbage-not-a-key" not in err
    assert "garbag..." in err


def test_the_list_length_is_bounded():
    """A paste accident must not become an unbounded chain of doomed calls."""
    many = ";".join(f"sk-or-v1-{i:0>40}" for i in range(50))
    assert len(agent_module._split_keys(many)) == agent_module.MAX_PROVIDER_KEYS


def test_a_real_key_survives_validation():
    """The pattern must accept the shape OpenRouter actually issues.

    Built from the documented shape rather than copied from a live key. An
    earlier version of this test pasted a real key in as ``real_shape``, which is
    a secret in the repository and in every clone of it -- and GitHub's push
    protection blocked the push for exactly that, which is the only reason this
    was caught before it went anywhere. The length and alphabet are what the
    regex actually tests, so constructing them tests the same thing.
    """
    real_shape = "sk-or-v1-" + "9671d7e5" * 8
    assert len(real_shape) == len("sk-or-v1-") + 64
    assert agent_module._OPENROUTER_KEY_RE.match(real_shape)
    for wrong in ("", "sk-or-v1-", "sk-or-v1-short", "your_openrouter_api_key_here"):
        assert not agent_module._OPENROUTER_KEY_RE.match(wrong), wrong


# --- the Zen timeout ----------------------------------------------------------
#
# The number that started this: a first sample of 10 calls had a median of 4.0s,
# and "median 4.0s" read aloud next to a timeout discussion sounds like a setting.
# It was not one -- nothing was configured at all.


def test_the_zen_timeout_clears_the_measured_tail():
    """A timeout below the slowest successful call manufactures false errors.

    Measured 40 calls, all HTTP 200: 37 land under 5s, then 17.83s and 46.46s.
    A timeout set from the *median* (4.0s on the first, smaller sample) would fail
    10% of calls that succeeded, and each of those sends the chain hunting for a
    fallback it did not need. So the constant has to clear the tail.
    """
    measured = [0.92, 0.97, 0.97, 1.01, 1.07, 1.1, 1.12, 1.12, 1.12, 1.13,
                1.13, 1.21, 1.22, 1.22, 1.25, 1.27, 1.3, 1.3, 1.33, 1.35,
                1.5, 1.56, 1.57, 1.65, 1.75, 1.91, 1.96, 1.99, 2.05, 2.06,
                2.14, 2.26, 2.49, 2.78, 3.26, 3.7, 4.62, 4.98, 17.83, 46.46]
    assert max(measured) < agent_module.OPENCODE_TIMEOUT_S

    # And the value that would have looked reasonable, asserted to be wrong.
    failures_at_median = len([x for x in measured if x > 4.0])
    assert failures_at_median == 4, "the sample changed; re-measure before trusting this"


def test_the_zen_client_carries_the_timeout_and_credentials():
    """The constant is only load-bearing if it reaches the completion call.

    Asserted against ``_additional_args`` rather than a ``timeout`` attribute
    because `LiteLlm` keeps no such attribute: its ``__init__`` is
    ``(self, model, **kwargs)`` and everything else is stashed in
    ``_additional_args`` to be merged into the litellm call. Reading
    ``llm.timeout`` raises `AttributeError`, which is how this test first failed
    -- and an assertion that had *silently* passed against a plain attribute would
    have proved nothing about the value actually reaching the wire.
    """
    llm = agent_module._opencode_llm()
    assert isinstance(llm, LiteLlm)
    args = llm._additional_args
    assert args["timeout"] == agent_module.OPENCODE_TIMEOUT_S
    assert args["api_base"] == agent_module.OPENCODE_API_BASE


def test_an_openrouter_entry_binds_its_own_key():
    """Two chain entries must not resolve to the same account.

    This is the whole point of threading ``api_key`` through: left to litellm
    reading ``OPENROUTER_API_KEY`` from the environment, *both* entries would use
    key one, so the second account's 50/day would never be drawn on and the chain
    would spend the same exhausted budget twice.
    """
    keys = [f"sk-or-v1-{c * 40}" for c in ("a", "b")]
    first = agent_module._openrouter_llm("qwen/qwen3.8-27b:free", keys[0])
    second = agent_module._openrouter_llm("qwen/qwen3.8-27b:free", keys[1])
    assert first._additional_args["api_key"] == keys[0]
    assert second._additional_args["api_key"] == keys[1]


def test_the_zen_key_is_bound_when_one_is_configured(monkeypatch):
    """Separate from the timeout because ``api_key=None`` is a valid state here.

    ``conftest.py`` blanks ``OPENCODE_API_KEY`` for every test, so the unconfigured
    shape is ``None``. Asserting only that would pass whether the key were wired up
    or not -- which is the failure this whole change is about.
    """
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test_key")
    llm = agent_module._opencode_llm()
    assert llm._additional_args["api_key"] == "oc_sk_test_key"

    monkeypatch.setenv("OPENCODE_API_KEY", "")
    assert agent_module._opencode_llm()._additional_args["api_key"] is None
