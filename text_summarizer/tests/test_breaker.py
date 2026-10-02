"""The rate-limit breaker: which accounts are suppressed, and for how long.

**The failure it prevents.** Session ``b6f09fca``: Gemini served eight calls of a
turn, the ninth 429'd, and the chain walked the OpenRouter tier -- where every
entry answered ``free-models-per-day, remaining 0`` -- and the turn **wrote its
notes and returned no answer at all**.

**What is worth stating plainly, because it is smaller than it sounds.** A 429
comes back in 0.23s (measured), so skipping dead entries saves tenths of a second,
not seconds. These tests exist because a dead tier is *indistinguishable from a slow
one* in a trace, and because the alternative -- a turn that produces no answer after
doing all its work -- is the worst outcome available. They are not here to make the
chain fast.

The parsing tests run against a **real captured 429 body** rather than a hand-written
one, because the whole mechanism depends on string-scraping a provider's error
message (LiteLLM discards the response headers -- measured, they arrive empty) and a
fixture invented to match the regex would prove nothing.

Every "what does it do" test below is paired with a "what must it not do": the
breaker keys on *account* rather than model, holds a cooldown as a clock rather than
a latch, never shortens a deadline it already learned, and never holds a key in
anything it returns.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from text_summarizer import breaker

# A real 429 body from OpenRouter, verbatim except that the reset epoch is
# recomputed per test. Captured 2026-10-02 from a genuinely exhausted account:
#   {"error":{"message":"Rate limit exceeded: free-models-per-day...",
#    "code":429,"metadata":{"headers":{"X-RateLimit-Limit":"50",
#    "X-RateLimit-Remaining":"0","X-RateLimit-Reset":"1790985600000"},...}}}
# LiteLLM embeds this in the exception message after the provider name.
BODY_TEMPLATE = (
    'litellm.RateLimitError: RateLimitError: OpenrouterException - '
    '{{"error":{{"message":"Rate limit exceeded: free-models-per-day. Add 5 '
    'credits to unlock 1000 free model requests per day","code":429,'
    '"metadata":{{"headers":{{"X-RateLimit-Limit":"50",'
    '"X-RateLimit-Remaining":"0","X-RateLimit-Reset":"{reset}"}},'
    '"limit_source":"openrouter_free_tier_daily"}}}}'
)


def _rate_limit_error(reset_in_s: float) -> Exception:
    """A real-shaped 429 exception whose reset is ``reset_in_s`` from now."""
    epoch_ms = int((time.time() + reset_in_s) * 1000)
    return RuntimeError(BODY_TEMPLATE.format(reset=epoch_ms))


def _no_network(monkeypatch, error):
    """Make the *delegate* raise ``error`` without any HTTP.

    Stubs ``LiteLlm.generate_content_async`` rather than ``litellm.acompletion``:
    the breaker wraps ``super().generate_content_async``, so stubbing anything below
    it is both ineffective and, worse, would let a fabricated key reach OpenRouter.
    """
    from google.adk.models.lite_llm import LiteLlm

    async def _raise(self, llm_request, stream=False):
        raise error
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(LiteLlm, "generate_content_async", _raise)


def _request():
    """A minimal real ``LlmRequest``.

    Passing ``None`` looks like a shortcut and is not one: ``LiteLlm`` reads
    ``llm_request.contents`` and ``llm_request.config`` before dispatch, so a
    ``None`` fails inside LiteLLM and never reaches the code under test -- the test
    would pass or fail for a reason unrelated to the breaker.
    """
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    return LlmRequest(
        model="",
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )


@pytest.fixture(autouse=True)
def _clean_breaker():
    """The state is module-level, so every test starts and ends empty."""
    breaker.clear()
    yield
    breaker.clear()


# --- reading the reset out of a real error ------------------------------------


def test_the_reset_time_is_read_from_a_real_openrouter_429():
    """The provider's own deadline, not a guess.

    This is the load-bearing test: if the scraping breaks, every trip silently
    becomes the 6-hour fallback, and a deployment with an hourly reset would be
    suppressed for six hours after each one. Silent, and slow to notice.
    """
    delta = breaker.reset_at_from_error(_rate_limit_error(3600))
    assert delta is not None
    assert 3500 < delta < 3700, f"expected ~1h, got {delta}"


def test_an_error_with_no_reset_header_falls_back_rather_than_guessing():
    """No header is the common case for providers that do not send one."""
    assert breaker.reset_at_from_error(RuntimeError("429 Too Many Requests")) is None


def test_a_reset_in_the_past_is_refused():
    """A clock behind the provider's must not read as "resets immediately".

    Read literally this would return a negative delta, and `is_spent` would treat it
    as expired -- so the account would be probed again immediately, forever, which is
    the no-breaker behaviour with extra steps.
    """
    past = int((time.time() - 7200) * 1000)
    assert breaker.reset_at_from_error(RuntimeError(BODY_TEMPLATE.format(reset=past))) is None


def test_an_absurd_reset_is_refused():
    """A garbage header must not be able to pin an account off for a week."""
    far = int((time.time() + 30 * 24 * 3600) * 1000)
    assert breaker.reset_at_from_error(RuntimeError(BODY_TEMPLATE.format(reset=far))) is None


def test_parsing_never_raises_on_junk():
    """It runs on the failure path, where raising would replace a 429 with a
    different and harder failure."""
    for junk in ("", "{}", "not json at all", '{"error":', "X-RateLimit-Reset: abc"):
        breaker.reset_at_from_error(RuntimeError(junk))


# --- tripping and recovering ---------------------------------------------------


def test_a_tripped_account_is_suppressed_and_recovers_on_its_own():
    key = "sk-or-v1-" + "a" * 40
    assert breaker.is_spent(key) is False

    breaker.trip(key, _rate_limit_error(3600))
    assert breaker.is_spent(key) is True

    # An hour later the deadline has passed, so the account is tried again -- and the
    # stale entry is dropped rather than accumulating across days.
    assert breaker.is_spent(key, now=time.monotonic() + 3700) is False
    assert breaker.spent_accounts() == []


def test_the_cooldown_is_a_clock_and_not_a_latch():
    """Reading must not clear the entry.

    A turn makes several model calls. Clearing on read would re-probe within the
    same turn, which is the exact behaviour the breaker exists to remove.
    """
    key = "sk-or-v1-" + "b" * 40
    breaker.trip(key, _rate_limit_error(3600))
    for _ in range(10):
        assert breaker.is_spent(key) is True


def test_a_trip_without_an_error_uses_the_fallback_cooldown():
    key = "sk-or-v1-" + "c" * 40
    applied = breaker.trip(key)
    assert applied == breaker.DEFAULT_COOLDOWN_S
    assert breaker.is_spent(key) is True


def test_a_second_trip_cannot_shorten_a_deadline_already_learned():
    """A 429 with no reset header must not cut short a real one.

    Without this, one uninformative error partway through a cooldown would halve it,
    and a noisy client could keep an account permanently half-suppressed.

    **The first trip must carry a real reset header and the second must not** -- the
    opposite pairing asserts nothing. An earlier version of this test used two
    headerless errors, both of which fell back to :data:`DEFAULT_COOLDOWN_S`, so the
    remaining time was identical either way and the test passed against code with
    the guard deleted. Verified: removing ``max()`` did not fail it.
    """
    key = "sk-or-v1-" + "d" * 40
    # A short, provider-declared deadline: 120s, well under the 6h fallback, so a
    # shortening is unmistakable rather than lost in the noise.
    breaker.trip(key, _rate_limit_error(120))
    learned = breaker.spent_accounts()[0].until - time.monotonic()
    assert learned < breaker.DEFAULT_COOLDOWN_S, (
        "the first trip did not use the provider's reset time, so this test "
        "cannot distinguish a shortened deadline from an unchanged one"
    )

    breaker.trip(key, RuntimeError("429 Too Many Requests"))
    after = breaker.spent_accounts()[0].until - time.monotonic()

    # Both directions matter, and this is the one that is easy to miss: a headerless
    # 429 must not *lengthen* a deadline the provider said was 120s away either.
    # Measured before the guard: learned=120.0s, after=21600.0s -- a 6-hour guess
    # silently replaced a two-minute fact, and the account then stayed suppressed
    # long past the moment its quota actually returned.
    assert abs(after - learned) < 2, (
        f"a headerless 429 moved the cooldown from {learned:.0f}s to {after:.0f}s"
    )


def test_a_real_reset_never_shortens_a_longer_known_deadline():
    """The other direction: a contradictory shorter header must not cut a deadline.

    A provider reporting a shorter reset for the same account is reporting an
    inconsistency, not a new fact, and acting on it would re-probe an account that
    is still exhausted.
    """
    key = "sk-or-v1-" + "6" * 40
    breaker.trip(key, _rate_limit_error(7200))
    learned = breaker.spent_accounts()[0].until - time.monotonic()
    breaker.trip(key, _rate_limit_error(60))
    after = breaker.spent_accounts()[0].until - time.monotonic()
    assert after >= learned - 2, (
        f"a contradictory shorter reset cut the cooldown from {learned:.0f}s to {after:.0f}s"
    )


def test_tripping_an_empty_key_is_a_no_op():
    assert breaker.trip("") is None
    assert breaker.is_spent("") is False


# --- per-account, not per-model ------------------------------------------------


def test_two_accounts_are_tracked_independently():
    """The budget is per account, so one being spent says nothing about the other.

    This is the property the whole design turns on: keying by *model* would suppress
    one entry while the other five on the same dead account kept probing.
    """
    first, second = "sk-or-v1-" + "e" * 40, "sk-or-v1-" + "f" * 40
    breaker.trip(first, _rate_limit_error(3600))
    assert breaker.is_spent(first) is True
    assert breaker.is_spent(second) is False


def test_the_same_account_is_one_entry_across_all_its_models():
    """Two entries, one account, one suppression."""
    key = "sk-or-v1-" + "9" * 40
    breaker.trip(key, _rate_limit_error(3600))
    breaker.trip(key, _rate_limit_error(3600))
    assert len(breaker.spent_accounts()) == 1


# --- the key must never leak ----------------------------------------------------


def test_nothing_returned_ever_contains_the_key():
    """A breaker that logs a credential is worse than no breaker.

    Asserted over every accessor rather than trusted: `fingerprint` is the only thing
    permitted out, and it has to stay that way under repr, str and the diagnostics.
    """
    key = "sk-or-v1-" + "s" * 40 + "SECRETSECRETSECRET"
    breaker.trip(key, _rate_limit_error(3600))

    assert key not in str(breaker.spent_accounts())
    assert key not in repr(breaker.spent_accounts())
    assert key not in breaker._fingerprint(key)
    # And the fingerprint is stable, which is what lets it key on the account at all.
    assert breaker._fingerprint(key) == breaker._fingerprint(key)
    assert len(breaker._fingerprint(key)) == 8


def test_the_fingerprint_separates_two_accounts():
    assert breaker._fingerprint("sk-or-v1-aaa") != breaker._fingerprint("sk-or-v1-bbb")


# --- the chain integration -----------------------------------------------------


def test_a_spent_account_is_left_out_of_the_chain(monkeypatch):
    """The cheap half: an account already known dead never enters the chain."""
    from text_summarizer import agent as agent_module

    spent = "sk-or-v1-" + "1" * 40
    live = "sk-or-v1-" + "2" * 40
    monkeypatch.setenv("OPENROUTER_API_KEY", f"{spent};{live}")
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_test")
    breaker.clear()
    breaker.trip(spent, _rate_limit_error(3600))

    chain = agent_module.get_model()
    bound = {
        getattr(e, "_additional_args", {}).get("api_key")
        for e in chain.models
        if hasattr(e, "_additional_args")
    }
    assert spent not in bound, "a spent account is still in the chain"
    assert live in bound, "the live account was dropped along with it"


def test_an_openrouter_entry_trips_on_429_and_raises(monkeypatch):
    """Tripping is what makes the *next* call cheap; raising is what this turn needs.

    Both halves matter. Raising without tripping would leave the account in the chain
    for every subsequent turn; tripping without raising would not stop this call.
    """
    import asyncio

    import litellm
    from text_summarizer import agent as agent_module

    key = "sk-or-v1-" + "3" * 40
    llm = agent_module._BreakerLiteLlm("qwen/qwen3.8-27b:free", key)
    boom = litellm.RateLimitError(
        message=BODY_TEMPLATE.format(reset=int((time.time() + 3600) * 1000)),
        llm_provider="openrouter",
        model="qwen/qwen3.8-27b:free",
    )
    boom.status_code = 429
    _no_network(monkeypatch, boom)

    async def _drive():
        async for _ in llm.generate_content_async(_request()):
            pass

    with pytest.raises(litellm.RateLimitError):
        asyncio.run(_drive())

    assert breaker.is_spent(key) is True


def test_a_non_429_error_does_not_trip_the_breaker(monkeypatch):
    """A 500 or a 400 says nothing about quota.

    Trips on those would suppress a healthy account for six hours because one
    upstream had a bad minute -- which is a far worse failure than the 0.23s this
    module saves.
    """
    import asyncio

    import litellm
    from text_summarizer import agent as agent_module

    key = "sk-or-v1-" + "4" * 40
    llm = agent_module._BreakerLiteLlm("qwen/qwen3.8-27b:free", key)
    boom = litellm.InternalServerError(
        message="upstream is down", llm_provider="openrouter", model="qwen"
    )
    boom.status_code = 500
    _no_network(monkeypatch, boom)

    async def _drive():
        async for _ in llm.generate_content_async(_request()):
            pass

    with pytest.raises(litellm.InternalServerError):
        asyncio.run(_drive())

    assert breaker.is_spent(key) is False


def test_a_spent_account_raises_without_making_a_request(monkeypatch):
    """The whole point: no round trip to be told what we already know.

    Asserted by never letting the call through -- a network attempt would raise a
    *different* error, so the test distinguishes them by message.
    """
    from text_summarizer import agent as agent_module

    key = "sk-or-v1-" + "5" * 40
    breaker.trip(key, _rate_limit_error(3600))
    llm = agent_module._BreakerLiteLlm("qwen/qwen3.8-27b:free", key)
    # No stub: this test asserts the request is *never attempted*, so the delegate
    # must be the real one. If the breaker leaked, this would reach the network --
    # which is why the key here is fabricated and any request would 401 rather than
    # do anything.

    async def _drive():
        async for _ in llm.generate_content_async(_request()):
            pass

    with pytest.raises(RuntimeError, match="rate-limited until its reset time"):
        asyncio.run(_drive())
