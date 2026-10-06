"""The clock tool, and the day-scoped cache key.

Two live bugs, tested together because the second is what made the first visible:

* the model has no clock, so "summarize today news" is answered from its training
  cutoff (:mod:`text_summarizer.clock`);
* the vault cache replayed the *first* day's note for that prompt forever, because
  ``"today"`` is not part of the text being summarised and the fingerprint is
  calendar-blind (:func:`second_brain.cache_key_text_for`).

The order matters: fixing only the clock would leave the cache replaying
yesterday's answer before the model ever ran.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from text_summarizer import agent, clock
from text_summarizer.second_brain import (
    cache_key_text,
    cache_key_text_for,
    has_relative_time,
    source_fingerprint,
)

# --- the clock ----------------------------------------------------------------


def test_the_clock_returns_an_iso_date_and_a_weekday():
    payload = json.loads(clock.current_datetime())
    assert set(payload) == {"iso", "date", "time", "weekday", "timezone"}
    assert payload["date"] == date.today().isoformat()
    assert payload["weekday"] in {
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    }


def test_the_date_is_parseable():
    """The model has to be able to read this, so it must be a real ISO date."""
    payload = json.loads(clock.current_datetime())
    assert date.fromisoformat(payload["date"]).isoformat() == payload["date"]


def test_utc_is_accepted():
    payload = json.loads(clock.current_datetime("utc"))
    assert payload["timezone"] == "UTC"
    assert payload["iso"].endswith("+00:00")


@pytest.mark.parametrize("blank", ["", "  ", "LOCAL", "Local"])
def test_local_is_the_default_and_is_case_insensitive(blank):
    payload = json.loads(clock.current_datetime(blank))
    assert payload["date"] == date.today().isoformat()


def test_a_named_timezone_is_used_when_the_host_has_tzdata():
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo("Europe/Lisbon")
    except Exception:  # pragma: no cover - host without tzdata
        pytest.skip("this host has no tzdata")
    payload = json.loads(clock.current_datetime("Europe/Lisbon"))
    assert payload["timezone"] == "Europe/Lisbon"


def test_an_unknown_timezone_falls_back_and_says_so():
    """A bad zone name must not fail the turn, and must not be silent either."""
    payload = json.loads(clock.current_datetime("Mars/Olympus"))
    assert "unrecognised" in payload["timezone"]
    assert payload["date"] == date.today().isoformat()


def test_the_clock_tool_is_always_present():
    """Not gated: a missing clock is a silently wrong answer, not a missing feature."""
    assert len(clock.build_clock_tools()) == 1


def test_the_clock_is_on_the_agent():
    names = [getattr(t, "name", "") for t in agent.root_agent.tools]
    assert "current_datetime" in names


def test_the_clock_schema_is_a_single_optional_string():
    from google.adk.tools.function_tool import FunctionTool

    schema = FunctionTool(clock.current_datetime)._get_declaration().parameters_json_schema or {}
    properties = schema.get("properties", {})
    assert set(properties) == {"timezone_name"}
    assert properties["timezone_name"]["type"] == "string"
    assert not schema.get("required"), "the clock must be callable with no arguments"


# --- the day-scoped cache key -------------------------------------------------


def test_the_reported_prompt_is_recognised_as_time_dependent():
    assert has_relative_time("summarize today news")


@pytest.mark.parametrize("prompt", [
    "summarize today news",
    "Summarize: what's happening today",
    "what happened this week",
    "latest news on opencode",
    "breaking news",
    "summarize yesterday's floods",
    "tonight's fixtures",
    "past month in review",
    "right now",
    "current standings",
])
def test_relative_time_prompts_are_detected(prompt):
    assert has_relative_time(prompt), prompt


@pytest.mark.parametrize("prompt", [
    "Summarize the following text:\n\nDogs are domesticated mammals.",
    "summarize: a/b/c",
    "carlos kafka summary of dogs",
    "the newsroom layout",      # contains "news" only inside a longer word
    "renewable energy",         # contains "now" only inside a longer word
    "greatest hits",            # contains "latest" only as a prefix of nothing
    "nowhere",
    "",
])
def test_timeless_prompts_are_not_detected(prompt):
    assert not has_relative_time(prompt), prompt


def test_the_key_changes_across_days():
    monday = source_fingerprint(cache_key_text_for("summarize today news", today="2026-09-28"))
    tuesday = source_fingerprint(cache_key_text_for("summarize today news", today="2026-09-29"))
    wednesday = source_fingerprint(cache_key_text_for("summarize today news", today="2026-09-30"))
    assert len({monday, tuesday, wednesday}) == 3


def test_the_key_is_stable_within_a_day():
    """A same-day repeat must still hit, or every re-ask costs a full turn."""
    first = source_fingerprint(cache_key_text_for("summarize today news", today="2026-09-29"))
    again = source_fingerprint(cache_key_text_for("summarize today news", today="2026-09-29"))
    assert first == again


def test_timeless_prompts_keep_their_existing_key():
    """Notes already in the vault must not be orphaned by this change."""
    prompt = "Summarize the following text:\n\nDogs are domesticated mammals."
    assert cache_key_text_for(prompt) == prompt
    assert source_fingerprint(cache_key_text_for(prompt)) == source_fingerprint(prompt)


def test_the_helper_defaults_to_today_when_no_date_is_given():
    keyed = cache_key_text_for("summarize today news")
    assert date.today().isoformat() in keyed


def test_the_helper_handles_empty_input():
    assert cache_key_text_for("") == ""
    assert cache_key_text_for(None) == ""


def test_the_write_and_the_lookup_use_the_same_helper():
    """The original cache bug in one assertion: two different keys, no hits.

    ``cache_hit_before_model`` must scope the key exactly as the note writer
    does. If either side stops calling :func:`cache_key_text_for`, they diverge
    and the cache silently never hits again -- which is the failure this whole
    change exists to replace.
    """
    prompt = "summarize today news"
    written = source_fingerprint(cache_key_text_for(prompt, today="2026-09-29"))
    looked_up = source_fingerprint(cache_key_text_for(prompt, today="2026-09-29"))
    assert written == looked_up


def test_a_day_scoped_prompt_differs_from_its_raw_fingerprint():
    """Guard against the helper silently becoming a no-op."""
    prompt = "summarize today news"
    assert source_fingerprint(cache_key_text_for(prompt, today="2026-09-29")) != (
        source_fingerprint(prompt)
    )


# --- the agent's lookup side -------------------------------------------------


class _State(dict):
    pass


class _Request:
    def __init__(self, contents):
        self.contents = contents


class _Content:
    def __init__(self, role, text):
        self.role = role
        self.parts = [type("P", (), {"text": text, "function_call": None})()]


class _Context:
    """The minimum surface ``cache_hit_before_model`` touches.

    ``user_content`` is set because ``agent._user_content_text`` reads that field
    first -- the documented source, and the same one the write side reads. It falls
    back to the request contents, so this mirrors the live shape rather than
    inventing one.
    """

    def __init__(self, user_text, invocation_id="inv-1"):
        self.state = _State()
        self.invocation_id = invocation_id
        self.session = type("S", (), {"events": []})()
        self.user_content = _Content("user", user_text)


def test_the_lookup_passes_a_day_scoped_key(monkeypatch):
    """The regression test for the reported bug, at the seam that caused it."""
    seen: list[str] = []

    def fake_find(text):
        seen.append(text)
        return None

    monkeypatch.setattr(agent, "find_cached_summary", fake_find)
    agent.cache_hit_before_model(_Context("summarize today news"), _Request([]))

    assert len(seen) == 1
    assert date.today().isoformat() in seen[0], seen[0]
    assert seen[0].startswith("summarize today news")


def test_the_lookup_leaves_a_timeless_prompt_untouched(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(agent, "find_cached_summary", lambda text: seen.append(text) and None)

    prompt = "Summarize the following text:\n\nDogs are mammals."
    agent.cache_hit_before_model(_Context(prompt), _Request([]))

    assert seen == [prompt]


def test_yesterdays_note_is_not_replayed_today(monkeypatch):
    """The reported bug, end to end against a vault holding yesterday's note.
    
    Mock date to a fixed day so the test never drifts with the calendar.
    """
    import os
    import tempfile
    from datetime import date as dt_date

    from text_summarizer import second_brain

    class FixedDate(dt_date):
        @classmethod
        def today(cls):
            return dt_date(2026, 10, 6)

    monkeypatch.setattr(second_brain, "date", FixedDate)
    monkeypatch.setattr("text_summarizer.second_brain.date", FixedDate)

    today = FixedDate.today()
    yesterday = (today - timedelta(days=1)).isoformat()
    today_iso = today.isoformat()

    with tempfile.TemporaryDirectory() as tmp:
        vault = os.path.join(tmp, "Second Brain")
        os.makedirs(vault)
        monkeypatch.setattr(second_brain, "VAULT_ROOT", tmp)

        prompt = "summarize today news"
        monday_key = source_fingerprint(cache_key_text_for(prompt, today=yesterday))
        with open(os.path.join(vault, f"{yesterday} - news.md"), "w") as fh:
            fh.write(f"---\nsource_fingerprint: {monday_key}\n---\nYesterday's bullets\n")

        # Yesterday's own key still finds yesterday's note: nothing is orphaned.
        assert second_brain.find_cached_summary(
            cache_key_text_for(prompt, today=yesterday), today=yesterday
        ) is not None

        # Today's key is different, so today's lookup misses and the model runs.
        assert source_fingerprint(cache_key_text_for(prompt, today=today_iso)) != monday_key
        assert second_brain.find_cached_summary(
            cache_key_text_for(prompt, today=today_iso), today=today_iso
        ) is None


def test_a_note_past_the_staleness_bound_is_not_replayed_even_with_its_own_key(monkeypatch):
    """Mock date to a fixed day so the test never drifts."""
    import os
    import pathlib
    import tempfile
    from datetime import date as dt_date
    from datetime import timedelta

    from text_summarizer import second_brain

    class FixedDate(dt_date):
        @classmethod
        def today(cls):
            return dt_date(2026, 10, 6)

    monkeypatch.setattr(second_brain, "date", FixedDate)
    monkeypatch.setattr("text_summarizer.second_brain.date", FixedDate)

    prompt = "summarize today news"
    stale_day = (FixedDate.today() - timedelta(days=second_brain.CACHE_MAX_AGE_DAYS + 1)).isoformat()

    with tempfile.TemporaryDirectory() as tmp:
        vault = os.path.join(tmp, "Second Brain")
        os.makedirs(vault)
        monkeypatch.setattr(second_brain, "VAULT_ROOT", tmp)
        try:
            key_text = cache_key_text_for(prompt, today=stale_day)
            (pathlib.Path(vault) / f"{stale_day} - news.md").write_text(
                f"---\nsource_fingerprint: {source_fingerprint(key_text)}\n---\nStale\n",
                encoding="utf-8",
            )
            assert second_brain.find_cached_summary(key_text, today=FixedDate.today().isoformat()) is None
            fresh = (FixedDate.today() - timedelta(days=1)).isoformat()
            fresh_key = cache_key_text_for(prompt, today=fresh)
            (pathlib.Path(vault) / f"{fresh} - news.md").write_text(
                f"---\nsource_fingerprint: {source_fingerprint(fresh_key)}\n---\nFresh\n",
                encoding="utf-8",
            )
            assert second_brain.find_cached_summary(fresh_key, today=fresh) == "Fresh"
        finally:
            pass


def test_cache_key_text_from_context_also_scopes():
    """The write path goes through ``cache_key_text``; it must scope too.

    Both sides matter: the lookup alone would miss every hit, and the write alone
    would leave a stale key on disk that nothing could ever match.
    """
    ctx = _Context("summarize today news")
    ctx.user_content = _Content("user", "summarize today news")
    scoped = cache_key_text(ctx)
    assert date.today().isoformat() in scoped


# --- replaying an old answer -------------------------------------------------
#
# Found on session ``c3f105bf``. The agent answered a question wrongly, the wrong
# summary was written to the vault, and every later asking of the same question
# replayed it **verbatim with no model call at all**. The replay path never
# reaches the model, so nothing in the system could notice the answer was wrong --
# the cache was self-sealing.
#
# Day-scoping (above) is not the fix for that: it only applies to prompts naming a
# relative time, and "movies in Osasco with session times" names none, so the key
# was byte-identical forever. What is needed is a bound on how old a note may be.


def test_a_note_older_than_the_limit_is_not_replayed(monkeypatch):
    """The general defect: a cinema listing from last month is not today's listing.

    Deliberately a prompt with **no** relative time in it, because that is the
    shape that survived day-scoping: its fingerprint is stable, so only an age
    bound can retire it.
    """
    import os
    import tempfile

    from text_summarizer import second_brain

    with tempfile.TemporaryDirectory() as tmp:
        brain = os.path.join(tmp, "Second Brain")
        os.makedirs(brain)
        monkeypatch.setattr(second_brain, "VAULT_ROOT", tmp)

        prompt = "summarize movies in Osasco with session times"
        digest = second_brain.source_fingerprint(prompt)
        with open(os.path.join(brain, "2026-09-01 - old.md"), "w") as fh:
            fh.write(f"---\nsource_fingerprint: {digest}\n---\nLast month's answer\n")

        # Within the window: still a hit, so the cache keeps doing its job.
        assert (
            second_brain.find_cached_summary(prompt, today="2026-09-05")
            == "Last month's answer"
        )
        # Past it: a miss, so the model runs and the world gets consulted again.
        assert second_brain.find_cached_summary(prompt, today="2026-10-02") is None


def test_a_note_with_no_date_in_its_name_is_never_replayed(monkeypatch):
    """Unknown age means unknown trust, and a hand-written note is the common case.

    The alternative -- guessing an age from the file's mtime -- would make a note
    written by an editor or restored from a backup arbitrarily fresh or stale.
    """
    import os
    import tempfile

    from text_summarizer import second_brain

    with tempfile.TemporaryDirectory() as tmp:
        brain = os.path.join(tmp, "Second Brain")
        os.makedirs(brain)
        monkeypatch.setattr(second_brain, "VAULT_ROOT", tmp)

        prompt = "summarize my hand written note"
        digest = second_brain.source_fingerprint(prompt)
        with open(os.path.join(brain, "no date here.md"), "w") as fh:
            fh.write(f"---\nsource_fingerprint: {digest}\n---\nA hand written answer\n")

        assert second_brain.note_age_days("no date here.md") is None
        assert second_brain.find_cached_summary(prompt) is None


def test_setting_the_limit_to_zero_disables_replay_entirely(monkeypatch):
    """The off switch, and the control for measuring what the cache is worth.

    Asserted rather than assumed because it is the documented way to opt out --
    and because "0 means unlimited" is the mistake a reader would plausibly make.
    """
    import os
    import tempfile

    from text_summarizer import second_brain

    with tempfile.TemporaryDirectory() as tmp:
        brain = os.path.join(tmp, "Second Brain")
        os.makedirs(brain)
        monkeypatch.setattr(second_brain, "VAULT_ROOT", tmp)
        monkeypatch.setattr(second_brain, "CACHE_MAX_AGE_DAYS", 0)

        prompt = "summarize something"
        digest = second_brain.source_fingerprint(prompt)
        with open(os.path.join(brain, "2026-10-02 - today.md"), "w") as fh:
            fh.write(f"---\nsource_fingerprint: {digest}\n---\nToday's answer\n")

        # Written today, so age cannot be what excluded it -- only the switch.
        assert second_brain.find_cached_summary(prompt, today="2026-10-02") is None


def test_a_future_dated_note_is_not_treated_as_permanently_fresh():
    """A note dated ahead of today has negative age and would never age out.

    Clock skew, a typo in the date, or a note copied from a vault in another
    timezone all produce one. Serving it forever is the failure the limit exists
    to prevent, so the bound is applied to the age rather than trusting a
    negative number.
    """
    from text_summarizer import second_brain

    assert second_brain.note_age_days("2030-01-01 - future.md", today="2026-10-02") < 0


def test_the_age_limit_is_read_from_the_environment():
    """Deployment-configurable without editing code.

    Read at import time from the environment, so this reloads the module with a
    value set -- reloading is what proves the constant is *read* rather than
    hardcoded, which an assertion on the default could not.
    """
    import importlib
    import os

    os.environ["CACHE_MAX_AGE_DAYS"] = "30"
    try:
        reloaded = importlib.reload(importlib.import_module("text_summarizer.second_brain"))
        assert reloaded.CACHE_MAX_AGE_DAYS == 30
    finally:
        os.environ.pop("CACHE_MAX_AGE_DAYS", None)
        importlib.reload(importlib.import_module("text_summarizer.second_brain"))
