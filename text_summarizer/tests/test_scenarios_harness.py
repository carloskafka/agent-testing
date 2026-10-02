"""Offline tests for the scenario harness in ``tools/scenarios.py``.

The harness is the only thing in this repo that can prove a *prompt* produces the
intended tool calls against the code that is actually deployed, and it cannot run in
``pytest``: it spends Gemini quota, writes to the real vault and needs both Docker
services up. So the checkers -- the part that decides pass from fail -- are tested
here against synthetic events instead, and the harness itself stays out of the
ruff/mypy gate the way the rest of ``tools/`` is.

**A checker suite that is only ever fed a good turn proves nothing.** Every one of
these tests also feeds a turn that should fail, because the failure mode is
specific and quiet: a check that returns ``True`` for an empty answer, or that
matches a URL against an empty allow-list, is green forever and reports a healthy
agent while the web tier is dead. That is the same shape as
``test_a_wrongly_decoding_page_is_reported_rather_than_passed`` in
``test_tools_gif.py`` -- a decoder that returns zeros agrees with a correct one, so
something has to break on purpose for the comparison to mean anything.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_APP = "text_summarizer"

spec = importlib.util.spec_from_file_location(
    "scenarios", pathlib.Path(__file__).resolve().parents[2] / "tools" / "scenarios.py"
)
scenarios = importlib.util.module_from_spec(spec)
sys.modules["scenarios"] = scenarios
spec.loader.exec_module(scenarios)


# --- event builders ------------------------------------------------------------


def _event(author: str, parts: list[dict], **extra) -> dict:
    return {"author": author, "content": {"role": "model", "parts": parts}, **extra}


def _text_event(text: str, author: str = _APP, **extra) -> dict:
    return _event(author, [{"text": text}], **extra)


def _call(name: str, args: dict) -> dict:
    return {"function_call": {"name": name, "args": args}}


def _response(name: str, payload) -> dict:
    return {"function_response": {"name": name, "response": payload}}


def _turn(*events: dict) -> scenarios.Turn:
    return scenarios.Turn(list(events), "a prompt")


def _web_turn(url: str = "https://example.org/python", answer: str | None = None) -> scenarios.Turn:
    """A healthy web turn: search, a payload, and a cited answer."""
    body = answer or (
        f"**Text Summarizer Agent**\n\n- Python 3.13 is current.\n\n"
        f"**Sources**\n- [web][searxng][gemini-3.5-flash-lite]<{url}>: the release page"
    )
    return _turn(
        _text_event("a prompt", author="user"),
        _event(_APP, [_call("web_search", {"query": "python release"})], model_version="gemini-3.5-flash-lite"),
        _event(_APP, [_response("web_search", {"results": [{"url": url, "title": "Python"}]})]),
        _text_event(body, model_version="gemini-3.5-flash-lite"),
    )


# --- the properties the Turn view has to get right ----------------------------


def test_a_failed_turn_is_visible_as_an_error():
    """A turn that died on a 400 has an error event and no answer at all.

    The reason the harness reads the session rather than the HTTP response: the
    response to a dead turn is still 200.
    """
    turn = _turn(
        _text_event("a prompt", author="user"),
        _event(_APP, [{"error": True}], error_message="400 INVALID_ARGUMENT"),
    )

    assert turn.error
    assert not turn.answer.strip()
    assert scenarios._no_error(turn)[0] is False


def test_the_answer_joins_every_text_part_not_just_the_last():
    """Gotcha 13: the *last* assistant text is the correct copy even when the answer
    was emitted twice, so a last-event assertion passes against a duplicated answer.

    Concatenating every text event is what makes the duplication detectable at all,
    and ``_single_answer`` is the check that acts on it.
    """
    turn = _turn(_text_event("first copy"), _text_event("second copy"))

    assert "first copy" in turn.answer
    assert "second copy" in turn.answer
    assert scenarios._single_answer(turn)[0] is False


def test_a_healthy_turn_is_exactly_one_text_event():
    """The other side of the duplication check, so it is not vacuous."""
    assert scenarios._single_answer(_web_turn())[0] is True


def test_tool_results_are_read_in_order():
    """Two calls to one tool, distinguished by order.

    Order is what lets a check say "the search happened before the fetch", which is
    the difference between a two-hop scenario and two unrelated calls.
    """
    turn = _turn(
        _event(_APP, [_call("web_search", {})]),
        _event(_APP, [_response("web_search", {"results": [{"url": "https://a.example"}]})]),
        _event(_APP, [_call("web_fetch", {})]),
        _event(_APP, [_response("web_fetch", {"url": "https://a.example"})]),
    )

    assert turn.tool_calls == ["web_search", "web_fetch"]
    assert len(turn.web_urls_offered()) == 1


# --- the web checks ------------------------------------------------------------


def test_a_cited_web_url_must_have_the_provider_in_slot_two():
    """``[web][searxng]`` -- not ``[web][ck]``.

    Slot two is "where this came from". A web citation claiming the *vault* is the
    source is a renderer bug that would look like a working citation in the UI.
    """
    bad = _web_turn(answer="**Sources**\n- [web][ck][gemini]<https://example.org/python>: wrong slot")
    good = _web_turn()

    assert scenarios._cited_web_source(bad)[0] is False
    assert scenarios._cited_web_source(good)[0] is True


def test_a_cited_url_the_search_never_returned_is_caught():
    """The rule the renderer is supposed to enforce, asserted from both sides.

    The offer-list is read from this turn's own ``web_search`` payload, so the check
    cannot pass by having nothing to compare -- and it fails loudly when the search
    returned nothing rather than accepting a citation on faith.
    """
    invented = _web_turn(answer="**Sources**\n- [web][searxng][m]<https://made.up.example/x>: cited")
    assert scenarios._cited_urls_are_real(invented)[0] is False

    honest = _web_turn(answer="**Sources**\n- [web][searxng][m]<https://example.org/python>: cited")
    assert scenarios._cited_urls_are_real(honest)[0] is True


def test_a_citation_with_no_search_behind_it_fails():
    """A turn where the web tier returned nothing cannot pass the citation check.

    This is the vacuity guard. An earlier version of the check compared cited URLs
    against the offer-list; with both sets empty the intersection is empty, the
    "no invented URLs" condition holds, and the check reports a healthy citation on
    a turn where the web tier was never reached at all.
    """
    turn = _turn(
        _text_event("a prompt", author="user"),
        _text_event("**Sources**\n- [web][searxng][m]<https://example.org/python>: cited"),
    )

    ok, detail = scenarios._cited_urls_are_real(turn)
    assert ok is False
    assert "no URLs" in detail


def test_no_answer_fails_the_answered_check():
    """An empty answer is not an answer, and a length floor is what says so.

    A substring check ("does the answer mention Python?") passes on an empty string
    for a surprising number of inputs, so the check is on length instead.
    """
    assert scenarios._answered(_turn())[0] is False
    assert scenarios._answered(_turn(_text_event("too short")))[0] is False
    assert scenarios._answered(_turn(_text_event("x" * 200)))[0] is True


def test_a_tool_error_is_caught_by_the_coercion_scenario_check():
    """The live São Paulo failure, in the shape the harness must reject.

    ``failed to deserialize parameters: invalid type: string "50", expected usize``
    arrives as *data* with ``isError`` set -- the turn looks healthy from the
    outside, which is exactly why it needs asserting rather than noticing.
    """
    turn = _turn(
        _text_event("a prompt", author="user"),
        _event(_APP, [_call("search_text", {"max_results": "5"})]),
        _event(
            _APP,
            [
                _response(
                    "search_text",
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": 'failed to deserialize parameters: invalid type: string "50", expected usize',
                            }
                        ],
                        "isError": True,
                    },
                )
            ],
        ),
        _text_event("The weather in Sao Paulo is 22 degrees Celsius today." * 4),
    )

    ok, detail = scenarios._no_search_tool_error(turn)
    assert ok is False
    assert "search_text" in detail
    # ...and the turn as a whole still looks fine to the other checks, which is
    # what makes this one necessary.
    assert scenarios._no_error(turn)[0] is True


def test_a_clean_search_turn_has_no_tool_error():
    """The other side, so the check is not simply always-false."""
    turn = _turn(
        _text_event("a prompt", author="user"),
        _event(_APP, [_call("search_text", {"max_results": 5})]),
        _event(_APP, [_response("search_text", {"results": []})]),
        _text_event("Nothing in the vault matches that." * 6),
    )

    assert scenarios._no_search_tool_error(turn)[0] is True


def test_the_vault_scenario_fails_when_the_web_tier_fires():
    """The tier must be additive.

    Both directions are asserted: a turn that searched the web for a vault-only
    question fails, and one that did not passes. A check that only asserted the
    negative would pass a harness that never asked the question.
    """
    assert scenarios._web_not_used(_web_turn())[0] is False
    quiet = _turn(
        _text_event("a prompt", author="user"),
        _event(_APP, [_call("search_text", {"query": "dogs"})]),
        _text_event("You wrote about dogs in three notes." * 6),
    )
    assert scenarios._web_not_used(quiet)[0] is True


def test_a_metadata_leak_fails_the_ssrf_check():
    """A key in the answer is a failed refusal, whatever else it says."""
    leaked = _turn(
        _text_event("The endpoint returned access_key AKIAIOSFODNN7EXAMPLE and a secret token.")
    )
    assert scenarios._no_metadata_leak(leaked)[0] is False

    refused = _turn(
        _text_event(
            "I cannot fetch that address: the link-local range is refused as a "
            "cloud-metadata endpoint, so I have no data from it."
        )
    )
    assert scenarios._no_metadata_leak(refused)[0] is True


def test_a_stale_fact_answered_from_memory_fails():
    """A current fact asserted flatly, uncited and unhedged.

    The check is deliberately soft -- if the web tier failed, answering from
    knowledge is *correct* -- so it only fails the combination of no citation and no
    caveat, and reports which of the two it found.
    """
    confident = _turn(
        _text_event("The current Secretary-General of the United Nations is António Guterres.")
    )
    assert scenarios._no_confident_stale_answer(confident)[0] is False

    hedged = _turn(
        _text_event(
            "My training data may predate the current holder of that post. As of what I "
            "can verify, António Guterres held it, but please check a current source."
        )
    )
    assert scenarios._no_confident_stale_answer(hedged)[0] is True
    assert scenarios._no_confident_stale_answer(_web_turn())[0] is True


# --- the composition -----------------------------------------------------------


def test_a_scenario_names_every_expectation_that_broke():
    """``_all`` must not collapse into a single boolean.

    A composition that returned only "False" would leave a failing scenario with no
    way to tell which expectation broke -- and with six expectations per web
    scenario, that is the difference between a two-minute fix and an afternoon. So
    the detail carries every check's own message, not just the first.
    """
    dead = _turn(_event(_APP, [{"error": True}], error_message="400 INVALID_ARGUMENT"))
    check = scenarios._all([scenarios._no_error, scenarios._answered, scenarios._called("web_search")])

    ok, detail = check(dead)

    assert ok is False
    assert "INVALID_ARGUMENT" in detail
    assert "web_search" in detail


def test_a_scenario_that_never_ran_its_checks_cannot_pass():
    """A composition of zero checks is a vacuous pass, so it is rejected.

    Worth one test because a typo in a scenario's ``check=`` -- ``_all([])``, or a
    list comprehension that filtered everything out -- would otherwise report a
    healthy agent for a scenario that verified nothing.
    """
    with pytest.raises(ValueError, match="must assert something"):
        scenarios._all([])(_turn(_text_event("anything")))


def test_every_scenario_has_a_check_that_can_fail():
    """Structural guard over the scenario table itself.

    Each scenario's checks are run against a deliberately broken turn -- one that
    errors, answers nothing and calls nothing. A scenario that still passes has a
    check that cannot fail, which is the same defect as a vacuous composition, one
    level up: a scenario that always reports PASS.
    """
    broken = _turn(_event(_APP, [{"error": True}], error_message="400 INVALID_ARGUMENT"))

    for scenario in scenarios.SCENARIOS:
        ok, _ = scenario.check(broken)
        assert ok is False, f"{scenario.name} passes on a dead turn"


def test_every_scenario_prompt_is_non_empty_and_unique():
    """Two scenarios with the same prompt would share a cache entry.

    The vault cache is keyed on the prompt, so a repeated prompt makes the second
    scenario replay the first one's note with zero model calls -- and report PASS
    while testing nothing. That is gotcha 10, and it is invisible in a harness
    rather than obvious in a test.
    """
    prompts = [s.prompt for s in scenarios.SCENARIOS]
    assert all(p.strip() for p in prompts)
    assert len(prompts) == len(set(prompts)), "two scenarios share a prompt"


def test_every_scenario_declares_its_tags():
    """Tags are the only way to select a subset, and the ``--only`` prefix is the
    coarse half of that. An untagged scenario can only be found by reading the list.
    """
    for scenario in scenarios.SCENARIOS:
        assert scenario.tags, f"{scenario.name} has no tags"
        assert scenario.note, f"{scenario.name} has no note explaining what it is for"
