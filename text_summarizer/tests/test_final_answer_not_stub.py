"""The turn must end on an answer, not on a stub.

Found live on session `flow-r5` (2026-10-03), on **every turn** of a three-turn
booking flow, with the deployed model chain:

```
  8. TEXT len=2701  "Hoje é sábado, 03/10/2026…"   + CALL save_summary_to_second_brain
 10. CALL log_conversation
 12. TEXT len= 500  "**Text Summarizer Agent**\n\n**Sources**\n- [web]…"
```

The answer was real — it carried the session time, the seat map, the age-gate notice
and the checkout URL — and it sat on the event that also carried a tool call. The
turn then *ended* on a stub: the renderer faithfully wrapping a response the model had
used for nothing but its citation lines.

**Why this is the check that matters.** It is gotcha 16's family, inverted. That one
was two copies of the answer; this is one copy and a stub after it. Every other check
in the harness passes on it:

- `_no_error` — there is no error
- `_answered` — there *is* text
- `_cited_web_source` — the Sources block is exactly what it looks for
- `_single_answer` — the stub is not a *repeat* of anything

and so do the trace, `response_match_score`, and any UI that collapses tool-call
turns. The turn looked healthy everywhere except where the user was reading.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

APP = "text_summarizer"

_spec = importlib.util.spec_from_file_location(
    "scenarios", pathlib.Path(__file__).resolve().parents[2] / "tools" / "scenarios.py"
)
scenarios = importlib.util.module_from_spec(_spec)
# Registered before exec: scenarios.py uses dataclasses, and a dataclass whose
# module is absent from sys.modules cannot resolve its own __qualname__.
sys.modules["scenarios"] = scenarios
_spec.loader.exec_module(scenarios)

STUB = (
    "**Text Summarizer Agent**\n\n"
    "**Sources**\n"
    "- [web][searxng][space-bunny-free]<https://www.ingresso.com/filmes?city=osasco>: listagem.\n"
)


def _event(author, parts):
    return {"author": author, "content": {"parts": parts}}


def _call(name):
    return {"function_call": {"name": name, "args": {}}}


def _turn(*events):
    return scenarios.Turn(list(events), "prompt")


def _text_event(text):
    return _event(APP, [{"text": text}])


# --- answer_body: the furniture is not the answer -----------------------------


def test_the_name_stamp_is_not_an_answer():
    assert scenarios.answer_body("**Text Summarizer Agent**") == ""


def test_a_sources_block_is_not_an_answer():
    assert scenarios.answer_body(STUB) == "", (
        "the stamp and the Sources block are both written by this repo's "
        "after_model_callback; neither is the model saying anything"
    )


def test_real_content_survives_both_strips():
    body = scenarios.answer_body(
        "**Text Summarizer Agent**\n\n- 15:20 no Cinemark Osasco\n\n"
        "**Sources**\n- [web][searxng][m]<https://x.test/>: why\n"
    )
    assert body == "- 15:20 no Cinemark Osasco", repr(body)


def test_an_answer_with_no_sources_is_untouched():
    """The control. A stripper that ate real text would pass the tests above."""
    assert scenarios.answer_body("**Text Summarizer Agent**\n\n- two dogs") == "- two dogs"


# --- the check ---------------------------------------------------------------


def test_a_stub_last_event_fails_even_though_the_answer_was_produced():
    """The exact live shape, replayed.

    Every other check in the harness passes this turn, which is why it needs its own.
    """
    answer = (
        "Confirmado: **15:20 é *Resident Evil*** no Cinemark Osasco.\n\n"
        "**Link da sessão:** [Resident Evil 15:20, Sala 6]"
        "(https://checkout.ingresso.com/?sessionId=87205315&partnership=home)"
    )
    turn = _turn(
        _text_event("Vou confirmar."),
        _event(APP, [{"text": answer}, _call("save_summary_to_second_brain")]),
        _event(APP, [_call("log_conversation")]),
        _text_event(STUB),
    )

    assert scenarios._no_error(turn)[0] is True
    assert scenarios._single_answer(turn)[0] is True, "the stub is not a repeat"

    ok, detail = scenarios._final_answer_has_a_body(turn)
    assert ok is False, "the turn ends on a stub and every other check passed"
    assert "stub" in detail
    assert "one message" in detail, (
        "the detail has to say what the fix is, not only that something is wrong"
    )


def test_a_healthy_turn_passes():
    """The control, or the check would pass by refusing everything."""
    turn = _turn(
        _event(APP, [{"text": "I'll look that up."}, _call("web_search")]),
        _text_event("**Text Summarizer Agent**\n\n- Python 3.13 is current.\n\n"
                    "**Sources**\n- [web][searxng][m]<https://x.test/>: why\n"),
    )
    ok, detail = scenarios._final_answer_has_a_body(turn)
    assert ok is True, detail


def test_a_turn_with_no_sources_at_all_passes():
    """A vault-only answer has no Sources block and must not be judged a stub."""
    turn = _turn(_text_event("**Text Summarizer Agent**\n\n- Two notes about dogs."))
    assert scenarios._final_answer_has_a_body(turn)[0] is True


def test_a_turn_with_no_text_at_all_fails():
    """It is not a stub, it is nothing -- and it still must not pass."""
    turn = _turn(_event(APP, [_call("current_datetime")]))
    ok, detail = scenarios._final_answer_has_a_body(turn)
    assert ok is False
    assert "no answer text" in detail


# --- it is applied to every scenario, not composed into them ------------------


def test_the_check_is_applied_by_the_runner_and_not_per_scenario():
    """A universal guard has to be unreachable from a scenario's `check=`.

    Composing it into each scenario's list would work until someone adds a scenario
    and forgets, and a forgotten guard is a guard that is not there.
    """
    import inspect

    src = inspect.getsource(scenarios.run_scenario)
    assert "_final_answer_has_a_body(turn)" in src, (
        "the runner must apply this itself; a scenario that omits it would otherwise "
        "pass a stub turn"
    )
    # And it runs even when the scenario's own check passed.
    assert src.index("scenario.check(turn)") < src.index("_final_answer_has_a_body(turn)")


def test_no_scenario_passes_a_stub_on_its_own_but_the_runner_catches_every_one():
    """Why the check had to be universal rather than composed into each scenario.

    Scenarios that assert on *tool calls* pass a stub turn perfectly well: the
    tools ran, nothing errored, there is text, the Sources block is there. So a
    per-scenario `check=` could not have carried this — the harness has to apply it,
    and applying it is the only thing that makes the whole set honest.
    """
    stub = scenarios.Turn([_event(APP, [{"text": STUB}])], "p")

    rescued_by_its_own_checks = [
        s.name for s in scenarios.SCENARIOS if s.check(stub)[0] is True
    ]
    assert rescued_by_its_own_checks, (
        "no scenario passes a stub on its own, which means the fixture no longer "
        "represents the defect and this test would pass for the wrong reason"
    )

    for name in rescued_by_its_own_checks:
        scenario = next(s for s in scenarios.SCENARIOS if s.name == name)
        assert scenario.check(stub)[0] is True
        # And what the runner does with it.
        assert scenarios._final_answer_has_a_body(stub)[0] is False, (
            f"{name} would report PASS on a turn that told the user nothing"
        )
