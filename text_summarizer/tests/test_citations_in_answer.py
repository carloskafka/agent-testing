"""The answer and its citations must be one message.

Found live on session `flow-r5` (2026-10-03), on **every turn** of a three-turn
booking flow, with the deployed model chain and a real prompt:

```
  8. TEXT len=2701  "Hoje é sábado, 03/10/2026…"   + CALL save_summary_to_second_brain
 10. CALL log_conversation
 12. TEXT len= 500  "**Text Summarizer Agent**\n\n**Sources**\n- [web]…"
```

The answer was real — session time, seat map, age-gate notice, checkout URL — and it
sat on the event that also carried a tool call. The turn then **ended on a stub**:
the renderer faithfully wrapping a response the model had used for nothing but its
citation lines.

**The stub wins everywhere it matters.** The trace reports the last event.
`response_match_score` scores the last event. `_answered`, `_cited_web_source` and
`_single_answer` all pass it. So does any UI that collapses tool-call turns. The turn
looked healthy in every one of those places while the user read a Sources block with
nothing above it.

**Why the model did it.** Rule 13 said to cite "at the very END of your answer", and
rules 8 and 9 put `save_summary_to_second_brain` and `log_conversation` *after* the
answer. So the citation lines came after a tool call, and "the very end of your
answer" was available to be read as "a final message of its own". The instruction
invited the split; the model took it.
"""

from __future__ import annotations

import pathlib
import re

AGENT = pathlib.Path(__file__).resolve().parents[1] / "agent.py"


def _instructions() -> str:
    return AGENT.read_text(encoding="utf-8")


def _rule(number: int) -> str:
    text = _instructions()
    start = re.search(rf"^{number}\. ", text, re.M)
    assert start, f"rule {number} not found"
    rest = text[start.start():]
    end = re.search(rf"^{number + 1}\. ", rest, re.M)
    return rest[: end.start()] if end else rest


# --- the instruction that caused it -------------------------------------------


def test_rule_13_puts_the_source_lines_in_the_answer_message():
    """The clause has to be a positive instruction, not a prohibition.

    "Do not send the citations separately" leaves the model to work out what
    *instead*; "they go in the same message as your bullets" is the shape.
    """
    rule = _rule(13)
    assert "IN THE SAME MESSAGE as your bullets" in rule


def test_rule_13_no_longer_says_the_very_end():
    """The phrase that caused the split must be gone, not merely supplemented.

    Leaving it in place alongside the new clause gives the model two readings of one
    instruction, and the old one is the one it was already following.
    """
    rule = _rule(13)
    assert "at the very END of your answer" not in rule, (
        "'at the very END of your answer' is what a model reads as 'a final message "
        "of its own'; it has to be removed, not just joined by a second reading"
    )


def test_rule_13_still_demands_the_citations():
    """The control: the clause must not have turned into "skip the sources"."""
    rule = _rule(13)
    assert "[web][@@ADK_WEB@@][@@ADK_MODEL@@]" in rule
    assert "@@ADK_WEB@@" in rule


def test_rules_8_and_9_still_run_after_the_answer():
    """Why the split was possible, recorded so a future fix knows the constraint.

    Persistence and logging are required to happen *after* the answer is produced, so
    there is always a tool call between the answer and the final model response. The
    instruction has to survive that rather than the ordering being changed -- a
    ``save_summary`` carrying the answer would also put the name stamp in the note.
    """
    for number in (8, 9):
        rule = _rule(number)
        assert "ALWAYS" in rule, f"rule {number} must still require its tool"


# --- the shape it must not regress into ---------------------------------------



def test_the_stamp_is_applied_after_the_model_not_during():
    """Why this is fixed in the instruction rather than by reordering the tools.

    The name stamp is added by ``after_model_callback``, so an answer that reached
    ``save_summary_to_second_brain`` *after* the stamp would write
    ``**Text Summarizer Agent**`` into the user's vault. Moving the persistence call
    earlier to close the window would open that one, so the instruction is where the
    fix belongs.
    """
    src = AGENT.read_text(encoding="utf-8")
    assert "after_model_callback" in src
    bot = AGENT.with_name("bot_name.py")
    assert not bot.exists(), (
        "if the stamp moved into its own module, update this test -- it asserts "
        "where the stamp is applied, and that fact is the reason the fix is in the "
        "instruction"
    )
