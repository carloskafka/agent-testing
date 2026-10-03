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

import pytest

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

#: Both rules that emit source lines. **Both, and that is the point.**
#:
#: The first version of this fix corrected rule 13 only, and the split then shipped
#: live on the vault path unchanged. Rule 7 is the *primary* path -- retrieve before
#: summarizing -- so a vault-sourced answer was the one most likely to end on a stub,
#: and it was the one left inviting it. Parameterised rather than duplicated so a
#: third source-emitting rule cannot be added without appearing here.
SOURCE_RULES = (7, 13)


@pytest.mark.parametrize("number", SOURCE_RULES)
def test_the_source_lines_go_in_the_answer_message(number):
    """The clause has to be a positive instruction, not a prohibition.

    "Do not send the citations separately" leaves the model to work out what
    *instead*; "in the same message as your bullets" is the shape.
    """
    rule = _rule(number)
    assert "IN THE SAME MESSAGE" in rule, f"rule {number} does not say where the lines go"


@pytest.mark.parametrize("number", SOURCE_RULES)
def test_no_rule_says_the_very_end(number):
    """The phrase that caused the split must be gone, not merely supplemented.

    Leaving it in place alongside the new clause gives the model two readings of one
    instruction, and the old one is the one it was already following.
    """
    rule = _rule(number)
    assert "at the very END of your answer" not in rule, (
        f"rule {number} still says 'at the very END of your answer', which is what a "
        "model reads as 'a final message of its own'"
    )


@pytest.mark.parametrize("number", SOURCE_RULES)
def test_the_citations_are_still_required(number):
    """The control: the clause must not have turned into "skip the sources"."""
    rule = _rule(number)
    assert "@@ADK_VAULT@@" in rule or "@@ADK_WEB@@" in rule
    assert "EXACTLY" in rule, f"rule {number} must still demand the template verbatim"


def test_no_rule_at_all_says_the_very_end():
    """The sweep, so a future rule cannot reintroduce it unnoticed.

    The tests above cover the rules that exist today. This one reads the whole block,
    which catches the case they cannot: a *new* rule emitting source lines with the
    old wording, which would ship the defect with no test failing.
    """
    offenders = re.findall(r"^(\d+)\. [^\n]*very END", _instructions(), re.M)
    assert not offenders, (
        f"rule(s) {offenders} say 'very END of your answer', which invites the "
        "citations into a message of their own"
    )


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
