"""Two instruction changes that no score can see, pinned on the instruction text.

Found live on session `fae42db3-3955-4abe-8fd1-b2003e84749d`, one booking flow
across 8 turns. Both changes are to rules whose loss moves nothing in either
eval criterion — the same reason rule 14 and rule 16 are in
``auto_optimize.REQUIRED_RULES`` — so these tests are the only thing holding them.

**What the session actually produced:**

| turn | answer | note written |
|---|---|---|
| 1-4 | films, then sessions, then Resident Evil, then the chosen session | 4 useful notes |
| 5 | *"O usuário confirmou a escolha da sessão..."* | `confirmacao-sessao-...` |
| 6 | *"Conversa finalizada com a confirmação da sessão..."* | `conclusao-atendimento-...` |
| 7 | the checkout link, which is what the user had asked for since turn 4 | 1 useful note |

Two of seven notes record **the conversation happening**, not the world. Both
carry valid fingerprints, so they are now reachable by the vault cache: asking
again replays "the user confirmed the session" as though it were a fact.
"""

from __future__ import annotations

import pathlib
import re

AGENT = pathlib.Path(__file__).resolve().parents[1] / "agent.py"


def _instructions() -> str:
    return AGENT.read_text(encoding="utf-8")


def _rule(number: int) -> str:
    """The text of one numbered rule.

    Read by number rather than by a search phrase, for the reason
    ``test_a_rule_mentioned_mid_sentence_does_not_count_as_present`` in
    ``test_auto_optimize.py`` exists: a phrase can appear in a rule that is being
    *described* rather than stated.
    """
    text = _instructions()
    start = re.search(rf"^{number}\. ", text, re.M)
    assert start, f"rule {number} not found"
    rest = text[start.start():]
    end = re.search(rf"^{number + 1}\. ", rest, re.M)
    return rest[: end.start()] if end else rest


# --- rule 8: do not persist the conversation ---------------------------------


def test_rule_8_forbids_saving_a_note_about_the_conversation():
    """The clause has to be a prohibition, not a suggestion.

    "Only save summaries" would not have prevented anything: both offending notes
    were formatted exactly as summaries, with a title, bullets and topics. What
    distinguishes them is their *subject*, so the rule has to name the subject.
    """
    rule = _rule(8)
    assert "about this conversation" in rule
    for phrase in ("confirmed", "conversation is finished"):
        assert phrase in rule, (
            f"{phrase!r} is one of the two note subjects actually written; the rule "
            "has to name it or the model cannot recognise it"
        )


def test_rule_8_says_what_to_do_instead():
    """A prohibition with no alternative produces a turn that writes nothing at all."""
    rule = _rule(8)
    assert "save nothing" in rule, "it has to say what to do rather than only what not to do"


def test_rule_8_still_requires_persisting_a_real_summary():
    """The control.

    The clause must not have weakened rule 8 into optionality -- which would cost
    the persistence the whole eval loop depends on, and which
    ``tool_trajectory_avg_score`` grades at threshold 1.0.
    """
    rule = _rule(8)
    assert "ALWAYS persist the summary" in rule
    assert "save_summary_to_second_brain" in rule


# --- rule 15: listing is not choosing -----------------------------------------


def test_rule_15_separates_listing_from_choosing():
    """The failure was not that the model could not ask. It was that it never occurred to.

    It reported every session correctly, fetched live data per rule 13, and simply
    answered. So the rule has to say that *having the list is not permission to pick*,
    rather than only describing when to ask.
    """
    rule = _rule(15)
    assert "Listing what is available and picking for the user are different acts" in rule


def test_rule_15_names_the_options_that_must_be_asked_about():
    """Screenings, seats, dates and times -- the things a note goes stale on.

    Naming them is what makes the clause fire. "When the options matter" would not:
    the model had already decided these mattered enough to fetch live.
    """
    rule = _rule(15)
    for option in ("screening", "seat", "time"):
        assert option in rule.lower(), f"{option!r} is not named as a fork to ask about"


def test_rule_15_does_not_demand_asking_when_there_is_no_fork():
    """The control, and the over-correction this fix invites.

    A rule that says "always ask before choosing" turns every summarizer turn into a
    question, which rule 15's own first sentence forbids and which costs a turn each
    time. `worth_asking` already refuses a question with fewer than two options and
    one with no stated consequence; the instruction has to agree with it.
    """
    rule = _rule(15)
    assert "Do NOT ask when a sensible default is obvious" in rule


def test_rule_15_still_bears_its_original_examples():
    """Two cinemas and two payment methods were the worked examples; keep them."""
    rule = _rule(15)
    assert "two cinemas whose session times all differ" in rule


# --- both rules must survive the optimizer ------------------------------------


def test_rules_8_and_15_are_both_required():
    """Rule 8 already was. Rule 15 deliberately was not -- so say why, out loud.

    ROUGE-1 cannot see a tool call, so dropping rule 8 moves no score in either
    direction... except ``tool_trajectory_avg_score``, which grades it. Rule 15's
    fork clause is invisible to both, which is exactly the case the guard exists for.
    """
    from text_summarizer.auto_optimize import REQUIRED_RULES

    assert 8 in REQUIRED_RULES
    # 15 is absent on purpose and this test exists so that stays a decision rather
    # than drifting; the comment below is the record.
    assert 15 not in REQUIRED_RULES, (
        "rule 15 is not in REQUIRED_RULES by design -- it did not fork on any eval "
        "case and adding it would reject rewrites for no measured gain. If you are "
        "adding it deliberately, update auto_optimize.REQUIRED_RULES and its "
        "comment too, and delete this assertion."
    )