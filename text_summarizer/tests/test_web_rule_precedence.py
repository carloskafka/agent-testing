"""Rule 13: when the vault is not allowed to end the question.

Rule 13 is the web tier's gate, and it is deliberately a *last resort* -- rule 7
searches the vault first, and a note on the topic normally settles the matter. The
tests here are about the two exceptions to that, and about the one requirement
that exists because a fetched URL was being thrown away.

Nothing observes any of this. No eval case asks for an exhaustive listing, and
neither eval criterion can see whether the web tier was used at all
(``response_match_score`` is ROUGE-1 over words; ``tool_trajectory_avg_score``
grades rules 8 and 9). So an edit that quietly demoted the exceptions back to
advice would move no score in either direction -- these are the only guard.
"""

from __future__ import annotations

from text_summarizer.agent import root_agent


def _rule(instruction: str, number: int) -> str:
    """One numbered rule's text, taken from the instruction as the model reads it.

    Rule bodies are single lines, so the line that starts the rule is the rule --
    which is what makes this check meaningful even if the wording is rewritten.
    """
    prefix = f"\n{number}. "
    start = instruction.find(prefix)
    assert start != -1, f"rule {number} is not in the instruction at all"
    end = instruction.find("\n", start + 1)
    return instruction[start:end]


def test_the_web_rule_s_exceptions_override_the_vault_first_rule():
    """Rule 13's two exceptions have to say they override rule 7, and say so in its
    own words rather than by implication.

    Found live on session ``f0842db4-9d42-4911-8b34-255a3731021f``: asked *"todos
    os filmes disponíveis pra hoje à noite"*, the agent searched the vault, found a
    note titled almost exactly that, stopped, and answered from it -- no web call
    at all, so not one link in the answer. The note was a legitimate rule-7 hit and
    an illegitimate rule-13 answer, because the exhaustive exception lived *inside*
    "only as a last resort" and so was never reached: the model never entered rule
    13's branch at all.

    Nothing in the eval loop sees this: no case asks for an exhaustive listing, and
    both criteria are blind to whether the web tier was used. So the precedence has
    to be asserted on the instruction text, or it is not asserted anywhere.
    """
    from text_summarizer.auto_optimize import missing_required_rules

    instruction = root_agent.instruction
    rule_13 = _rule(instruction, 13)

    assert "OVERRIDE RULE 7" in rule_13, (
        "rule 13's exceptions must state that they override the vault-first rule; "
        "otherwise a note on the topic ends the turn before the exception is read"
    )
    # Both triggers, and the staleness reason -- a note is a record of an earlier
    # fetch, so answering "tonight" from one is a wrong answer rather than a
    # cached one.
    assert "(a) EXHAUSTIVE" in rule_13
    assert "(b) LIVE" in rule_13
    assert "EXCEPTION -" not in rule_13, (
        "the old single exception is back; the exhaustive case is what failed live"
    )
    # And the links half: the second complaint on that session was an answer with
    # session times and no clickable URL anywhere.
    assert "GIVE THE READER THE LINKS" in rule_13
    assert "save_summary_to_second_brain" in rule_13, (
        "links that are not written into the note are lost to the next turn, which "
        "is how a follow-up cites a page nobody can click"
    )
    assert missing_required_rules(instruction) == []


def test_the_url_rule_is_stated_once_and_not_duplicated():
    """The "copy the URL verbatim" clause appears twice after the links change.

    Two copies of an instruction the model must follow exactly is one copy too
    many: they can drift, and a reader has no way to tell which one governs. This
    is not a hypothetical tidiness concern -- it is the shape the edit actually
    took, and it is easier to catch here than to notice in the prompt.
    """
    rule_13 = _rule(root_agent.instruction, 13)
    verbatim = "never invent, guess"
    assert rule_13.count(verbatim) == 1, (
        f"the URL-fidelity clause appears {rule_13.count(verbatim)} times in rule 13; "
        "state it once"
    )


def test_rule_thirteen_still_refuses_to_search_for_no_reason():
    """The exceptions must not have swallowed the rule they live in.

    The failure mode of the fix is over-correction: "the web tier is for anything
    live" would cost a fetch per question and make rule 7 pointless. A summarizer
    still has to summarise the text it was given without going to the web for it.
    """
    rule_13 = _rule(root_agent.instruction, 13)
    assert "SEARCH THE WEB ONLY AS A LAST RESORT" in rule_13
    assert "Never search to enrich a summary of text the user gave you" in rule_13