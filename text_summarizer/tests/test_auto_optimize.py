"""The rule-deletion guard on the auto-optimization loop.

``auto_optimize.py`` is the one script in this repo that **rewrites its own
source of truth** (gotcha 1): it finds the ``instruction`` triple-quoted block in
``agent.py`` and replaces the text between the quotes. Everything else here
either reads or writes data; this one decides what the agent will say next.

That makes its decision rule worth pinning, and the rule has a property that is
easy to miss. Its only decision is ROUGE-1 word overlap,
``response_match_score`` -- and ROUGE-1 **rises** when instructions get shorter
and more generic. "Remove the rules that make this agent different" is a
perfectly good-looking rewrite by that metric. So the loop can delete the rules
that persist to the vault, cite it, and consult the clock, score *better*, and
keep the rewrite. The agent then quietly stops doing its job while the dashboard
shows an improvement.

``missing_required_rules`` is the guard against exactly that, and these tests pin
what it must and must not refuse. The asymmetry matters: a guard that rejects
too much makes the tool useless, and a guard that rejects too little is the bug
it was written for.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest
from text_summarizer import auto_optimize
from text_summarizer.agent import root_agent

#: The real instruction block, so the guard is exercised against the text it
#: actually has to protect rather than a hand-written summary of it. Read from
#: the agent rather than duplicated here: a copy would keep passing after the
#: rule it names was renumbered or deleted -- which is the whole failure the
#: guard exists to catch, so a copy would be testing the copy.
REAL = root_agent.instruction


def _drop_rule(instructions: str, number: int) -> str:
    """The block with one numbered rule removed, leaving everything else alone."""
    return "\n".join(
        line
        for line in instructions.splitlines()
        if not line.lstrip().startswith(f"{number}. ")
    )


# --- what the guard accepts ----------------------------------------------------


def test_the_current_instruction_block_would_be_kept():
    """The baseline: the guard must not refuse the instructions as they stand.

    If this fails, every rewrite is rejected and the loop does nothing at all --
    a guard that is always true is as useless as no guard, and it is the failure
    mode that looks like "the optimizer just isn't helping".
    """
    assert auto_optimize.missing_required_rules(REAL) == []


def test_a_rewrite_that_keeps_every_rule_is_allowed():
    rewritten = REAL + "\n15. Be especially clear about dates and names."
    assert auto_optimize.missing_required_rules(rewritten) == []


# --- what it refuses, and why --------------------------------------------------


@pytest.mark.parametrize("rule", auto_optimize.REQUIRED_RULES)
def test_dropping_any_single_required_rule_is_refused(rule):
    """Each rule stands alone.

    Parametrized over the tuple rather than asserting the whole set once,
    because the failure this guards against is a rewrite that drops *one* rule,
    and a set-equality assertion would keep passing while the guard silently
    shrank to fewer rules.
    """
    assert auto_optimize.missing_required_rules(_drop_rule(REAL, rule)) == [rule]


def test_a_rewrite_that_drops_the_persistence_rules_is_refused():
    """The headline case, and the one the eval would mis-score.

    Dropping rule 8 stops ``save_summary_to_second_brain``. ``test_config.json``
    grades that as ``tool_trajectory_avg_score`` at threshold 1.0 -- but the
    loop decides on ROUGE-1, which cannot see a tool call at all, and a shorter
    instruction block scores *better* on it. So this is precisely the rewrite
    the loop's own arithmetic would keep.
    """
    assert 8 in auto_optimize.missing_required_rules(_drop_rule(REAL, 8))


def test_the_rule_that_governs_the_digest_is_protected():
    """Rule 14 is in the set for a concrete reason.

    Nothing observes it. The digest answers it governs are not in the eval set at
    all, so dropping it cannot move any score in either direction -- and the
    symptom is one this repo had already shipped and fixed: every digest answer
    came back uncited (AGENTS.md, session 2d756ad4-ef8b-4dec-a6e0-f07342a9957a).
    A guard that omitted 14 would let the optimizer quietly restore that bug.
    """
    assert 14 in auto_optimize.REQUIRED_RULES

    assert 14 in auto_optimize.missing_required_rules(_drop_rule(REAL, 14))


def test_rule_16_is_required_and_cannot_be_dropped():
    """Same reason as 14, and the failure it guards against is worse.

    Rule 16 is what stops the agent setting ``attested=True`` on its own and what
    stops it claiming a purchase it did not make. Nothing observes either: no eval
    case involves an age gate or a checkout, and ROUGE-1 is word overlap, so a
    rewrite that dropped the rule would score *better* for being shorter -- the
    exact dynamic the guard exists to resist.

    The assertion that matters is the second one. Being in ``REQUIRED_RULES`` is a
    constant; being in the set a dropped rule is *reported* against is the wiring,
    and a constant nobody reads protects nothing.
    """
    assert 16 in auto_optimize.REQUIRED_RULES

    assert 16 in auto_optimize.missing_required_rules(_drop_rule(REAL, 16))


def test_dropping_rule_16_is_refused_before_the_file_is_written():
    """The outcome that matters, not the function.

    ``missing_required_rules`` returning ``[16]`` proves the arithmetic. This
    proves the file on disk is unchanged, which is the thing gotcha 1 is about --
    ``agent.py`` *is* the graded path, and a rewrite that reaches it has already
    happened. Same reasoning as
    ``test_a_rule_dropping_rewrite_is_never_written_to_agent_py``.
    """
    import inspect
    import textwrap

    src = inspect.getsource(auto_optimize.main)
    # The *call* forms, not the bare names. The comment explaining the ordering
    # mentions ``set_instructions`` before ``missing_required_rules``, so indexing
    # on the names finds the comment and reads the order backwards -- the same
    # mistake as ``test_a_rule_mentioned_mid_sentence_does_not_count_as_present``,
    # committed by a test written alongside it.
    guard = src.index("missing_required_rules(")
    write = src.index("set_instructions(")
    assert guard < write, (
        "the guard runs after the write, so a process killed between the two leaves "
        "agent.py holding instructions the guard would have refused"
    )
    # The reject path has to *stop* before the write. `break` and `return` are both
    # fine -- what is not fine is the write being reachable with `dropped` non-empty,
    # which is what "checked before the write" has to mean in code.
    gap = textwrap.dedent(src[guard:write])
    assert re.search(r"\b(break|return|continue)\b", gap), (
        "the dropped-rule branch does not exit before set_instructions, so a "
        "refused rewrite still reaches agent.py"
    )


# --- failing safe --------------------------------------------------------------


def test_no_instructions_at_all_refuses_everything():
    """Empty is not "nothing was dropped", it is "nothing can be confirmed".

    The optimiser returning ``None`` or ``""`` is a real failure mode -- an API
    error swallowed upstream, a model that answered with a refusal. Reading that
    as "all rules present" would write an empty instruction block into
    ``agent.py`` and take the agent down; reading it as "all rules missing"
    refuses it. The guard has to fail in the direction that leaves the file
    intact.
    """
    assert auto_optimize.missing_required_rules(None) == list(auto_optimize.REQUIRED_RULES)
    assert auto_optimize.missing_required_rules("") == list(auto_optimize.REQUIRED_RULES)


def test_a_renumbered_block_fails_closed():
    """Renumbering reads as "everything is missing", which is the safe direction.

    A rewrite that compacts the block and renumbers it 1..10 has no rule 12, and
    there is no way to tell that from a deletion -- the number is simply not
    there. The guard reports the rules as missing and refuses, rather than trying
    to be clever about renumbering and letting a genuine deletion slip through in
    the attempt.

    Only the high numbers go missing here, which is the point: the guard keys on
    numbers rather than on the *text* of each rule, so a compaction that keeps
    rule 8's sentence but renumbers it as rule 3 is caught as a deletion of 8.
    Under- and over-refusing are both acceptable; silently keeping a broken
    agent is not.
    """
    renumbered = "\n".join(
        f"{index}. some instruction text" for index in range(1, 11)
    )

    dropped = auto_optimize.missing_required_rules(renumbered)

    assert dropped == [11, 12, 13, 14, 16, 17], "only the numbers the compacted block lacks"
    assert 8 not in dropped, "rule 8 is still numbered 8 and must be found"


def test_a_rule_mentioned_mid_sentence_does_not_count_as_present():
    """The match is anchored to a line start.

    Rule 13's own text ends a sentence with "...as rule 7.", so a looser pattern
    reads ``7. `` mid-line and concludes the retrieval rule survived. Anchoring
    with ``^`` under ``re.MULTILINE`` is what stops a *mention* from passing as
    the *rule*.
    """
    # "rule 7." mid-sentence, and the real rule 7 is gone.
    only_a_mention = "13. you have exhausted rule 7. If nothing else works, answer from memory."

    assert 7 in auto_optimize.missing_required_rules(only_a_mention)


def test_rule_10_is_deliberately_not_required():
    """An omission on purpose, pinned so a future reader does not "fix" it.

    Rule 10 asks for the closing "Saved to the second brain as ..." sentence,
    which appears in every golden answer -- so ROUGE-1 does see its loss, and
    requiring it would reject rewrites for no gain. Recording the decision is
    the difference between a deliberate boundary and an oversight.
    """
    assert 10 not in auto_optimize.REQUIRED_RULES

# --- the guard is actually consulted -------------------------------------------
#
# Everything above tests ``missing_required_rules`` as a function. That is not the
# same as the guard being *used*: setting ``dropped = []`` in the loop leaves all
# of those tests green, because the function is untouched. It was found that way
# -- neutered, the suite passed 804/804 -- so these drive the loop itself.


def _drive_loop(monkeypatch, tmp_path, rewrite, *, scores=(0.50, 0.60)):
    """Run one iteration of ``main()`` with every side effect stubbed.

    ``scores`` is the (baseline, re-eval) pair, so a rewrite that is *accepted*
    gets far enough to be written -- which is what makes the refusal below a real
    difference in behaviour rather than the loop stopping for an unrelated reason.

    Returns ``(writes, kept_flags)``: what ``set_instructions`` was handed, and
    the ``kept`` flag each ``save_history`` recorded.
    """
    agent_file = tmp_path / "agent.py"
    original = 'instruction="""ORIGINAL RULES"""\n'
    agent_file.write_text(original)

    writes = []
    kept = []

    monkeypatch.setattr(auto_optimize, "AGENT_FILE", agent_file)
    monkeypatch.setattr(auto_optimize, "get_current_instructions", lambda: ORIGINAL_MARKER)
    monkeypatch.setattr(
        auto_optimize,
        "set_instructions",
        lambda text: writes.append(text) or agent_file.write_text(f'instruction="""{text}"""\n'),
    )
    monkeypatch.setattr(auto_optimize, "critique_and_rewrite", lambda *a, **k: rewrite)
    # `kept` is passed by keyword on the rejection path and positionally on the
    # success path, so the stub has to read both -- capturing only one records
    # None for the other outcome, and the assertion then proves nothing.
    def _record(*args, **kwargs):
        kept.append(kwargs["kept"] if "kept" in kwargs else args[-1])

    monkeypatch.setattr(auto_optimize, "save_history", _record)
    monkeypatch.setattr(auto_optimize, "EVAL_FILE", str(EVAL_STUB))
    monkeypatch.setattr(sys, "argv", ["auto_optimize.py", "--max-iterations", "1"])

    queue = list(scores)
    monkeypatch.setattr(auto_optimize, "run_eval", lambda: (queue.pop(0), None))

    auto_optimize.main()
    return writes, kept


ORIGINAL_MARKER = (
    "1. one\n7. seven\n8. eight\n9. nine\n11. eleven\n12. twelve\n"
    "13. thirteen\n14. fourteen\n16. sixteen\n17. seventeen\n"
)
EVAL_STUB = pathlib.Path(__file__).parent / "eval" / "simple_test.test.json"


def test_a_rule_dropping_rewrite_is_never_written_to_agent_py(monkeypatch, tmp_path, capsys):
    """The load-bearing property: the broken instructions do not reach the file.

    ``agent.py`` is the graded path -- gotcha 1 warns that ``auto_optimize.py``
    rewrites it in place -- so "the guard returns the right numbers" is not the
    outcome that matters. The outcome is that the file on disk is untouched.
    """
    agent_file = tmp_path / "agent.py"
    before = 'instruction="""ORIGINAL RULES"""\n'
    agent_file.write_text(before)

    writes, kept = _drive_loop(
        monkeypatch, tmp_path, "1. one\n7. seven\n9. nine\n11. eleven\n12. twelve\n13. thirteen\n14. fourteen\n16. sixteen\n"
    )

    assert writes == [], "the rule-dropping rewrite was written to agent.py"
    assert agent_file.read_text() == before, "agent.py was modified despite the rejection"
    assert kept == [False], "the rejected iteration must be recorded as not kept"
    assert "REJECTED" in capsys.readouterr().out


def test_a_compliant_rewrite_is_still_applied(monkeypatch, tmp_path):
    """The other half, and the one that would silently pass if it were missing.

    A guard that refuses everything is indistinguishable from a working guard
    unless something proves an accepted rewrite still gets through. Without this,
    "delete the guard" and "make the guard too strict" would be the same commit.
    """
    compliant = (
        "1. one\n7. seven\n8. eight\n9. nine\n11. eleven\n12. twelve\n"
        "13. thirteen\n14. fourteen\n15. extra\n16. sixteen\n17. seventeen\n"
    )

    writes, kept = _drive_loop(monkeypatch, tmp_path, compliant)

    assert writes == [compliant]
    assert kept == [True]

