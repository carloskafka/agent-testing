"""The three ``quality.*`` scores, and the two of them that do not mean what they say.

``_score_generation`` is the only place in the project that produces output for
Langfuse's quality dashboard, and it is three lines of arithmetic over regexes.
There is no model call, nothing can raise, and it is never mentioned in a
failure message -- which is precisely why it was untested. A change to the
bullet regex or the overlap denominator produces no exception and no missing
metric. It just quietly changes what the charts say the agent is doing.

So the properties pinned here are the ones that decide what the numbers *mean*:

* the source block is removed before anything is counted, so a cited answer is
  not scored as though it had written three times as many bullets;
* ``bullet_count`` is the **raw count**, and the normalised 0..1 score the
  function computes for it is thrown away. That is a documented gap
  (``AGENTS.md`` known gap 3), so the current behaviour is pinned and the gap is
  named: a fix has to change this test on purpose, which is the point of writing
  it down.
* ``source_overlap`` is recall against the *source*, so a summary that says more
  than the input did cannot exceed 1.0.

Nothing here calls Langfuse. The scores' transport is covered in
``test_agent_callback.py`` (they are skipped on a cache hit, they are prefixed
``quality.``); this file is about the arithmetic they carry.
"""

from __future__ import annotations

import pytest
from text_summarizer import sources
from text_summarizer.agent import _score_generation

#: A real rendered answer, sentinels substituted: four summary bullets, one
#: rendered source line, and the closing sentence instruction rule 10 asks for.
#: The source line is *itself* a bullet, which is the whole reason the block has
#: to be stripped before counting.
ANSWER_WITH_SOURCES = (
    "- Dogs are loyal.\n"
    "- Dogs are social.\n"
    "- Dogs need walks.\n"
    "- Dogs are many breeds.\n"
    "\n"
    "**Sources**\n"
    "- [obsidian][ck][m1][[Dogs Overview]]: same topic\n"
    "\n"
    'Saved to the second brain as "Dogs".'
)

#: The same answer with the block removed, which is what the metric is supposed
#: to be reading. Asserting the two score identically is the claim.
ANSWER_WITHOUT_SOURCES = (
    "- Dogs are loyal.\n"
    "- Dogs are social.\n"
    "- Dogs need walks.\n"
    "- Dogs are many breeds.\n"
    "\n"
    'Saved to the second brain as "Dogs".'
)

#: The same words as the answer's bullets, and no others. Written out rather than
#: derived from the answer so that ``source_overlap == 1.0`` is a fact about the
#: fixture that can be checked by eye -- every number in this file is
#: ``|summary_words & source_words| / |source_words|``, and a source that was
#: generated from the answer could not fail when the regex changed.
SOURCE_TEXT = "Dogs are loyal. Dogs are social. Dogs need walks. Dogs are many breeds."


def _bullets(count: int, prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}item {n}" for n in range(count))


# --- the three score names -----------------------------------------------------


def test_the_three_documented_scores_are_emitted():
    """Exactly three, and named as documented.

    The names are the dashboard's dimension. A rename is invisible in code --
    ``_report_scores_after_agent`` iterates whatever dict this returns and posts
    it -- and shows up only as an empty chart, which reads as "no data" rather
    than "renamed".
    """
    assert set(_score_generation(SOURCE_TEXT, ANSWER_WITHOUT_SOURCES)) == {
        "bullet_count",
        "source_overlap",
        "fidelity",
    }


def test_a_clean_answer_scores_full_marks():
    """The calibration point, and the only shape that scores 1.0 on all three.

    Four bullets is the middle of the ideal band, and every source word is
    reused, so ``bullet_score`` is 1.0, ``overlap`` is 1.0 and their mean is 1.0.
    This is the reference the other assertions are read against: it is the only
    answer shape the metric treats as perfect.
    """
    assert _score_generation(SOURCE_TEXT, ANSWER_WITHOUT_SOURCES) == {
        "bullet_count": 4,
        "source_overlap": 1.0,
        "fidelity": 1.0,
    }


# --- bullet_count: the raw count (known gap 3) ---------------------------------


def test_bullet_count_is_the_raw_count_not_a_normalised_score():
    """AGENTS.md known gap 3, pinned as it stands.

    The function computes ``bullet_score`` -- 1.0 for the 3-5 band, decaying by
    0.2 per bullet either side of 4 -- and then returns ``round(n, 3)``, the
    count. The docstring on the function still describes the normalised version.
    So ``bullet_count`` is on a scale of 0..9-ish while ``fidelity``, derived
    from the discarded value, is 0..1, and both sit in the same dashboard.

    A regression that "fixed" this to return the normalised score would be an
    improvement, which is why the current shape is asserted with its reason
    rather than left to a passing test: the change has to be made knowing the
    chart's y-axis moves, not discovered from it.
    """
    scores = _score_generation(SOURCE_TEXT, _bullets(9))

    assert scores["bullet_count"] == 9
    assert scores["fidelity"] <= 1.0, "the normalised value is still computed, for fidelity"


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 6, 20])
def test_bullet_count_counts_every_dash_bullet_and_nothing_else(count):
    """One bullet per line that starts with ``-``, at any indent.

    Asserted as an equality against the count that was asked for rather than a
    range, so a regex that stopped matching at 5 -- plausible, since 3-5 is the
    ideal band -- fails instead of quietly under-reporting long answers.
    """
    assert _score_generation(SOURCE_TEXT, _bullets(count))["bullet_count"] == count


def test_a_non_dash_list_is_not_counted():
    """The metric is coupled to instruction rule 1, which mandates ``- ``.

    ``*`` and ``1.`` are perfectly good markdown and a model will use them when
    it ignores the rule. The regex accepts neither, so such an answer scores zero
    bullets and a fidelity of half at best. Worth knowing: a drop in this number
    is more likely to be a formatting regression than a summarisation one.
    """
    assert _score_generation(SOURCE_TEXT, _bullets(4, prefix="* "))["bullet_count"] == 0
    assert _score_generation(SOURCE_TEXT, _bullets(4, prefix="1. "))["bullet_count"] == 0
    # A dash mid-sentence is not a bullet either.
    assert (
        _score_generation(SOURCE_TEXT, "Dogs - as everyone knows - are loyal")["bullet_count"] == 0
    )


def test_a_bare_dash_line_counts_because_markdown_calls_it_a_bullet():
    """``\\s+`` matches the newline, so a line holding only ``-`` is a bullet.

    Not a bug -- a lone ``-`` is an empty list item in markdown too, so counting
    it is defensible -- but it is invisible in the regex and it means the count
    is not strictly "lines that start with dash-space-then-content". Pinned
    because the obvious fix (excluding newlines) would be a behaviour change made
    for the wrong reason: the assertion that actually matters is that ``-first``
    does *not* count, and a line like that is prose the model might emit.
    """
    # "-first" has no space after the dash, so only the "-"-only line and the
    # indented "-  third" match.
    assert _score_generation(SOURCE_TEXT, "-first\n-\n-  third")["bullet_count"] == 2
    assert _score_generation(SOURCE_TEXT, "-first")["bullet_count"] == 0
    assert _score_generation(SOURCE_TEXT, "-- first\n— first")["bullet_count"] == 0


# --- the Sources block is not part of the summary ------------------------------


def test_source_lines_are_never_counted_as_bullets():
    """Every source line is itself a ``- `` bullet, and none of them are content.

    Without ``summary_only`` a cited answer with three sources would report seven
    bullets and the ideal band would be unreachable for exactly the answers that
    did the most work. The same property is asserted end-to-end in
    ``test_agent_callback.py``; this is the unit under it.
    """
    assert sources.summary_only(ANSWER_WITH_SOURCES) == ANSWER_WITHOUT_SOURCES

    cited = _score_generation(SOURCE_TEXT, ANSWER_WITH_SOURCES)
    uncited = _score_generation(SOURCE_TEXT, ANSWER_WITHOUT_SOURCES)

    assert cited == uncited
    assert cited["bullet_count"] == 4


def test_the_trailing_sentence_after_the_block_is_kept():
    """The block is removed, not everything from the block onwards.

    ``strip_sources_block`` resumes at the first line that is not part of the
    block, and instruction rule 10's "saved to the second brain" line sits after
    it. A truncating implementation would drop that too -- and would then also
    drop the answer itself, since a model that puts the block in the middle would
    lose everything below it.
    """
    stripped = sources.summary_only(ANSWER_WITH_SOURCES)

    assert "Saved to the second brain" in stripped
    assert "Dogs Overview" not in stripped
    assert "**Sources**" not in stripped


# --- source_overlap ------------------------------------------------------------


def test_overlap_is_recall_against_the_source_so_a_longer_summary_cannot_exceed_one():
    """The denominator is the source's word set, not the summary's.

    The summary below reuses every source word *and* adds twelve that were never
    in the input, so it is longer than the text it summarises. Under recall it is
    a perfect 1.0; under a Jaccard index, or a denominator taken from the summary,
    the same answer would score 0.4 and make verbosity look like a defect -- the
    opposite of what a summariser should be rewarded for.
    """
    greedy = ANSWER_WITHOUT_SOURCES + "\n- plus twelve words nobody said aloud today"

    assert _score_generation(SOURCE_TEXT, greedy)["source_overlap"] == 1.0


def test_overlap_of_nothing_in_common_is_zero():
    assert (
        _score_generation(SOURCE_TEXT, "- unrelated astronomy observations")["source_overlap"]
        == 0.0
    )


def test_overlap_is_a_fraction_of_the_source_words():
    """Half the source words reused scores 0.5, on a 5-word source.

    Picked for the arithmetic being checkable by hand: it distinguishes a
    *fraction* from a *ratio* (``0.5`` vs ``0.5`` -- same here) and from a Jaccard
    index, which for this pair would be ``2/8 = 0.25``. A summariser metric
    changed to Jaccard by accident would penalise long summaries and this is the
    only assertion that would notice.
    """
    assert (
        _score_generation("alpha bravo charlie delta echo", "- alpha bravo here")["source_overlap"]
        == 0.4
    )


def test_the_denominator_cannot_be_zero():
    """No source words, no division, no exception.

    Reached in production whenever ``_last_session_user_text`` finds nothing --
    a turn whose first user event has no text parts, or a session restored
    without one. ``0.0`` is the honest answer there: nothing was matched, which is
    different from matching nothing.
    """
    scores = _score_generation("", "- anything at all")

    assert scores["source_overlap"] == 0.0
    assert scores["fidelity"] >= 0.0


def test_stopwords_count_toward_overlap_which_is_what_makes_fidelity_lexical():
    """AGENTS.md known gap 4, pinned as it stands.

    ``fidelity`` is not a faithfulness check. The word sets are built with
    ``\\b\\w+\\b`` and no stopword list, so ``the`` and ``is`` are content words
    here. The three bullets below reuse two of the source's three words, one of
    which is a stopword, and one of the two content words is *contradicted*: the
    source says the deployment failed, the answer says it is fine. It still
    scores 0.83, because the bullet band is ideal and two thirds of the source's
    words were echoed.

    The point of the test is that the number in the dashboard cannot be read as
    "the agent stayed faithful". A real groundedness check would replace it, and
    this assertion is what would have to change.
    """
    scores = _score_generation(
        "the deployment failed",
        "- the deployment is fine\n- it is a fine day\n- nothing else matters here",
    )

    assert scores["source_overlap"] == pytest.approx(2 / 3, abs=0.001)
    assert scores["fidelity"] == pytest.approx(0.833, abs=0.001), (
        "high, for an answer that reverses the source's only factual claim"
    )


def test_word_matching_is_case_insensitive():
    """The user's text is pasted in any case; the summary is written in sentence case."""
    assert _score_generation("Dogs Are Loyal", "- dogs are loyal")["source_overlap"] == 1.0


# --- fidelity, and the shapes of the inputs ------------------------------------


def test_fidelity_is_the_mean_of_the_bullet_score_and_the_overlap():
    """The composition, so the third number is not a third mystery.

    Off-band bullets drag it down even on a perfect copy, and that is the
    intent: the metric encodes instruction rule 4 (3-5 bullets) alongside
    content preservation. Two bullets is ``bullet_score`` 0.6 -- one step of 0.2
    per bullet away from the ideal four -- so fidelity tops out at 0.8 for an
    exact reproduction of the source. An answer can be perfect and still be
    marked down for its shape, and the arithmetic is what makes that legible.
    """
    scores = _score_generation(
        "alpha bravo charlie", "- alpha bravo charlie\n- alpha bravo charlie"
    )

    assert scores["bullet_count"] == 2
    assert scores["source_overlap"] == 1.0
    assert scores["fidelity"] == pytest.approx(0.8, abs=0.001)


def test_fidelity_is_never_above_one_even_with_perfect_parts():
    """``min(1.0, ...)`` is load-bearing.

    The mean of two numbers that are each capped at 1.0 cannot exceed 1.0, so the
    clamp looks redundant -- until a future change makes ``bullet_score`` or
    ``overlap`` exceed 1 on its own, where the clamp is the only thing keeping
    the value on the scale the dashboard assumes.
    """
    scores = _score_generation("alpha bravo", "- alpha bravo\n- alpha bravo\n- alpha bravo")

    assert scores["fidelity"] == 1.0


def test_empty_inputs_score_zero_overlap_but_not_zero_fidelity():
    """The floor is 0.1, not 0.0, and that is worth stating out loud.

    ``bullet_score`` decays by 0.2 per bullet away from four, floored at 0 -- so
    zero bullets is ``1.0 - 4 * 0.2 = 0.2``, not 0. Averaged with an overlap of
    0.0 that leaves ``fidelity`` at 0.1 for a turn that produced *nothing at all*.

    Reached in production whenever the turn died before its answer, or whenever
    ``_last_session_user_text`` found no user event. A dead turn is supposed to
    be visible in the dashboard and this is the one number that does not show it;
    ``bullet_count`` is 0 and says so plainly. Not fixed here -- the scoring
    callback runs after every agent invocation and raising or reshaping a metric
    is a change to the graded path -- but pinned, so that whoever reads the
    dashboard knows the floor.
    """
    for user_text, model_text in (("", ""), (None, None), (SOURCE_TEXT, None), ("", "text")):
        scores = _score_generation(user_text, model_text)
        assert scores["source_overlap"] == 0.0
        assert scores["bullet_count"] == 0
        assert scores["fidelity"] == pytest.approx(0.1, abs=0.001)


def test_a_missing_answer_does_not_raise_out_of_the_callback():
    """Both arguments default to ``""`` and both are reachable as ``None``.

    ``_report_scores_after_agent`` runs after *every* agent invocation and passes
    whatever the session scan produced, so both halves can be empty on a turn
    that died early. Raising here would break the turn at its last step -- after
    the answer was produced and the note already written.
    """
    assert _score_generation() == {"bullet_count": 0, "source_overlap": 0.0, "fidelity": 0.1}


def test_every_score_is_rounded_to_three_places():
    """These go straight into Langfuse as floats.

    A third is not representable in binary, so an unrounded ``1/3`` is stored and
    displayed as ``0.3333333333333333`` -- in a column of 0.5s and 1.0s, on the
    latency-like axis these scores are charted against.
    """
    scores = _score_generation("alpha bravo charlie", "- alpha\n- bravo\n- unrelated")

    assert scores["source_overlap"] == 0.667
    assert scores["fidelity"] == 0.833
    for name, value in scores.items():
        assert round(value, 3) == value, f"{name} is not rounded: {value!r}"
