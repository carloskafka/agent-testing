"""The three ``quality.*`` scores, and the boundary that keeps them honest.

``_score_generation`` is the only place in the project that produces output for
Langfuse's quality dashboard, and it is a few lines of arithmetic over regexes.
There is no model call, nothing can raise, and it is never mentioned in a
failure message -- which is precisely why it was untested. A change to the
bullet regex or the overlap denominator produces no exception and no missing
metric. It just quietly changes what the charts say the agent is doing.

So the properties pinned here are the ones that decide what the numbers *mean*:

* **every value is in 0..1.** These three land in one dashboard; a fourth
  member on a different scale does not add information, it flattens the other
  three into a line near zero. This was ``AGENTS.md`` known gap 3, and the fix
  was to score the normalised bullet value and move the raw count to metadata.
* the source block is removed before anything is counted, so a cited answer is
  not scored as though it had written three times as many bullets;
* ``lexical_recall`` is recall against the *source*, so a summary that says more
  than the input did cannot exceed 1.0;
* it is called *lexical recall* and not *fidelity*, because that is what it
  measures (known gap 4). The test named
  ``test_a_contradiction_still_scores_high_because_recall_is_lexical`` is the
  reason: a summary that reverses the source's only factual claim still scores
  well, and no reader of the dashboard should be invited to conclude otherwise.

Nothing here calls Langfuse. The scores' transport is covered in
``test_agent_callback.py`` (they are skipped on a cache hit, they are prefixed
``quality.``, and the raw count goes to metadata rather than to a score); this
file is about the arithmetic they carry.
"""

from __future__ import annotations

import pytest
from text_summarizer import sources
from text_summarizer.agent import _bullet_count, _score_generation

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
#: derived from the answer so that ``lexical_recall == 1.0`` is a fact about the
#: fixture that can be checked by eye -- every number in this file is
#: ``|summary_words & source_words| / |source_words|``, and a source that was
#: generated from the answer could not fail when the regex changed.
SOURCE_TEXT = "Dogs are loyal. Dogs are social. Dogs need walks. Dogs are many breeds."

def _bullets(count: int, prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}item {n}" for n in range(count))


#: Every shape this file feeds the scorer, for the range assertion below. Chosen
#: to include both extremes (nothing at all, and far past the band) because the
#: 0..1 property is what the previous score dict violated and it violated it at
#: the *large* end -- one big number is what made the chart unreadable.
SCORED_SHAPES = [
    ("", ""),
    (SOURCE_TEXT, ANSWER_WITHOUT_SOURCES),
    (SOURCE_TEXT, ANSWER_WITH_SOURCES),
    (SOURCE_TEXT, _bullets(9)),
    (SOURCE_TEXT, _bullets(40)),
    (SOURCE_TEXT, "no bullets whatsoever"),
    (None, None),
]


# --- the three score names -----------------------------------------------------


def test_the_three_documented_scores_are_emitted():
    """Exactly three, and named as documented.

    The names are the dashboard's dimension. A rename is invisible in code --
    ``_report_scores_after_agent`` iterates whatever dict this returns and posts
    it -- and shows up only as an empty chart, which reads as "no data" rather
    than "renamed". That is why the two renames were done deliberately and
    together (``source_overlap`` -> ``lexical_recall``, ``fidelity`` ->
    ``format_and_recall``) rather than one chart at a time.
    """
    assert set(_score_generation(SOURCE_TEXT, ANSWER_WITHOUT_SOURCES)) == {
        "bullet_score",
        "lexical_recall",
        "format_and_recall",
    }


@pytest.mark.parametrize("user_text, model_text", SCORED_SHAPES)
def test_every_score_is_on_the_zero_to_one_scale(user_text, model_text):
    """The property known gap 3 was about, asserted on the shapes that broke it.

    ``bullet_count`` used to be the raw integer here -- 9.0, 40.0 -- beside two
    scores capped at 1.0 in the same dashboard. Anyone reading that chart was
    looking at one series that is mostly noise and two that are squashed against
    the axis, and the natural reading of the dip was "quality collapsed", which
    it had not.
    """
    scores = _score_generation(user_text, model_text)

    assert scores, "an empty score dict is what makes a chart look like no data"
    for name, value in scores.items():
        assert 0.0 <= value <= 1.0, f"{name} is off-scale: {value!r}"


def test_a_clean_answer_scores_full_marks():
    """The calibration point, and the only shape that scores 1.0 on all three.

    Four bullets is the middle of the ideal band, and every source word is
    reused, so ``bullet_score`` is 1.0, ``lexical_recall`` is 1.0 and their mean
    is 1.0. This is the reference the other assertions are read against: it is
    the only answer shape the metric treats as perfect.
    """
    assert _score_generation(SOURCE_TEXT, ANSWER_WITHOUT_SOURCES) == {
        "bullet_score": 1.0,
        "lexical_recall": 1.0,
        "format_and_recall": 1.0,
    }


# --- bullet_score, and the count that is not a score ---------------------------


def test_bullet_score_is_the_normalised_value_and_not_the_count():
    """Known gap 3, closed: the score is the value that was being computed and
    thrown away.

    Nine bullets scored ``9.0`` under the name ``bullet_count`` while the
    normalised 0..1 value sat in a local, used only to build a mean. The count
    was real, but it is unbounded and it is not a score, so it is now recorded as
    trace metadata (``_bullet_count``) and the score carries the number a chart
    can use. A regression that reintroduced the count under any name in this dict
    would flatten the other two series, which is exactly what this asserts.
    """
    scores = _score_generation(SOURCE_TEXT, _bullets(9))

    assert scores["bullet_score"] == 0.0, "9 bullets is 0.2 * 5 away from the ideal 4"
    assert _bullet_count(_bullets(9)) == 9, "the count still exists, as metadata"


def test_the_count_and_the_score_read_the_same_text():
    """One definition of "how many bullets", so they cannot drift.

    Two functions now exist -- ``_bullet_count`` for metadata and
    ``_score_generation`` for the score -- over the same regex on the same text.
    Splitting them was the obvious way to give the count somewhere to live, and
    the obvious failure is one of them being fixed without the other. This is
    cheap to pin and would otherwise only show up as a score that disagrees with
    its own metadata on the same turn.
    """
    for user_text, model_text in SCORED_SHAPES:
        if model_text is None:
            continue
        count = _bullet_count(model_text)
        expected = 1.0 if 3 <= count <= 5 else max(0.0, 1.0 - abs(count - 4) * 0.2)
        assert _score_generation(user_text, model_text)["bullet_score"] == pytest.approx(
            expected, abs=0.0005
        )


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 6, 20])
def test_bullet_count_counts_every_dash_bullet_and_nothing_else(count):
    """One bullet per line that starts with ``-``, at any indent.

    Asserted as an equality against the count that was asked for rather than a
    range, so a regex that stopped matching at 5 -- plausible, since 3-5 is the
    ideal band -- fails instead of quietly under-reporting long answers.
    """
    assert _bullet_count(_bullets(count)) == count


def test_a_non_dash_list_is_not_counted():
    """The metric is coupled to instruction rule 1, which mandates ``- ``.

    ``*`` and ``1.`` are perfectly good markdown and a model will use them when
    it ignores the rule. The regex accepts neither, so such an answer scores zero
    bullets and a bullet_score of 0.2. Worth knowing: a drop in this number is
    more likely to be a formatting regression than a summarisation one.
    """
    assert _bullet_count(_bullets(4, prefix="* ")) == 0
    assert _bullet_count(_bullets(4, prefix="1. ")) == 0
    # A dash mid-sentence is not a bullet either.
    assert _bullet_count("Dogs - as everyone knows - are loyal") == 0


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
    assert _bullet_count("-first\n-\n-  third") == 2
    assert _bullet_count("-first") == 0
    assert _bullet_count("-- first\n— first") == 0


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
    assert _bullet_count(ANSWER_WITH_SOURCES) == 4


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


# --- lexical_recall ------------------------------------------------------------


def test_recall_is_measured_against_the_source_so_a_longer_summary_cannot_exceed_one():
    """The denominator is the source's word set, not the summary's.

    The summary below reuses every source word *and* adds twelve that were never
    in the input, so it is longer than the text it summarises. Under recall it is
    a perfect 1.0; under a Jaccard index, or a denominator taken from the summary,
    the same answer would score 0.4 and make verbosity look like a defect -- the
    opposite of what a summariser should be rewarded for.
    """
    greedy = ANSWER_WITHOUT_SOURCES + "\n- plus twelve words nobody said aloud today"

    assert _score_generation(SOURCE_TEXT, greedy)["lexical_recall"] == 1.0


def test_recall_of_nothing_in_common_is_zero():
    assert _score_generation(SOURCE_TEXT, "- unrelated astronomy observations")["lexical_recall"] == 0.0


def test_recall_is_a_fraction_of_the_source_words():
    """Two of five source words reused scores 0.4.

    Picked because the three plausible definitions all give different numbers on
    this pair, so it pins *which* one is implemented: the source has 5 words, the
    summary has 3, and they share 2 -- a fraction of the source is ``2/5 = 0.4``,
    a ratio would be ``2/3 = 0.667`` and a Jaccard index ``2/6 = 0.333``. Only
    the first is a recall, and only the first is what a summariser should be
    rewarded for: the denominator has to be the source, or adding words the input
    never contained would be punished.
    """
    assert (
        _score_generation("alpha bravo charlie delta echo", "- alpha bravo here")["lexical_recall"]
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

    assert scores["lexical_recall"] == 0.0
    assert scores["format_and_recall"] >= 0.0


def test_a_contradiction_still_scores_high_because_recall_is_lexical():
    """Known gap 4, and the test that *names* it.

    There is no faithfulness check here. The word sets are built with ``\\b\\w+\\b``
    and no stopword list, so ``the`` and ``is`` are content words. The three
    bullets below reuse two of the source's three words, one of which is a
    stopword, and one of the two content words is *contradicted*: the source says
    the deployment failed, the answer says it is fine. It still scores 0.83.

    The number is high because recall is lexical. The score is now called
    ``format_and_recall`` precisely so that this fact is legible from the
    dashboard rather than being something a reader has to know; the earlier name
    ``fidelity`` invited exactly the wrong conclusion here, and this assertion is
    what a real groundedness check would have to replace.

    Replacing it, not renaming it, is the actual fix. Kept as-is on purpose: a
    groundedness check needs a model call per turn, which is a different budget
    decision from this one.
    """
    scores = _score_generation(
        "the deployment failed",
        "- the deployment is fine\n- it is a fine day\n- nothing else matters here",
    )

    assert scores["lexical_recall"] == pytest.approx(2 / 3, abs=0.001)
    assert scores["format_and_recall"] == pytest.approx(0.833, abs=0.001), (
        "high, for an answer that reverses the source's only factual claim"
    )


def test_word_matching_is_case_insensitive():
    """The user's text is pasted in any case; the summary is written in sentence case."""
    assert _score_generation("Dogs Are Loyal", "- dogs are loyal")["lexical_recall"] == 1.0


# --- format_and_recall, and the shapes of the inputs ----------------------------


def test_format_and_recall_is_the_mean_of_the_bullet_score_and_the_recall():
    """The composition, so the third number is not a third mystery.

    Off-band bullets drag it down even on a perfect copy, and that is the
    intent: the metric encodes instruction rule 4 (3-5 bullets) alongside
    content preservation. Two bullets is ``bullet_score`` 0.6 -- one step of 0.2
    per bullet away from the ideal four -- so the mean tops out at 0.8 for an
    exact reproduction of the source. An answer can be perfect and still be
    marked down for its shape, and the arithmetic is what makes that legible.

    The name is the point: it is the mean of a formatting heuristic and a lexical
    overlap, and it says so. ``fidelity`` said neither.
    """
    scores = _score_generation(
        "alpha bravo charlie", "- alpha bravo charlie\n- alpha bravo charlie"
    )

    assert _bullet_count("- alpha bravo charlie\n- alpha bravo charlie") == 2
    assert scores["bullet_score"] == 0.6
    assert scores["lexical_recall"] == 1.0
    assert scores["format_and_recall"] == pytest.approx(0.8, abs=0.001)


def test_the_mean_is_never_above_one_even_with_perfect_parts():
    """``min(1.0, ...)`` is load-bearing.

    The mean of two numbers that are each capped at 1.0 cannot exceed 1.0, so the
    clamp looks redundant -- until a future change makes ``bullet_score`` or
    ``lexical_recall`` exceed 1 on its own, where the clamp is the only thing
    keeping the value on the scale the dashboard assumes.
    """
    scores = _score_generation("alpha bravo", "- alpha bravo\n- alpha bravo\n- alpha bravo")

    assert scores["format_and_recall"] == 1.0


def test_empty_inputs_score_zero_recall_but_not_zero_format():
    """The floor is 0.1, not 0.0, and that is worth stating out loud.

    ``bullet_score`` decays by 0.2 per bullet away from four, floored at 0 -- so
    zero bullets is ``1.0 - 4 * 0.2 = 0.2``, not 0. Averaged with a recall of
    0.0 that leaves ``format_and_recall`` at 0.1 for a turn that produced
    *nothing at all*.

    Reached in production whenever the turn died before its answer, or whenever
    ``_last_session_user_text`` found no user event. A dead turn is supposed to
    be visible in the dashboard and this is the one number that does not show it;
    the metadata count is 0 and says so plainly. Not fixed here -- the scoring
    callback runs after every agent invocation and reshaping a metric is a change
    to the graded path -- but pinned, so that whoever reads the dashboard knows
    the floor.
    """
    for user_text, model_text in (("", ""), (None, None), (SOURCE_TEXT, None), ("", "text")):
        scores = _score_generation(user_text, model_text)
        assert scores["lexical_recall"] == 0.0
        assert scores["bullet_score"] == 0.2
        assert scores["format_and_recall"] == pytest.approx(0.1, abs=0.001)
        assert _bullet_count(model_text) == 0


def test_a_missing_answer_does_not_raise_out_of_the_callback():
    """Both arguments default to ``""`` and both are reachable as ``None``.

    ``_report_scores_after_agent`` runs after *every* agent invocation and passes
    whatever the session scan produced, so both halves can be empty on a turn
    that died early. Raising here would break the turn at its last step -- after
    the answer was produced and the note already written.
    """
    assert _score_generation() == {
        "bullet_score": 0.2,
        "lexical_recall": 0.0,
        "format_and_recall": 0.1,
    }


def test_every_score_is_rounded_to_three_places():
    """These go straight into Langfuse as floats.

    A third is not representable in binary, so an unrounded ``1/3`` is stored and
    displayed as ``0.3333333333333333`` -- in a column of 0.5s and 1.0s, on the
    latency-like axis these scores are charted against.
    """
    scores = _score_generation("alpha bravo charlie", "- alpha\n- bravo\n- unrelated")

    assert scores["lexical_recall"] == 0.667
    assert scores["format_and_recall"] == 0.833
    for name, value in scores.items():
        assert round(value, 3) == value, f"{name} is not rounded: {value!r}"