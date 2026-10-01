"""Unit tests for ``eval_scoring`` -- the eval-set aware ROUGE-1 scorer.

This module posts ``response_match_score`` to Langfuse, and the whole point of it
is that the number it posts is **protocol-identical** to the one ``adk eval``
reports for the same case. AGENTS.md devotes a section to that, and until now
nothing here executed it, so a drift between the two scorers -- or a silent
degradation when ``google-adk[eval]`` is missing -- would have shown up as a
plausible-looking number and nothing else.

PLAN.md §2.2 lists the whole module as "real behaviour with no test". The
properties pinned below are the ones that would fail *silently*:

* the prompt lookup normalises like the eval loop does, so a live prompt matches
  its golden answer despite whitespace and case;
* an unmatched prompt returns ``None`` and the caller skips the score, rather
  than scoring against something arbitrary;
* a missing ``rouge_score`` degrades to ``None`` instead of raising into a turn;
* the f-measure equals ADK's own, computed by calling ADK's own helper.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from text_summarizer import eval_scoring
from text_summarizer.eval_scoring import (
    _golden_response,
    _load_eval_file,
    _normalize,
    _user_prompt,
    load_eval_cases,
    response_match_for_agent,
    response_match_score,
    rouge1_fmeasure,
)


@pytest.fixture(autouse=True)
def _isolate_the_eval_cache():
    """Clear ``load_eval_cases``'s lru_cache around every test in this module.

    It is decorated ``@lru_cache(maxsize=1)`` over a module-global directory, so a
    test that repoints ``_EVAL_DIR`` at a tmp_path leaves the *cached* answer in
    place after monkeypatch undoes the variable -- and every later test then reads
    the tmp result and concludes the shipped eval set is empty. Clearing on the
    way out as well as on the way in is what makes the ordering irrelevant.
    """
    load_eval_cases.cache_clear()
    yield
    load_eval_cases.cache_clear()


def _prompt_starting_with(prefix: str) -> str:
    """A real prompt from the shipped eval set, selected by prefix.

    Read out of the file rather than pasted in, for two reasons. A pasted prompt
    is brittle: one word of drift in the eval set and the lookup returns ``None``,
    at which point every assertion built on it silently degrades into testing the
    no-match path and still passes. And a prompt long enough to be realistic is
    long enough to be wrong by hand.
    """
    for prompt in load_eval_cases():
        if prompt.startswith(prefix):
            return prompt
    raise AssertionError(f"no shipped eval case starts with {prefix!r}")


def _golden_for(prefix: str) -> str:
    """The golden answer paired with :func:`_prompt_starting_with`."""
    return load_eval_cases()[_prompt_starting_with(prefix)]


WORKING_DOGS_PREFIX = "summarize the following text: working dogs"

# --- normalisation ------------------------------------------------------------


def test_normalize_collapses_whitespace_and_case():
    """Prompt matching must be insensitive to both, or a live turn never matches."""
    assert _normalize("Summarize:  Dogs\nare   Social ") == "summarize: dogs are social"


def test_normalize_of_nothing_is_empty_not_a_crash():
    for value in ("", None):
        assert _normalize(value) == ""


# --- reading an eval file -----------------------------------------------------


def _case(prompt: str, golden: str) -> dict:
    return {
        "eval_id": "x",
        "conversation": [
            {
                "user_content": {"parts": [{"text": prompt}]},
                "final_response": {"parts": [{"text": golden}]},
            }
        ],
    }


def test_user_prompt_joins_every_text_part():
    invocation = {"user_content": {"parts": [{"text": "a"}, {"text": "b"}]}}
    assert _user_prompt(invocation) == "ab"


def test_golden_response_joins_parts_with_a_newline():
    invocation = {"final_response": {"parts": [{"text": "a"}, {"text": "b"}]}}
    assert _golden_response(invocation) == "a\nb"


@pytest.mark.parametrize(
    "invocation",
    [
        {},
        {"user_content": None},
        {"user_content": {"parts": None}},
        {"user_content": {"parts": [{"no_text": 1}]}},
    ],
)
def test_missing_or_malformed_content_reads_as_empty(invocation):
    """A malformed eval file must not take the scoring down mid-turn."""
    assert _user_prompt(invocation) == ""
    assert _golden_response(invocation) == ""


def test_load_eval_file_extracts_pairs_and_skips_incomplete_cases(tmp_path):
    path = tmp_path / "cases.evalset.json"
    path.write_text(
        json.dumps(
            {
                "eval_cases": [
                    _case("Prompt One", "Golden one"),
                    _case("", "no prompt, dropped"),
                    _case("no golden, dropped", ""),
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _load_eval_file(str(path)) == [("prompt one", "Golden one")]


def test_load_eval_cases_reads_both_file_spellings(tmp_path, monkeypatch):
    """``.test.json`` is a single eval file and ``.evalset.json`` a set; both count."""
    (tmp_path / "a.test.json").write_text(
        json.dumps({"eval_cases": [_case("From Single", "g1")]}),
        encoding="utf-8",
    )
    (tmp_path / "b.evalset.json").write_text(
        json.dumps({"eval_cases": [_case("From Set", "g2")]}),
        encoding="utf-8",
    )
    (tmp_path / "ignored.md").write_text("not an eval file", encoding="utf-8")
    monkeypatch.setattr(eval_scoring, "_EVAL_DIR", str(tmp_path))
    load_eval_cases.cache_clear()

    assert load_eval_cases() == {"from single": "g1", "from set": "g2"}


def test_load_eval_cases_is_empty_when_the_directory_is_gone(monkeypatch):
    monkeypatch.setattr(eval_scoring, "_EVAL_DIR", "/nonexistent/eval/dir")
    load_eval_cases.cache_clear()
    assert load_eval_cases() == {}


# --- the prompt -> golden lookup ----------------------------------------------


def test_a_live_prompt_matches_its_golden_answer_despite_formatting():
    """The live prompt arrives however the user typed it; the golden is fixed."""
    real = _prompt_starting_with(WORKING_DOGS_PREFIX)
    # Same words, mangled the way a live prompt arrives: padded, ragged, shouted.
    mangled = "  " + re.sub(r" ", "   ", real).upper() + "  "

    assert real != mangled, "the mangling must actually differ"
    assert response_match_score(mangled) == response_match_score(real)


def test_an_unmatched_prompt_is_none_and_never_a_default():
    for value in ("", None, "a question that is not in the eval set"):
        assert response_match_score(value) is None


# --- the score itself ---------------------------------------------------------


def test_the_fmeasure_is_adks_own_number():
    """Not a re-implementation: the module calls ADK's helper, and this proves it."""
    golden = _golden_for(WORKING_DOGS_PREFIX)
    assert golden

    model_text = golden
    ours = rouge1_fmeasure(model_text, golden)
    theirs = eval_scoring._calculate_rouge_1_scores(model_text, golden).fmeasure

    assert ours == pytest.approx(float(theirs), abs=1e-9)
    assert ours == pytest.approx(1.0, abs=1e-9), "identical text scores 1.0"


def test_a_partial_answer_scores_between_zero_and_one():
    golden = _golden_for(WORKING_DOGS_PREFIX)
    score = rouge1_fmeasure("- Dogs are good.", golden)
    assert 0.0 < score < 1.0


@pytest.mark.parametrize(
    ("model_text", "golden"),
    [("", "g"), ("m", ""), (None, "g"), ("m", None)],
)
def test_an_empty_side_yields_none_rather_than_zero(model_text, golden):
    """Zero would be a *finding*; None means 'not scorable', and the two differ."""
    assert rouge1_fmeasure(model_text, golden) is None


def test_a_missing_rouge_backend_degrades_to_none(monkeypatch):
    """No ``google-adk[eval]`` must not raise into a live turn (AGENTS.md gotcha 6)."""
    monkeypatch.setattr(eval_scoring, "_calculate_rouge_1_scores", None)
    assert rouge1_fmeasure("anything", "anything") is None


def test_a_raising_backend_degrades_to_none(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("rouge exploded")

    monkeypatch.setattr(eval_scoring, "_calculate_rouge_1_scores", boom)
    assert rouge1_fmeasure("anything", "anything") is None


# --- the whole pipeline -------------------------------------------------------


def test_the_pipeline_scores_a_matching_prompt():
    prompt = _prompt_starting_with(WORKING_DOGS_PREFIX)
    assert response_match_for_agent(prompt, "irrelevant") == pytest.approx(0.0, abs=1e-9)


def test_the_pipeline_returns_none_for_an_unknown_prompt():
    assert response_match_for_agent("never seen this prompt", "- a") is None


def test_the_name_stamp_does_not_count_against_response_match_score():
    """Presentation must not move the number the eval loop exists to measure.

    The goldens are the bullets plus the "Saved to the second brain" line. ROUGE-1
    is word overlap, so a stamp in the model's output that is absent from the golden
    costs precision: measured at 1.00 without the strip and 0.94 with it, on an
    otherwise *exact* match. That still clears the 0.5 threshold, so no run would
    fail -- it would just move the baseline every instruction edit is compared
    against, which is the failure mode gotcha 4 warns about.
    """
    prompt = _prompt_starting_with(WORKING_DOGS_PREFIX)
    golden = response_match_score(prompt)
    assert golden is not None

    bare = response_match_for_agent(prompt, golden)
    stamped = response_match_for_agent(prompt, f"**Text Summarizer Agent**\n\n{golden}")
    # Not 1.0 even for the bare golden: ROUGE-1 is word overlap against a golden
    # the model never matches exactly. What matters is that the stamp changes
    # nothing -- an exact-equality comparison against the un-stamped score.
    assert stamped == pytest.approx(bare, abs=1e-9)


def test_the_sources_block_does_not_count_against_response_match_score():
    """Same reasoning for the Sources block, which is also absent from the goldens."""
    prompt = _prompt_starting_with(WORKING_DOGS_PREFIX)
    golden = response_match_score(prompt)
    assert golden is not None
    with_sources = golden + (
        "\n\n**Sources**\n"
        "- [obsidian][ck][gemini-3.5-flash-lite][[2026-09-25 - dogs]]: related\n"
    )
    assert response_match_for_agent(prompt, with_sources) == pytest.approx(
        response_match_for_agent(prompt, golden), abs=1e-9
    )


def test_the_real_eval_set_on_disk_is_loadable():
    """The shipped eval set must actually parse, or every score silently vanishes."""
    load_eval_cases.cache_clear()
    cases = load_eval_cases()
    assert cases, "tests/eval/ should contain at least one case"
    assert any("Sources" in golden for golden in cases.values()), (
        "the eval set is expected to include a **Sources** case"
    )


def test_the_eval_directory_constant_points_at_the_shipped_set():
    assert os.path.isdir(eval_scoring._EVAL_DIR)
    assert os.path.isfile(os.path.join(eval_scoring._EVAL_DIR, "test_config.json"))
