"""A wrong-typed tool argument must not end the turn.

Found live, not reasoned about. Session `f89fcd02-3961-43e9-91d4-7296a9e397cf`,
asked *"Liste os filmes para hoje em osasco"*, served by the OpenRouter fallback
`liquid/lfm-2.5-2.6b:free`:

    event 14  save_summary_to_second_brain(title=..., summary_content=..., themes="Filmes, Osasco, ...")
    event 15  {"error": "...mandatory input parameters are not present: topics ... you could retry"}
    event 16  save_summary_to_second_brain(title=..., summary_content=..., topics=["Filmes", "Osasco", "Cinema", "Programação"])
    event 17  AttributeError: 'list' object has no attribute 'split'      <- the turn ends here

Event 15 is the behaviour working: ADK reports a **missing** argument as data, and
the model retries. What it does not do is inspect an argument that is present and
the wrong shape, so the retry -- which was correct, and which fixed the *name* --
died on the *type* instead. No note was written, no conversation logged, and the
user got no answer.

That is the whole shape of this bug: the recoverable failure and the fatal one
differ only in whether the value was there, not in whether it was right.
"""

from __future__ import annotations

import json

from text_summarizer import second_brain as sb

# The arguments exactly as the persisted event recorded them.
REAL_TOPICS = ["Filmes", "Osasco", "Cinema", "Programação"]
REAL_TITLE = "Filmes em Osasco para Hoje 2026-10-03"
REAL_BODY = "- Verity\n- Minha Melhor Amiga\n- Digger"


# --- the exact failure -------------------------------------------------------


def test_the_real_argument_from_that_session_no_longer_raises():
    """The replay. A list of topics is what a model sends, and it must work."""
    kept, dropped = sb._split_topics(REAL_TOPICS)
    assert dropped == []
    assert kept == REAL_TOPICS, (
        "a list is what 'topics' means; joining it must reproduce the same topics "
        "the string form would have produced"
    )


def test_a_list_and_the_equivalent_string_agree_exactly():
    """The coercion is invisible to the vault.

    This is the property that makes accepting a list legitimate rather than a
    fudge: the note written must be byte-identical either way, because a silently
    different note is worse than a rejected call.
    """
    from_list, _ = sb._split_topics(REAL_TOPICS)
    from_string, _ = sb._split_topics(", ".join(REAL_TOPICS))
    assert from_list == from_string


def test_the_turn_survives_a_list_topics_end_to_end(tmp_path, monkeypatch):
    """Not just the helper -- the tool, with no vault reachable.

    The point is that it *returns*. A unit test on `_split_topics` would have passed
    against the old code if the exception were caught one frame up, and the thing
    that killed the turn was the exception escaping the tool.
    """
    monkeypatch.setattr(sb, "VAULT_ROOT", str(tmp_path / "vault"))
    out = sb.save_summary_to_second_brain(
        title=REAL_TITLE, summary_content=REAL_BODY, topics=REAL_TOPICS
    )
    # Whatever it returns, it must be a string the model can read -- never a raise.
    assert isinstance(out, str)
    assert not out.lstrip().startswith("{"), (
        "a refusal is data, not a note: an empty vault plus a valid list should "
        "have written something, so getting JSON here means it bailed"
    )


# --- errors are data, not exceptions -----------------------------------------


def test_a_list_title_is_refused_as_data_not_raised():
    """A list is not what a title means, and `str()` would put it in a filename."""
    out = sb.save_summary_to_second_brain(
        title=["a", "b"], summary_content=REAL_BODY, topics="x"
    )
    assert json.loads(out)["error"].startswith("title must be a string")
    assert "list" in out, "the error has to name the type it actually received"


def test_a_list_body_is_refused_rather_than_stringified():
    """The mirror, and the one that would have produced a visible garbage note."""
    out = sb.save_summary_to_second_brain(
        title=REAL_TITLE, summary_content=["a", "b"], topics="x"
    )
    assert "summary_content must be a string" in out


def test_the_refusal_names_the_remedy():
    """"must be a string" with no next step is an error the model retries blindly."""
    out = sb.save_summary_to_second_brain(
        title={"a": 1}, summary_content=REAL_BODY, topics="x"
    )
    assert "plain text" in out


# --- the sibling tool had the same exposure -----------------------------------


def test_log_conversation_refuses_rather_than_raising():
    """It calls `.strip()` on both arguments, so a list killed it identically.

    Found by reading rather than by waiting for a second live failure: same shape,
    same file, one function away, and it is the other half of rule 9 -- so a crash
    here loses the conversation log as well as the note.
    """
    out = sb.log_conversation(user_message=["a"], agent_response="fine")
    assert "user_message must be a string" in out
    out = sb.log_conversation(user_message="fine", agent_response={"a": 1})
    assert "agent_response must be a string" in out


# --- what must NOT change ----------------------------------------------------


def test_the_string_path_is_untouched():
    """The control. A guard that refused everything would pass the tests above."""
    kept, dropped = sb._split_topics("Filmes, Osasco, Cinema")
    assert kept == ["Filmes", "Osasco", "Cinema"] and dropped == []


def test_the_existing_dropped_topic_reporting_still_works():
    """`_split_topics` returns two lists; the second is not decoration."""
    kept, dropped = sb._split_topics("Filmes, .., Osasco")
    assert kept == ["Filmes", "Osasco"]
    assert dropped == [".."], "a topic with no usable file name is still reported"


def test_a_scalar_is_still_accepted():
    """`None` and numbers are not "wrong shape", and used to be handled by `or ""`."""
    assert sb._split_topics(None)[0] == []
    assert sb._split_topics(5)[0] == ["5"]