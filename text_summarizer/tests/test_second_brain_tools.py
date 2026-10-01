"""The two vault-writing tools, at the level of the files they leave behind.

``save_summary_to_second_brain`` has two test modules against it already --
``test_second_brain_write.py`` treats every argument as hostile and asserts on
path containment and YAML quoting, and this one is everything else. The division
is by intent, not by convenience: that module asks "can untrusted input escape
the vault", this one asks "is the graph the tool is supposed to build actually
built".

**``log_conversation`` had no test at all**, which is worth stating plainly
because it is a *mandatory* eval criterion. ``test_config.json`` declares
``tool_uses: [save_summary_to_second_brain, log_conversation]`` with
``matchType: IN_ORDER`` and a threshold of 1.0, so an eval run where the agent
answers correctly and skips the chat log scores **0.00** on
``tool_trajectory_avg_score`` (AGENTS.md, gotcha 5). A tool that gates the whole
graded metric with no direct test is the largest single hole this MVP closes.

Its output is also the only record of what was asked. The chat log is how the
vault is searched when a question comes back later -- "reconstruct what was
discussed by reading the relevant day's note" is the function's own docstring --
so the format is a durable interface, not a debug aid. The shape pinned below is
the format: a dated note, one ``## HH:MM`` heading per exchange, and the two
labelled sections. Changing any of it retroactively makes older days unreadable
in the way the log is read.

The assertions are on file contents and on the returned paths, never on the
interior helpers. ``second_brain.py`` has been through an atomicity refactor
(``_append_chat_entry``, ``_append_once``, sidecar locks) whose whole purpose was
to keep this output byte-identical while making concurrent writes safe; a test
that reached into those helpers would have had to be rewritten for no gain.
"""

from __future__ import annotations

import re
from datetime import date

import pytest
from _helpers import _point_vault_at
from text_summarizer import second_brain
from text_summarizer.second_brain import (
    INDEX_NAME,
    log_conversation,
    save_summary_to_second_brain,
)

TODAY = date.today().isoformat()

#: ``## HH:MM`` -- the one timestamp format the log has ever used, and the reason
#: a day's note is navigable at all. Written out rather than recomputed with
#: ``datetime.now()``: a test that formats its expectation with the same call the
#: code uses would still pass across a change from ``%H:%M`` to ``%H:%M:%S``, and
#: every day already written would then carry a different shape.
TIME_HEADING = re.compile(r"^## \d{2}:\d{2}$", re.MULTILINE)

#: The link the day's chat log is filed under in the shared index. It is the only
#: handle on that day's note from the hub, so a rename would leave every existing
#: index entry pointing at a note that no longer exists.
CHAT_LOG_LINK = f"- [[{TODAY} - chat log]]"


@pytest.fixture
def vault(monkeypatch, tmp_path):
    """A throwaway vault, and its path as a string."""
    monkeypatch.setenv("VAULT_NAME", "ck")
    return _point_vault_at(monkeypatch, tmp_path)


def _chat_log(vault_path: str) -> str:
    return f"{vault_path}/Chat Log/{TODAY}.md"


def _index(vault_path: str) -> str:
    return f"{vault_path}/{INDEX_NAME}.md"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _topic_stub(vault_path: str, topic: str) -> str:
    return f"{vault_path}/Topics/{second_brain._safe_name(topic)}.md"


def _write_topic_stub(vault_path: str, topic: str, body: str) -> str:
    """Pre-create a topic note, as if a person or an older version wrote it.

    Writing the file directly rather than through the tool is what makes the
    assertions about *existing* notes meaningful: a note the tool created itself
    always has the exact shape the tool writes, so it can never exercise the
    branches that handle one it did not.
    """
    path = _topic_stub(vault_path, topic)
    second_brain._write(path, body)
    return path


# ==============================================================================
# log_conversation
# ==============================================================================


def test_the_first_exchange_creates_a_titled_dated_note(vault):
    """One file per day, under ``Chat Log/``, titled with that day.

    The day is in the *filename* as well as the heading, because the note is
    reached by reading the day's file and the heading is not what a glob finds.
    """
    log_conversation("Summarize: dogs.", "- Dogs are loyal.")

    body = _read(_chat_log(vault))

    assert body.startswith(f"# Chat Log -- {TODAY}\n\n")
    assert "- Dogs are loyal." in body


def test_the_entry_is_a_timestamped_heading_over_two_labelled_sections(vault):
    """The format, exactly: ``## HH:MM``, then ``**User:**``, then ``**Agent:**``.

    ``log_conversation`` is called by the model with the user's exact message and
    the final answer, and both are stored verbatim -- the leading and trailing
    whitespace stripped, nothing else touched. Stripping is the only editing
    done, because an answer that starts with a bullet keeps its bullets and one
    that begins with a quote keeps the quote, and a log that normalised them
    would stop being evidence of what the user actually saw.
    """
    log_conversation(
        "  Summarize: the Voyager probes.\n",
        "  - They left in 1977.\n- They are still transmitting.\n  ",
    )

    body = _read(_chat_log(vault))
    headings = TIME_HEADING.findall(body)

    assert len(headings) == 1, f"expected one time heading, found {headings}"
    assert "**User:**\n\nSummarize: the Voyager probes.\n" in body
    assert "**Agent:**\n\n- They left in 1977.\n- They are still transmitting.\n" in body


def test_a_second_exchange_is_appended_not_written_over(vault):
    """The day's note accumulates.

    ``_write`` would truncate, and a day's log would then hold only the last
    exchange -- which is the one the conversation is least likely to be asked
    about. Both exchanges have to be readable from the same file afterwards, and
    the first one's text must be byte-identical, because that is the record.
    """
    log_conversation("first question", "- first answer")
    log_conversation("second question", "- second answer")

    body = _read(_chat_log(vault))

    assert len(TIME_HEADING.findall(body)) == 2, "one heading per exchange"
    assert body.index("first question") < body.index("second question")
    assert "- first answer" in body and "- second answer" in body
    # Exactly one copy of each: a rewrite that re-read and re-wrote would risk
    # doubling or losing the earlier entry.
    assert body.count("first question") == 1


def test_two_exchanges_in_the_same_minute_lose_nothing_despite_one_timestamp(vault):
    """A real collision, asserted so it stays a known limitation.

    The heading is ``%H:%M``, so two exchanges in the same minute get the same
    heading and a reader cannot tell them apart by time alone -- only by position.
    Nothing is lost: both exchanges are present, in order. Recorded here because
    the alternative (seconds in the heading) would break the format of every day
    already written, and the cost of this ambiguity is much lower than that.
    """
    log_conversation("at 09:14", "- one")
    log_conversation("also at 09:14", "- two")

    body = _read(_chat_log(vault))

    assert len(TIME_HEADING.findall(body)) == 2
    assert body.count("at 09:14") == 2
    assert body.index("- one") < body.index("- two")


def test_an_identical_exchange_is_logged_twice_because_it_is_a_log(vault):
    """Deduplication would be the wrong feature here.

    The vault cache replays a repeat question without calling the tools at all,
    so a genuine second exchange only reaches this function when the cache is
    off -- which is the case ``adk eval`` runs in. Collapsing the two would make
    the eval run's chat log disagree with what the agent actually did, which is
    the opposite of what a log is for.
    """
    log_conversation("same question", "- same answer")
    log_conversation("same question", "- same answer")

    body = _read(_chat_log(vault))

    assert body.count("same question") == 2
    assert len(TIME_HEADING.findall(body)) == 2


def test_an_empty_exchange_still_produces_a_parseable_entry(vault):
    """Both labels are always written, even with nothing to put between them.

    Reachable: the instruction asks for "the user's exact message and your final
    answer", and an answer can be empty on a turn that died. An entry missing its
    ``**Agent:**`` label would make every reader -- including the search that
    finds this note later -- have to special-case the last entry of the day.
    """
    log_conversation("what is the weather?", "")

    body = _read(_chat_log(vault))

    assert "**User:**\n\nwhat is the weather?\n" in body
    assert "**Agent:**" in body


def test_the_shared_index_gains_the_day_once_and_only_once(vault):
    """One line per day, however many exchanges that day holds.

    The index is the only place the day's chat log is discoverable from, so a
    second line would put the same note in the vault twice; and the entry has to
    be there at all, since a chat log nothing links to is findable only by
    guessing today's date.
    """
    log_conversation("one", "- one")
    log_conversation("two", "- two")

    assert _read(_index(vault)).count(CHAT_LOG_LINK) == 1


def test_the_index_and_the_note_share_one_file_with_saved_summaries(vault):
    """Both tools append to ``Second Brain Index.md``, and neither may reset it.

    This is the collision the shared file exists to absorb. Whichever tool wrote
    last used to seed the file when it found it empty and append when it did not,
    so a save followed by a log left the summary's link gone or the log's -- and
    both look like a working vault, because the notes themselves are all there.
    """
    save_summary_to_second_brain(title="Dogs", summary_content="- loyal", topics="Dogs")
    log_conversation("a question", "- an answer")

    index = _read(_index(vault))

    assert index.startswith(f"# {INDEX_NAME}"), "the title must survive the second writer"
    assert CHAT_LOG_LINK in index
    assert f"- [[{TODAY} - dogs]]" in index
    assert index.count(CHAT_LOG_LINK) == 1


def test_an_index_that_already_has_content_is_extended_not_replaced(vault):
    """The branch that matters is the non-empty one.

    On a fresh vault the index is seeded with a title and ``None yet.``, and
    every test above reaches it. The dangerous branch is the other one -- where a
    populated index could be overwritten by a header -- and reaching it needs a
    pre-existing index rather than a tool call, so it is set up by hand here.
    """
    second_brain._write(_index(vault), f"# {INDEX_NAME}\n\n- [[some older note]]\n")

    log_conversation("a question", "- an answer")

    index = _read(_index(vault))

    assert "- [[some older note]]" in index, "an entry written before this run was lost"
    assert CHAT_LOG_LINK in index
    assert index.count("None yet.") == 0, (
        "the placeholder must be replaced by the first real entry, not joined to it"
    )


def test_the_return_string_names_both_paths_and_they_are_the_real_ones(vault):
    """The tool's output is read by the model, and these are its only coordinates.

    Two paths: where the exchange went, and where the index was updated. The
    model is told to mention the saved note and its title, and when a user asks
    "where did that go?" the answer comes from these lines -- so a path that is
    stale, relative, or the wrong file makes the tool's report a lie that nothing
    else would catch.
    """
    result = log_conversation("a question", "- an answer")

    assert result == f"Logged conversation to {_chat_log(vault)}\nUpdated index: {_index(vault)}"
    for path in result.split():
        if path.startswith("/"):
            assert path.split("/")[-1], f"a bare directory in the result: {path}"
    assert _read(_chat_log(vault)) and _read(_index(vault))


def test_the_chat_log_note_is_inside_the_vault(vault):
    """Containment, checked where the chat log is concerned rather than in the writer.

    ``save_summary_to_second_brain``'s path handling is covered thoroughly in
    ``test_second_brain_write.py``; this exists because ``log_conversation`` has
    no title or topic going through the sanitiser at all -- the path is built from
    ``CHAT_LOG_DIR`` and a date -- and a *different* function writing outside the
    vault would not be caught by a test aimed at the other one.
    """
    log_conversation("a question", "- an answer")

    written = [
        path
        for path in second_brain.Path(vault).rglob("*")
        if path.is_file() and path.name.endswith(".md")
    ]

    assert written, "the exchange has to be somewhere"
    for path in written:
        assert path.resolve().is_relative_to(second_brain.Path(vault).resolve())


# ==============================================================================
# save_summary_to_second_brain -- the graph it builds
# ==============================================================================


def test_a_new_topic_stub_is_a_heading_and_one_backlink(vault):
    """The stub's whole contract, written out once.

    ``# <Topic>`` (the readable original, not the slug) then ``## Backlinks`` then
    one ``- [[<note>]]``. A topic note is the hub of the graph the vault claims to
    be: it is reached from the summary's ``## Related`` section and from the index,
    and it points back at every note that mentions it. A stub created without the
    backlink section is a stub that will collect links forever with nowhere to
    show them, and it is indistinguishable from one that already has some.
    """
    save_summary_to_second_brain(title="Dogs", summary_content="- loyal", topics="Dogs")

    note_title = f"{TODAY} - dogs"
    assert _read(_topic_stub(vault, "Dogs")) == (f"# Dogs\n\n## Backlinks\n- [[{note_title}]]\n")


def test_two_notes_citing_one_topic_become_two_backlinks_under_one_heading(vault):
    """The mechanism, as opposed to the creation.

    Creating a stub proves a file can be made; this is why the file exists. Two
    different notes about the same topic each add a line, the ``## Backlinks``
    heading appears exactly once, and both notes survive -- which is the property
    that makes the vault a graph rather than a folder.
    """
    save_summary_to_second_brain(title="Dogs", summary_content="- a", topics="Dogs")
    save_summary_to_second_brain(title="Dog Breeds", summary_content="- b", topics="Dogs")

    stub = _read(_topic_stub(vault, "Dogs"))

    assert stub.count("## Backlinks") == 1, "the heading must not be repeated"
    assert f"- [[{TODAY} - dogs]]" in stub
    assert f"- [[{TODAY} - dog-breeds]]" in stub
    assert stub.index("dogs") < stub.index("dog-breeds"), "backlinks append, in order"


def test_the_same_note_citing_a_topic_twice_gets_one_backlink(vault):
    """The guard, which is a no-op branch and therefore easy to drop.

    Without it, two saves of the same title (the documented same-day overwrite)
    would leave the note linked twice from its own topic, and a graph view would
    draw the edge twice.
    """
    save_summary_to_second_brain(title="Dogs", summary_content="- a", topics="Dogs")
    save_summary_to_second_brain(title="Dogs", summary_content="- b", topics="Dogs")

    assert _read(_topic_stub(vault, "Dogs")).count(f"- [[{TODAY} - dogs]]") == 1


def test_a_backlink_is_appended_to_an_existing_heading(vault):
    """The ``"## Backlinks" in body`` branch, on a note that already has one.

    Set up by hand because no code path creates that state: the tool only ever
    writes a stub *with* the heading. A note a person added the heading to, or one
    from a version that did, is the case the branch is for.
    """
    _write_topic_stub(vault, "Dogs", "# Dogs\n\n## Backlinks\n- [[some older note]]\n")

    save_summary_to_second_brain(title="New Note", summary_content="- x", topics="Dogs")

    stub = _read(_topic_stub(vault, "Dogs"))

    assert stub.count("## Backlinks") == 1
    assert stub.endswith(f"- [[{TODAY} - new-note]]\n"), "appended after the existing one"
    assert "- [[some older note]]" in stub


def test_a_backlink_creates_the_section_on_a_note_that_lacks_one(vault):
    """The other branch, and the only one no test covered.

    A topic note without a ``## Backlinks`` heading -- written by a person, or by
    an older version of this tool -- has to *gain* the section rather than have a
    bare list appended after its prose. Both spellings are close to each other in
    the source and produce visibly different files, and the difference is exactly
    whether a reader opening the note can find the links.
    """
    _write_topic_stub(vault, "Dogs", "# Dogs\n\nA note I wrote by hand.\n")

    save_summary_to_second_brain(title="New Note", summary_content="- x", topics="Dogs")

    stub = _read(_topic_stub(vault, "Dogs"))

    assert stub.count("## Backlinks") == 1
    assert "## Backlinks" in stub
    assert stub.index("A note I wrote by hand.") < stub.index("## Backlinks"), (
        "the hand-written prose must come first: the section is appended, not prepended"
    )
    assert stub.endswith(f"- [[{TODAY} - new-note]]\n")


def test_only_genuinely_new_stubs_are_reported_as_created(vault):
    """The one line of the result string that carries new information.

    The note is always written, so ``Saved note`` is always true and says nothing.
    ``Created topic stubs`` is the only part that distinguishes a turn that
    extended the graph from one that started it -- and the model reads this string
    to decide what to tell the user, so a topic that already existed appearing in
    that list is a claim that a new note was created when none was.
    """
    _write_topic_stub(vault, "Dogs", "# Dogs\n\n## Backlinks\n- [[older]]\n")

    result = save_summary_to_second_brain(
        title="Mixed", summary_content="- x", topics="Dogs, Checkout, Vaults"
    )

    assert "Created topic stubs: Checkout, Vaults" in result
    assert "Dogs," not in result.split("Created topic stubs:")[1]
    # The existing one was still linked, which is the other half of the claim.
    assert f"- [[{TODAY} - mixed]]" in _read(_topic_stub(vault, "Dogs"))


def test_the_related_block_is_one_wikilink_per_topic_in_the_models_order(vault):
    """``## Related`` is the outgoing half of the edge, and its format is load-bearing.

    Bare lines are joined into one run-on paragraph by every markdown renderer and
    by the dev UI's message component, so the ``- `` prefix is part of the format
    rather than decoration -- the same reason ``sources.SOURCE_BULLET`` is a
    constant. Order follows the model's topic list so the note reads the way it
    was written, and nothing may follow the last link.
    """
    save_summary_to_second_brain(
        title="Shopping", summary_content="- x", topics="Mobile App, Checkout, Payments"
    )

    note = f"{vault}/{second_brain.BRAIN_DIR}/{TODAY} - shopping.md"
    related = _read(note).split("## Related\n", 1)[1]

    assert related == "- [[Mobile App]]\n- [[Checkout]]\n- [[Payments]]\n"


# ==============================================================================
# _safe_name / _slug
# ==============================================================================


def test_the_legacy_alias_is_the_same_function_not_a_second_implementation(vault):
    """``_slug`` exists only so pre-provenance callers keep compiling.

    Its docstring says it outright: "two sluggers is how the old one got a weaker
    one", and the weaker one is exactly what leaked ``..`` into filenames (the B1
    traversal in ``test_second_brain_write.py``). So the alias has to stay an
    alias. A second implementation would go green here and the sanitiser would
    stop being the single thing every name passes through.
    """
    for value in ("Dogs", "Mobile App", "../../etc/passwd", "..", "", "x" * 80):
        assert second_brain._slug(value) == second_brain._safe_name(value), value


def test_the_alias_forwards_the_length_cap_rather_than_dropping_it(vault):
    """``max_len`` is a parameter of both names.

    Dropped in the alias, every old caller would silently get the 60-character
    default -- a silent change to every filename they produce, in the one function
    where the length is the only thing standing between a model's output and
    ``ENAMETOOLONG``.
    """
    assert second_brain._slug("x" * 100, max_len=10) == "x" * 10
    assert second_brain._safe_name("x" * 100, max_len=10) == "x" * 10


def test_truncation_cannot_leave_a_trailing_dash(vault):
    """The one rule of the sanitiser that the cap alone would break.

    ``text[:max_len].strip("-")``: truncating ``abc-defgh`` at four characters
    yields ``abc-``, and a filename ending in ``-`` is legal but wrong -- it reads
    as a truncated slug to anyone looking at the Topics folder, and two different
    titles can collide on it. Not a safety issue; a correctness one, and it is
    only reachable *at* the cap, which is why the boundary is asserted directly
    rather than with a long input. The unbounded cases are pinned in
    ``test_second_brain_write.py``; these are the ones only a short ``max_len``
    can reach.
    """
    assert second_brain._safe_name("abc-defgh", max_len=4) == "abc"
    assert second_brain._safe_name("abc-defgh", max_len=3) == "abc"
    assert second_brain._safe_name("--abc", max_len=2) == "", (
        "a name that is nothing but separators still reduces to nothing rather "
        "than to the separators themselves"
    )
    # Just past the cap: the dash at index 3 is the last character, so the trim
    # has something to remove. A cap that stopped at 4 would leave "abc-".
    assert second_brain._safe_name("abc-def", max_len=4) == "abc"
    assert second_brain._safe_name("abc-def", max_len=5) == "abc-d"
    # And the default cap, where the trailing-dash input is long enough to reach it.
    assert second_brain._safe_name("x" * 59 + "-y") == "x" * 59


def test_a_title_that_collapses_to_nothing_still_names_the_note(vault):
    """The fallback name, asserted through the tool rather than on the slugger.

    ``<date> - .md`` is a real filename that opens as a blank note, and it is what
    a slugger returning ``""`` would produce if the ``or "summary"`` branch were
    dropped. The containment tests in ``test_second_brain_write.py`` already cover
    the paths; this is about the name being legible to a person in the vault.
    """
    save_summary_to_second_brain(title="..", summary_content="- x", topics="Dogs")

    note = f"{vault}/{second_brain.BRAIN_DIR}/{TODAY} - summary.md"
    assert "- loyal" in _read(note) or "- x" in _read(note)


def test_a_long_title_is_truncated_in_the_slug_and_leaves_the_date_prefix(vault):
    """The cap applies to the slug, not to the note name.

    ``max_len`` is 60 and a note name is ``<date> - <slug>``, so a long title has
    its slug shortened while the date prefix -- which is what sorts the Second
    Brain folder and what makes the note findable by date -- survives intact. A
    cap applied to the whole name would truncate the date off the front and every
    long-titled note would become undated.
    """
    title = "Customer Feedback " + "and more " * 12

    save_summary_to_second_brain(title=title, summary_content="- x", topics="Dogs")

    stem = f"{TODAY} - " + second_brain._safe_name(title)
    note = f"{vault}/{second_brain.BRAIN_DIR}/{stem}.md"

    assert second_brain._safe_name(title) == (
        "customer-feedback-and-more-and-more-and-more-and-more-and-mo"
    ), "60 characters of slug; the tail of the title is the part that goes"
    assert len(second_brain._safe_name(title)) == 60
    assert stem.startswith(TODAY)
    assert len(stem) == len(TODAY) + 3 + 60
    assert "- x" in _read(note)
