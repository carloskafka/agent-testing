"""Tests for digest mode: reading a day out of the vault without the agent.

What is being pinned here is not the wording of a rendering -- it is a set of
properties that a filesystem reader can quietly get wrong, each of which would make a
digest *lie* rather than fail:

* it reports provenance that was recorded, and **reports nothing** where none was;
* it never lets a ``[obsidian][ck][…]`` marker reach a digest line;
* a damaged note still appears;
* the same vault produces the same output, in a known order;
* a day with nothing in it is an answer, not a failure;
* and it reaches neither ``google.adk`` nor ``google.genai``, which is what makes
  "no quota" a property of the code rather than a claim about how it is invoked.

Notes are planted in the writer's exact on-disk shape -- built with
``second_brain._yaml_str`` and ``second_brain._safe_name``, the same helpers
``save_summary_to_second_brain`` uses -- so these tests check the reader against the
format that is really written rather than against a convenient approximation of it.

The vault is repointed with the shared ``_point_vault_at`` from :file:`_helpers.py`.
It works on this module unchanged because ``digest.default_vault_root`` reads
``second_brain.VAULT_ROOT`` at *call* time instead of binding a third copy of the path
at import; that is the difference between one vault binding to repoint and three.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap
from datetime import date

import pytest
from _helpers import SERVED_MODEL, _point_vault_at
from text_summarizer import digest, second_brain

DAY = "2026-09-30"
OTHER_DAY = "2026-09-29"
FP = "a" * 64

#: A rendered ``**Sources**` block, in the shape ``sources.render_sources`` emits and in
#: the one shape it is most likely to be recognised by: the linked-title spelling.
SOURCES_BLOCK = (
    "\n**Sources**\n"
    "- [obsidian][ck][gemini-3.5-flash-lite][\\[\\[[Other Note]\\]\\]"
    "(http://127.0.0.1:8000/vault/Second%20Brain/Other.md): same topic\n"
    "- [obsidian][ck][unknown][[Older Note]]: provenance not recorded\n"
)

#: The raw form the model produces, which is what a chat log holds: the sentinel tokens
#: are still in place because ``log_conversation`` runs before the answer is rewritten.
RAW_SOURCES_BLOCK = (
    "\n[obsidian][@@ADK_VAULT@@][@@ADK_MODEL@@][[Other Note]]: same topic\n"
    "[obsidian][@@ADK_VAULT@@][@@ADK_MODEL@@][[Older Note]]: older\n"
)


def _plant(
    vault,
    stem: str,
    *,
    summary: str = "",
    topics: tuple[str, ...] = (),
    model: str | None = SERVED_MODEL,
    in_vault: str | None = "ck",
    fingerprint: str | None = FP,
    day: str = DAY,
    alias: str | None = None,
    raw_frontmatter: str | None = None,
    related: str = "",
) -> str:
    """Write one note into ``Second Brain/`` in the writer's own format.

    Every field is individually switchable so a test can plant a note that is missing
    exactly one thing, which is the only way to check that the reader degrades per
    field. ``raw_frontmatter`` replaces the assembled block entirely and is the
    **complete** block, both fences included, exactly as it appears in the file -- which
    is what lets a test plant the shapes the writer cannot emit: a note somebody edited
    by hand, one whose ``tags:`` will not parse, or one whose closing fence never landed.
    """
    brain = vault / second_brain.BRAIN_DIR
    brain.mkdir(parents=True, exist_ok=True)

    if raw_frontmatter is None:
        lines = ["---", "tags:"]
        lines += [f"  - {second_brain._yaml_str(topic)}" for topic in topics]
        lines.append(f"date: {day}")
        if fingerprint:
            lines.append(f"source_fingerprint: {fingerprint}")
        if model:
            lines.append(f"{second_brain.GENERATED_BY_MODEL_KEY}: {second_brain._yaml_str(model)}")
        if in_vault:
            lines.append(
                f"{second_brain.GENERATED_IN_VAULT_KEY}: {second_brain._yaml_str(in_vault)}"
            )
        if alias is not None:
            lines.append("aliases:")
            lines.append(f"  - {second_brain._yaml_str(alias)}")
        lines.append("---")
        raw_frontmatter = "\n".join(lines)

    links = "\n".join(f"- [[{topic}]]" for topic in topics)
    # `related` defaults to the sentinel empty string rather than to None so that
    # "build the writer's own Related block" and "append exactly these lines" stay
    # distinguishable: passing an empty Related section is itself a case worth planting.
    tail = f"{SOURCES_BLOCK}\n\n## Related\n{links}\n" if related == "" else related
    (brain / f"{stem}.md").write_text(f"{raw_frontmatter}\n\n{summary}{tail}", encoding="utf-8")
    return str(brain / f"{stem}.md")


def _plant_topic(vault, topic: str, backlinks: tuple[str, ...]) -> str:
    """Write a ``Topics/`` stub, resolved through the writer's own sanitiser."""
    folder = vault / second_brain.TOPICS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{second_brain._safe_name(topic)}.md"
    links = "".join(f"- [[{note}]]\n" for note in backlinks)
    path.write_text(f"# {topic}\n\n## Backlinks\n{links}", encoding="utf-8")
    return str(path)


def _plant_index(vault, stems: tuple[str, ...]) -> str:
    path = vault / f"{second_brain.INDEX_NAME}.md"
    links = "".join(f"- [[{stem}]]\n" for stem in stems)
    path.write_text(f"# {second_brain.INDEX_NAME}\n\nNone yet.\n{links}", encoding="utf-8")
    return str(path)


def _plant_chat(vault, day: str, entries: tuple[tuple[str, str, str], ...]) -> str:
    """Write a ``Chat Log/`` day in the shape ``log_conversation`` produces."""
    folder = vault / second_brain.CHAT_LOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    body = f"# Chat Log -- {day}\n\n"
    for time, user, agent in entries:
        body += f"## {time}\n\n**User:**\n\n{user}\n\n**Agent:**\n\n{agent}\n"
    path = folder / f"{day}.md"
    path.write_text(body, encoding="utf-8")
    return str(path)


@pytest.fixture
def vault(monkeypatch, tmp_path):
    """A writable vault named ``ck``, with ``Second Brain/`` in place.

    ``_point_vault_at`` hands back the path it resolved as a string, which is what its
    other callers want; these plant files, so it is wrapped in a ``Path`` once here
    rather than at every call site.
    """
    return pathlib.Path(_point_vault_at(monkeypatch, tmp_path, "ck", modules=("second_brain",)))


# --- the happy path --------------------------------------------------------------


def test_a_day_with_notes_lists_them_with_provenance(vault):
    """Title, summary, topics, provenance, hub position -- all read off the note.

    Asserted against ``--json`` because that is the structure a script consumes, and
    because asserting on a rendering would not notice that a field had quietly moved
    from the JSON into prose.
    """
    _plant(
        vault,
        f"{DAY} - apollo-program-summary",
        summary="- The Apollo program ran from 1961 to 1972.\n- It landed eleven humans.",
        topics=("Apollo program", "Moon landings"),
        alias="Apollo Program Summary",
    )
    _plant(
        vault,
        f"{DAY} - voyager-1",
        summary="- Voyager 1 is the most distant human-made object.",
        topics=("Space exploration",),
    )
    _plant_topic(vault, "Apollo program", (f"{DAY} - apollo-program-summary",))
    _plant_topic(vault, "Space exploration", (f"{DAY} - voyager-1", f"{OTHER_DAY} - older"))
    _plant_index(vault, (f"{OTHER_DAY} - older", f"{DAY} - apollo-program-summary"))

    payload = digest.to_dict(digest.build_digest(DAY, vault_root=vault))

    assert payload["date"] == DAY
    assert payload["note_count"] == 2
    assert payload["vault"]["name"] == "ck"

    apollo = payload["notes"][0]
    assert apollo["title"] == f"{DAY} - apollo-program-summary"
    assert apollo["alias"] == "Apollo Program Summary"
    assert apollo["path"] == f"Second Brain/{DAY} - apollo-program-summary.md"
    assert apollo["summary"].splitlines() == [
        "- The Apollo program ran from 1961 to 1972.",
        "- It landed eleven humans.",
    ]
    assert apollo["topics"] == ["Apollo program", "Moon landings"]
    assert apollo["provenance"] == {
        "generated_by_model": SERVED_MODEL,
        "generated_in_vault": "ck",
    }
    assert apollo["index_position"] == 2
    assert apollo["has_frontmatter"] is True

    topics = {tally["topic"]: tally for tally in payload["topics"]}
    assert topics["Space exploration"]["backlinks"] == 2
    assert topics["Apollo program"]["backlinks"] == 1
    assert topics["Space exploration"]["notes"] == [f"{DAY} - voyager-1"]


def test_provenance_is_omitted_rather_than_invented_when_absent(vault):
    """A note with no provenance gains none -- in any rendering, in any key.

    Notes written before the frontmatter fields existed are the reason. Reporting the
    model that happens to be serving *now*, or the vault the reader happens to be in,
    would turn an absent record into a false one, and a digest is read as a record.
    The JSON key is checked for absence rather than for a value, because ``None`` and
    ``"unknown"`` are both ways of failing that quietly.
    """
    _plant(
        vault,
        f"{DAY} - pre-provenance",
        summary="- An old summary.",
        topics=("History",),
        model=None,
        in_vault=None,
        alias="Pre Provenance",
    )
    # A vault whose own name is knowable, so "we could have filled this in" is a real
    # temptation and not a theoretical one.
    assert digest.default_vault_root() == str(vault)

    built = digest.build_digest(DAY, vault_root=vault)
    note = built.notes[0]
    assert note.provenance == {}
    assert not any("model" in key for key in digest.to_dict(built)["notes"][0]["provenance"])

    text = digest.render_text(built)
    md = digest.render_md(built)
    assert SERVED_MODEL not in text and SERVED_MODEL not in md
    assert "model:" not in text
    assert "**Provenance:**" not in md
    # The name is reported where it is a fact about the *vault being read*, which is
    # different from a fact about the note.
    assert "vault ck at" in text


def test_the_sources_block_is_stripped_from_every_digest_line(vault):
    """No ``[obsidian][…]`` marker survives, in the summaries or in the chat log.

    Those markers are this project's own provenance syntax and are meaningless outside
    a rendered answer -- ``[obsidian][ck][gemini-3.5-flash-lite][[Note]]`` is noise in a
    terminal and, worse, something a reader could mistake for content. The two spellings
    differ deliberately: the summary carries the *rendered* block with a clickable title,
    the chat log the *raw* one with sentinel tokens still in place, because
    ``log_conversation`` records the exchange before the answer is rewritten. A stripper
    that only knew the first would leak the second.
    """
    _plant(
        vault,
        f"{DAY} - cited",
        summary="- A cited summary.",
        topics=("Other Note",),
        alias="Cited",
    )
    _plant_chat(
        vault,
        DAY,
        (("09:00", "Summarize: something", f"- An answer.{RAW_SOURCES_BLOCK}"),),
    )

    built = digest.build_digest(DAY, vault_root=vault, include_chat=True)
    for text in (
        built.notes[0].summary,
        built.chat[0].agent,
        digest.render_text(built),
        digest.render_md(built),
        digest.to_json(built),
    ):
        assert "[obsidian]" not in text
        assert "@@ADK_VAULT@@" not in text
        assert "@@ADK_MODEL@@" not in text

    # Stripping must not take the summary with it.
    assert built.notes[0].summary == "- A cited summary."
    assert built.chat[0].user == "Summarize: something"
    assert "- An answer." in built.chat[0].agent


# --- degenerate days and damaged notes ------------------------------------------


def test_an_empty_day_is_a_clean_result_not_an_error(vault, capsys):
    """Exit 0, a message that names the day, and no invented content.

    A day the agent learned nothing on is a fact worth being able to ask for, and a
    command that exited non-zero there could not be used from a cron job that reports
    only when there is something to say. The message has to name the day: "no notes"
    without it is not an answer, it is a shrug.
    """
    _plant(vault, f"{OTHER_DAY} - yesterday", summary="- Yesterday's note.")
    assert digest.build_digest(OTHER_DAY, vault_root=vault).notes, "control: the vault has notes"

    assert digest.main(["--date", "2026-01-01", "--vault", str(vault)]) == 0
    out = capsys.readouterr().out
    assert "no notes for 2026-01-01" in out
    assert "yesterday" not in out

    # Same through the machine-readable path, where the emptiness has to be visible as
    # a count rather than inferred from a missing section.
    assert digest.main(["--date", "2026-01-01", "--vault", str(vault), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["notes"] == [] and payload["note_count"] == 0
    assert payload["topics"] == []


def test_a_malformed_date_exits_non_zero_with_a_usage_message(vault, capsys):
    """Both spellings of wrong are rejected, and the message names the expected format.

    ``30/09/2026`` and ``2026-13-01`` fail for different reasons -- one is the wrong
    shape, the other is a well-shaped date that does not exist -- and both must be
    caught. Silently coercing either one would report on a day nobody asked for, which
    is worse than failing: the output would look entirely reasonable.
    """
    for bad in ("30/09/2026", "20260930", "2026-13-01", "yesterday", ""):
        with pytest.raises(SystemExit) as excinfo:
            digest.main(["--date", bad, "--vault", str(vault)])
        assert excinfo.value.code != 0, f"{bad!r} was accepted"
        err = capsys.readouterr().err
        assert "usage:" in err
        assert "YYYY-MM-DD" in err
        assert bad in err

    assert digest.main(["--date", DAY, "--vault", str(vault)]) == 0


def test_malformed_frontmatter_still_appears_with_the_resolvable_fields(vault):
    """A damaged note is reported with what could be read, and flagged as damaged.

    Two things are checked that pull in opposite directions. The note must **appear**,
    because a digest that dropped unreadable notes would report a busy day as a quiet
    one. And the *fields* must be read independently, because a real YAML parse is
    all-or-nothing: one unparseable ``tags:`` line would take ``generated_by_model``
    down with it, turning a partial read into no read at all. That is why the reader
    is per key.
    """
    _plant(
        vault,
        f"{DAY} - half-written",
        summary="- Readable even though the frontmatter is not.",
        raw_frontmatter=(
            "---\n"
            "tags: [unterminated\n"
            f"date: {DAY}\n"
            f'{second_brain.GENERATED_BY_MODEL_KEY}: "{SERVED_MODEL}"\n'
            "aliases:\n"
            '  - "Half Written"\n'
            "---"
        ),
        related="\n\n## Related\n- [[Cats]]\n- [[Pets]]\n",
    )
    # No closing fence at all: the writer cannot produce this, a half-flushed write or
    # a hand edit can, and the reader must not treat the whole body as frontmatter.
    _plant(
        vault,
        f"{DAY} - unterminated-fence",
        summary="- Still readable.",
        raw_frontmatter=f"---\ndate: {DAY}\ngenerated_by_model: {SERVED_MODEL}",
    )

    payload = digest.to_dict(digest.build_digest(DAY, vault_root=vault))
    by_title = {note["title"]: note for note in payload["notes"]}
    assert len(by_title) == 2

    half = by_title[f"{DAY} - half-written"]
    assert half["has_frontmatter"] is True
    # Provenance survived the broken tags line, and the topics were recovered from the
    # one other place the note writes them.
    assert half["provenance"] == {"generated_by_model": SERVED_MODEL}
    assert half["topics"] == ["Cats", "Pets"]
    assert half["alias"] == "Half Written"
    assert half["summary"] == "- Readable even though the frontmatter is not."

    unterminated = by_title[f"{DAY} - unterminated-fence"]
    assert unterminated["has_frontmatter"] is False
    # The point of reusing the writer's reader: an unclosed fence is *not* a fence, so
    # nothing is read as frontmatter -- which is why the model field is not picked up
    # from the middle of what is really the body, and why the body is the whole file.
    assert unterminated["provenance"] == {}
    assert unterminated["topics"] == []
    assert "- Still readable." in unterminated["summary"]


def test_notes_are_sorted_deterministically(vault):
    """Same vault, same order -- and the order is the note titles, not the write order.

    Sorting on the *filename* title is what makes it total: the title is unique within
    one directory and always present, while the alias is model-chosen text that can be
    absent and can tie. A digest whose order depended on ``os.listdir`` would diff
    differently on every run of the same vault, which is exactly what ``--json`` is for.
    """
    stems = [
        f"{DAY} - voyager-1",
        f"{DAY} - apollo-program-summary",
        f"{DAY} - Cats-and-pets",
        f"{DAY} - characteristics-and-purpose-of-dogs",
        f"{DAY} - voyager-1-extra",
    ]
    for stem in stems:
        _plant(vault, stem, summary=f"- {stem}.")

    built = digest.build_digest(DAY, vault_root=vault)
    titles = [note.title for note in built.notes]

    assert titles == sorted(stems, key=str.casefold)
    assert titles == sorted(titles, key=str.casefold)
    # Upper case sorts before lower under a case-sensitive comparison and after it
    # under a case-folded one; the point is that the choice is the case-folded one, so
    # "Cats-and-pets" does not jump to the end because of one capital letter.
    assert titles.index(f"{DAY} - Cats-and-pets") == 1

    text = digest.render_text(built)
    md = digest.render_md(built)
    assert [line for line in titles if line in text] == titles
    assert [line for line in titles if line in md] == titles
    assert [note["title"] for note in digest.to_dict(built)["notes"]] == titles


def test_notes_belong_to_a_day_by_either_their_name_or_their_frontmatter(vault):
    """A note renamed in Obsidian is not lost, and a disagreement is visible.

    The filename prefix is what the writer names files by; the frontmatter ``date:`` is
    the same value. A human who renames a note in Obsidian breaks the first and not the
    second, so requiring both would silently drop it. When they disagree, the reported
    ``date`` is the note's own -- the disagreement is then visible instead of smoothed
    away, which is the honest outcome when two fields of one file claim different days.
    """
    _plant(vault, f"{DAY} - agrees", summary="- Agrees.", day=DAY)
    _plant(vault, "renamed-by-hand", summary="- Renamed.", day=DAY)
    _plant(vault, f"{OTHER_DAY} - belongs-elsewhere", summary="- Elsewhere.", day=OTHER_DAY)

    payload = digest.to_dict(digest.build_digest(DAY, vault_root=vault))
    by_title = {note["title"]: note for note in payload["notes"]}

    assert set(by_title) == {f"{DAY} - agrees", "renamed-by-hand"}
    assert by_title["renamed-by-hand"]["date"] == DAY
    assert by_title[f"{DAY} - agrees"]["date"] == DAY


# --- the two renderings, and the opt-in chat -------------------------------------


def test_text_and_md_formats_differ(vault):
    """Two documents, not one with different separators.

    The claim is not that the strings are unequal -- it is that each is built for its
    medium. The terminal one is flat, numbered and labelled; the note one is headings,
    wikilinks and a table, so what is pasted joins the vault's own graph. Both carry
    the same facts, which is asserted too: a difference in *content* would mean one of
    the two is lying rather than merely formatted differently.
    """
    _plant(
        vault,
        f"{DAY} - apollo",
        summary="- Eleven humans landed on the Moon.",
        topics=("Apollo program",),
        alias="Apollo",
    )
    _plant_topic(vault, "Apollo program", (f"{DAY} - apollo",))
    built = digest.build_digest(DAY, vault_root=vault)
    text = digest.render_text(built)
    md = digest.render_md(built)

    assert text != md

    # Terminal: numbered notes, explicit labels, no markdown heading or table.
    assert "  [1] Apollo" in text
    assert "note: " in text and "topics: " in text
    assert "**Topics:**" not in text
    assert not any(line.startswith("#") for line in text.splitlines())
    assert "| --- |" not in text

    # Note: headings, wikilinks, a table, and no terminal numbering.
    assert md.startswith("# Second Brain digest - ")
    assert "## Apollo" in md
    assert "**Topics:** [[Apollo program]]" in md
    assert "| [[Apollo program]] | 1 |" in md
    assert "  [1] " not in md

    # Same facts in both.
    for shared in ("- Eleven humans landed on the Moon.", SERVED_MODEL):
        assert shared in text and shared in md

    with pytest.raises(ValueError):
        digest.render(built, "html")


def test_chat_entries_are_opt_in(vault):
    """Off by default, in the JSON and in both renderings.

    The chat log is a different kind of record from a summary -- raw prompts and answers
    rather than distilled findings -- so putting it in the default output would bury the
    part that is a digest. ``null`` rather than ``[]`` when it is off, because an empty
    list cannot be told apart from "there was a log and it was empty".
    """
    _plant(vault, f"{DAY} - apollo", summary="- A note.")
    _plant_chat(vault, DAY, (("09:00", "Summarize: dogs", "- About dogs."),))

    default = digest.build_digest(DAY, vault_root=vault)
    assert default.chat is None
    assert digest.to_dict(default)["chat"] is None
    assert "Chat log" not in digest.render_text(default)
    assert "## Chat log" not in digest.render_md(default)

    with_chat = digest.build_digest(DAY, vault_root=vault, include_chat=True)
    assert with_chat.chat is not None and len(with_chat.chat) == 1
    entry = with_chat.chat[0]
    assert (entry.time, entry.user, entry.agent) == ("09:00", "Summarize: dogs", "- About dogs.")

    payload = digest.to_dict(with_chat)
    assert payload["chat"] == [
        {"time": "09:00", "user": "Summarize: dogs", "agent": "- About dogs."}
    ]
    assert "Chat log (1 entries)" in digest.render_text(with_chat)
    assert "**User:** Summarize: dogs" in digest.render_md(with_chat)

    # Opting in to a day that has a note but no log is empty, not an error and not
    # absent -- and it is still reached in the rendering, so the day has to be non-empty.
    _plant(vault, f"{OTHER_DAY} - a-note", summary="- A note.")
    nothing = digest.build_digest(OTHER_DAY, vault_root=vault, include_chat=True)
    assert nothing.chat == []
    assert "no chat entries for" in digest.render_md(nothing)


# --- no model, no network -------------------------------------------------------

_IMPORT_GUARD = textwrap.dedent(
    """
    import pathlib
    import sys
    import types

    BLOCKED = ("google.adk", "google.genai")
    reached = []

    class Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith(BLOCKED):
                reached.append(fullname)
                raise AssertionError("digest imported " + fullname)
            return None

    sys.meta_path.insert(0, Blocker())

    # The parent package initializer exports root_agent and therefore imports the ADK
    # and the GenAI SDK; that is the package's doing, not this module's. It is stubbed
    # out so what is measured is the module under test's own import graph. Everything
    # digest itself pulls in -- second_brain, sources, vaults -- is still imported for
    # real through it.
    package = types.ModuleType("text_summarizer")
    package.__path__ = [sys.argv[2]]
    sys.modules["text_summarizer"] = package

    import text_summarizer.digest as digest

    assert digest.build_digest is not None
    assert not reached, reached
    assert not [m for m in sys.modules if m.startswith(BLOCKED)], sorted(
        m for m in sys.modules if m.startswith(BLOCKED)
    )
    print("OK")
    """
)


def test_the_import_graph_reaches_neither_adk_nor_genai(tmp_path):
    """No model SDK is reachable from this module -- the property "no quota" rests on.

    Asserted two ways because either alone is weak. An import *hook* proves nothing
    tries to reach them; a ``sys.modules`` scan proves they were not already loaded
    before the hook went in. Together they are the real claim, and the assertion is in
    a subprocess so this test's own in-process import of the module -- which does go
    through the package initializer, and does load both -- cannot mask the result.

    The subprocess also gets a writable ``SECOND_BRAIN_VAULT``, because importing
    ``second_brain`` resolves the vault, and a root it cannot create prints a warning
    that would otherwise sit in the captured output for no reason.
    """
    import tempfile

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(digest.__file__)))
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_GUARD, "unused", package_dir],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(
            os.environ,
            SECOND_BRAIN_VAULT=tempfile.mkdtemp(prefix="digest-import-"),
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="",
        ),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.stdout.strip().splitlines()[-1] == "OK"


def test_the_cli_defaults_to_today_and_refuses_a_vault_that_is_not_there(vault, capsys):
    """Two defaults, both observable: today, and a hard error on a path that is not a vault.

    The bad-vault case is distinguished from an empty day on purpose. A path that is
    not a directory is a typo or a missing mount, and answering "no notes" for it would
    read as "the agent learned nothing there" -- a confident, wrong answer instead of a
    refusal. It also never creates the path: a read-only report must not mutate the
    filesystem it was pointed at.
    """
    assert digest.main(["--vault", str(vault), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["date"] == date.today().isoformat()

    missing = str(vault.parent / "not-there")
    assert digest.main(["--vault", missing]) == 2
    assert "not a directory" in capsys.readouterr().err
    assert not os.path.exists(missing)


def test_a_topic_with_no_stub_is_reported_as_touched_with_zero_backlinks(vault):
    """A touched topic is reported even when its stub is missing.

    "Touched today" is decided by the day's notes, not by whether a file happens to
    exist, so a stub that was deleted -- or a topic the writer dropped because it had no
    usable filename -- must not read as "this topic was never filed". The count is 0 and
    the stub path is empty, which are facts; ``backlinks: 0`` next to a real stub is
    not, and the two are kept distinguishable.
    """
    _plant(vault, f"{DAY} - a", summary="- A.", topics=("Cats",))
    _plant(vault, f"{DAY} - b", summary="- B.", topics=("Dogs",))
    _plant_topic(vault, "Dogs", (f"{DAY} - b",))

    tallies = {t.topic: t for t in digest.build_digest(DAY, vault_root=vault).topics}
    assert tallies["Dogs"].backlinks == 1
    assert tallies["Dogs"].stub == f"{second_brain.TOPICS_DIR}/dogs.md"
    assert tallies["Cats"].backlinks == 0
    assert tallies["Cats"].stub == ""
