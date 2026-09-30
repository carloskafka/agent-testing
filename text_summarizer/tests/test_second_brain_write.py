"""Untrusted-input tests for the vault writer: path containment and YAML quoting.

``save_summary_to_second_brain`` takes three model-supplied strings and turns two of
them into filenames and the third into note content. Model output is attacker-
influenced -- anything the user pasted ends up in the title, and a jailbreak ends
up in the topics -- so these tests treat every argument as hostile and assert on
the filesystem rather than on the return value.

Two live defects are pinned here, both reproduced before the fix:

* **B1, path traversal.** ``topics`` was split on ``,``, stripped, and joined
  straight into a path with no sanitiser, so a traversing topic created a file
  outside the vault entirely. ``title`` was already passed through ``_slug``, but
  ``_slug`` removed punctuation rather than allowing characters, so it let ``..``
  and separators through too.
* **B2, frontmatter injection.** The frontmatter was built by unquoted
  interpolation, so a title containing a newline closed the ``---`` block early
  and the rest of it was read as new YAML keys.

The two rules that fix them are independent and both are tested: the *name* is
reduced to a filename allowlist (:func:`second_brain._safe_name`), and every write
is confined by an assertion that would fire if that allowlist ever regressed
(:func:`second_brain._assert_in_vault`). The ``## Related`` wikilinks are the one
place the readable original is deliberately kept, because that is the copy a
person clicks; ``test_a_topic_filename_is_slugged_but_the_link_is_not`` pins that
so the exception cannot quietly become the rule.

Notes are parsed back with ``yaml.safe_load`` rather than pattern-matched, because
the claim under test is that a reader can still read the note. ``PyYAML`` is a
hard dependency of ``google-adk`` (``pyyaml>=6.0.2``), not an extra added here.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest
import yaml
from google.genai import types
from text_summarizer import second_brain
from text_summarizer.second_brain import save_summary_to_second_brain


def _point_vault_at(monkeypatch, tmp_path, name: str = "ck") -> Path:
    """Repoint the vault at ``tmp_path/name`` and return it.

    ``second_brain.VAULT_ROOT`` is resolved at import time, so setting the env var
    alone would not move it; both are set. ``VAULT_NAME`` is pinned too --
    ``sources.resolve_vault_name`` prefers ``basename(vault_root)``, and the child
    is named ``ck`` so the recorded provenance is deterministic.
    """
    vault = tmp_path / name
    (vault / "Second Brain").mkdir(parents=True)
    monkeypatch.setenv("SECOND_BRAIN_VAULT", str(vault))
    monkeypatch.setenv("VAULT_NAME", "ck")
    monkeypatch.setattr(second_brain, "VAULT_ROOT", str(vault))
    return vault


def _note_files(vault: Path) -> list[Path]:
    return sorted((vault / "Second Brain").glob("*.md"))


def _only_note(vault: Path) -> Path:
    notes = _note_files(vault)
    assert len(notes) == 1, f"expected exactly one note, found {[p.name for p in notes]}"
    return notes[0]


def _stub_files(vault: Path) -> list[Path]:
    return sorted((vault / "Topics").glob("*.md"))


def _frontmatter_of(note: Path) -> tuple[dict, str]:
    """Parse a note's frontmatter and return it with the body after it.

    Uses the module's own :func:`second_brain._split_frontmatter` on purpose: it is
    the reader the cache uses, so a note that this cannot parse is a note the cache
    cannot replay. Deliberately strict -- if an unquoted value can close the block
    early, the B2 defect, the injected remainder lands in "body" and the parse gains
    keys, both of which the callers assert against.
    """
    text = note.read_text(encoding="utf-8")
    frontmatter_text, body = second_brain._split_frontmatter(text)
    assert frontmatter_text, f"note has no readable frontmatter: {text[:60]!r}"
    return yaml.safe_load(frontmatter_text), body


def _files_below(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def _stamp(paths: list[Path]) -> dict[Path, int | None]:
    """Record whether each path exists, and its mtime if it does.

    A bare ``not path.exists()`` is the wrong shape for this suite. An escape lands
    *outside* ``tmp_path``, in a directory pytest reuses between runs, so a file left
    there once by an unsanitised run would fail every later run for a bug that is
    already fixed. Snapshotting immediately before the call asserts what is actually
    meant -- *this* call created and touched nothing there -- and still fails the
    moment the code under test does write there.
    """
    return {p: (p.stat().st_mtime_ns if p.exists() else None) for p in paths}


def _assert_untouched(snapshot: dict[Path, int | None], what: str) -> None:
    for path, before in snapshot.items():
        if before is None:
            assert not path.exists(), f"{what} wrote outside the vault: {path}"
        else:
            assert path.exists(), f"{what} deleted {path}"
            assert path.stat().st_mtime_ns == before, f"{what} rewrote {path}"


def test_a_traversing_topic_cannot_escape_the_vault(monkeypatch, tmp_path):
    """B1: the exact payload that escaped before, asserted at the path it escaped to.

    The old code joined ``VAULT_ROOT/Topics/<topic>.md`` with the topic verbatim, so
    this string landed outside the vault entirely. Two things are asserted, because
    either alone is satisfiable by accident: the escape destination is not written,
    *and* the write is still confined rather than merely skipped -- "the file is not
    there" is also what a dropped write looks like.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)
    payload = "../../../../tmp/PWNED"
    escaped = [Path(os.path.normpath(vault / "Topics" / f"{payload}.md"))]
    # Guards against a vacuous pass: if this payload ever stopped resolving outside
    # the vault, the assertions below would hold for the wrong reason.
    assert not escaped[0].resolve().is_relative_to(vault.resolve())
    before = _stamp(escaped)

    save_summary_to_second_brain(
        title="Traversal Probe", summary_content="- x", topics=payload
    )

    _assert_untouched(before, "topic")
    assert [p.name for p in _stub_files(vault)] == ["tmp-pwned.md"]
    for written in _files_below(tmp_path):
        assert written.resolve().is_relative_to(vault.resolve()), written


def test_a_topic_filename_is_slugged_but_the_link_is_not(monkeypatch, tmp_path):
    """The one place the readable original survives: the link and the stub heading.

    The file on disk is named for the filesystem, the body is written for a person,
    and the model chose the string -- so the two are sanitised differently on
    purpose. This is the exception that keeps the exception from becoming the rule:
    a topic containing a separator must never reappear in a path.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)

    save_summary_to_second_brain(
        title="Slug Probe", summary_content="- x", topics="Mobile App, a/b"
    )

    assert [p.name for p in _stub_files(vault)] == ["a-b.md", "mobile-app.md"]
    body = _only_note(vault).read_text(encoding="utf-8")
    assert "[[Mobile App]]" in body
    assert "[[a/b]]" in body
    # The stub's heading is the human-readable copy, not the slug.
    assert "# a/b" in (vault / "Topics" / "a-b.md").read_text(encoding="utf-8")
    assert "# Mobile App" in (vault / "Topics" / "mobile-app.md").read_text(encoding="utf-8")


def test_a_title_with_a_newline_cannot_break_the_frontmatter(monkeypatch, tmp_path):
    """B2: a newline in a title used to close the ``---`` block and add YAML keys."""
    vault = _point_vault_at(monkeypatch, tmp_path)
    hostile = "Hostile\ninjected: yes\n---\n\n## Owned\nsecond block"

    save_summary_to_second_brain(
        title=hostile, summary_content="- x", topics="Dogs"
    )

    note = _only_note(vault)
    frontmatter, body = _frontmatter_of(note)
    assert "injected" not in frontmatter, frontmatter
    assert frontmatter["aliases"] == [hostile]
    assert frontmatter["tags"] == ["Dogs"]
    # The value is one quoted scalar: the raw frontmatter's alias entry is a single
    # line, so the newlines are escapes *inside* it rather than line breaks, and the
    # body after the closing fence is still the summary the tool was given.
    frontmatter_text, _ = second_brain._split_frontmatter(note.read_text(encoding="utf-8"))
    alias_entry = frontmatter_text.split("aliases:\n", 1)[1]
    assert alias_entry.count("\n") == 1, alias_entry
    assert body.lstrip().startswith("- x\n\n## Related")


def test_the_cache_replays_a_note_whose_title_contains_a_fence(monkeypatch, tmp_path):
    """The same injection, read back.

    A ``^---.*?---`` strip stops at the first ``---`` it finds, and a quoted scalar
    can contain that string, so the cache would return the tail of the note's own
    frontmatter as the "summary" -- served to the user with no model call. The reader
    has to recognise the fence as a line, not as a substring, or fixing the write
    side only moves the failure.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)
    prompt = "Summarize: a hostile title."

    class _Context:
        """The shape ``user_text_from_context`` reads: a ``Content``, not a list of parts."""

        user_content = types.Content(role="user", parts=[types.Part(text=prompt)])

        class _Session:
            events: list = []

        session = _Session()

    save_summary_to_second_brain(
        title="Title\n---\n## Injected\nnot a summary",
        summary_content="- the real answer",
        topics="Dogs",
        tool_context=_Context(),
    )

    assert _only_note(vault).read_text(encoding="utf-8").count("---") == 3
    assert second_brain.find_cached_summary(prompt) == "- the real answer"


def test_yaml_values_with_quotes_and_colons_round_trip(monkeypatch, tmp_path):
    """Quoting has to be a real quoting, not escaping of the characters seen so far.

    ``title``/``topics`` are quoted with ``json.dumps``, which is why ``"``, ``:``,
    ``#``, a tab and a trailing backslash all survive: they come back as the same
    Python string a YAML reader produces, not as a mangled approximation of it.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)
    title = 'He said "hi": not # a comment \\ end\ttabbed'
    topic = 'Q&A: "quoted" #50%'

    save_summary_to_second_brain(
        title=title, summary_content="- x", topics=topic
    )

    frontmatter, _ = _frontmatter_of(_only_note(vault))
    assert frontmatter["aliases"] == [title]
    assert frontmatter["tags"] == [topic]
    # A round trip through YAML's own reader is the claim, not a substring match.
    assert yaml.safe_load(yaml.safe_dump(frontmatter)) == frontmatter


#: Shapes that matter: separators, bare parent references, backslashes, a NUL byte, a
#: name no filesystem accepts, and two that sanitise away to nothing.
HOSTILE_PAYLOADS = [
    "../../../../tmp/PWNED",
    "..",
    ".",
    "...",
    "/etc/passwd",
    "a/../../b",
    "..\\..\\windows",
    "a\0b",
    "topic with spaces/../..",
    "-",
    "--",
    "x" * 500,
    "🏴‍☠️",
]


def _would_have_escaped(vault: Path, payload: str) -> list[Path]:
    """The paths the *unsanitised* code would have chosen for this payload.

    Scoped to ``tmp_path`` and below by design: an escape lands outside the vault,
    which is outside ``tmp_path`` too, so a test that only walks ``tmp_path`` walks
    straight past the bug it is meant to catch. ``os.path.normpath`` collapses the
    ``..`` the way the old ``os.path.join`` did, so this is the real destination.
    """
    brain = vault / second_brain.BRAIN_DIR
    return [
        Path(os.path.normpath(brain / f"{payload}.md")),
        Path(os.path.normpath(vault / second_brain.TOPICS_DIR / f"{payload}.md")),
    ]


@pytest.mark.parametrize("payload", HOSTILE_PAYLOADS)
def test_every_written_path_is_inside_the_vault_root(monkeypatch, tmp_path, payload):
    """Whatever the model puts in a topic, every file created is inside the vault.

    Two assertions, because either alone is satisfiable by accident: the path the old
    code would have written must not exist (that is where an escape *goes*), and
    nothing may appear anywhere else under the test's own tree either. The stub is
    not asserted to exist -- some of these have no usable file name and are dropped,
    and the invariant here is containment, not liveness.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)
    before = _stamp(_would_have_escaped(vault, payload))

    save_summary_to_second_brain(
        title=f"Probe {payload}", summary_content="- x", topics=payload
    )

    _assert_untouched(before, "topic")
    for written in _files_below(tmp_path):
        assert written.resolve().is_relative_to(vault.resolve()), written


@pytest.mark.parametrize("payload", HOSTILE_PAYLOADS)
def test_a_hostile_title_cannot_escape_either(monkeypatch, tmp_path, payload):
    """Same containment property for ``title``, which names the note itself.

    ``title`` always went through a slugger, so this is the regression guard for that
    slugger having become a *general* one: a title that sanitises to nothing must fall
    back to the fixed ``<date> - summary`` name rather than to ``<date> - .md`` or an
    empty path component.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)
    before = _stamp(_would_have_escaped(vault, payload))

    save_summary_to_second_brain(
        title=payload, summary_content="- x", topics="Dogs"
    )

    _assert_untouched(before, "title")
    for written in _files_below(tmp_path):
        assert written.resolve().is_relative_to(vault.resolve()), written
    note = _only_note(vault)
    assert note.stem.startswith(f"{date.today().isoformat()} - ")
    assert note.stem.split(" - ", 1)[1] == (second_brain._safe_name(payload) or "summary")
    assert "/" not in note.name


def test_the_containment_assert_refuses_a_path_outside_the_vault(monkeypatch, tmp_path):
    """The second line of defence, tested directly.

    :func:`_assert_in_vault` is unreachable while the sanitiser holds, which is
    exactly why it needs its own test: it is the thing that would still be there if
    the sanitiser regressed, and an untested guard that only runs on a bug is a
    guard nobody knows works.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)
    inside = vault / "Topics" / "ok.md"
    escaped = vault / ".." / "outside.md"

    second_brain._assert_in_vault(str(inside))  # no raise
    with pytest.raises(ValueError, match="outside the vault"):
        second_brain._assert_in_vault(str(escaped))
    # A sibling whose name merely starts the same way is not a match.
    with pytest.raises(ValueError, match="outside the vault"):
        second_brain._assert_in_vault(f"{vault}-sibling/note.md")


def test_a_write_outside_the_vault_raises_instead_of_landing(monkeypatch, tmp_path):
    """``_write`` is the only place a path becomes a syscall, so that is where the check is.

    Testing the hook rather than the intent: any future writer in this module is
    confined without having to remember anything, and this proves it rather than
    assuming it.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)
    target = tmp_path.parent / "escaped.md"

    with pytest.raises(ValueError, match="outside the vault"):
        second_brain._write(str(target), "- x")

    assert not target.exists()
    assert (vault / "Second Brain").exists()


def test_a_note_written_twice_on_the_same_day_is_updated_not_duplicated(monkeypatch, tmp_path):
    """Two saves on one day are one file, and the return string says so truthfully.

    This is also what the dead ``if note_title not in index_body or index_body``
    branch got wrong: its right-hand operand is a non-empty string the function had
    just written, so the branch was always true and the list it guarded was always
    ``[note_path]``. It was never read, so the honest fix is to drop it -- and the
    "Created topic stubs" line, which *is* read, is asserted below to be empty on
    the second call, because the stub already existed.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)

    first = save_summary_to_second_brain(
        title="Same Day", summary_content="- first", topics="Dogs"
    )
    second = save_summary_to_second_brain(
        title="Same Day", summary_content="- second", topics="Dogs"
    )

    assert _only_note(vault).read_text(encoding="utf-8").count("- second") == 1
    assert "- first" not in _only_note(vault).read_text(encoding="utf-8")
    index = (vault / f"{second_brain.INDEX_NAME}.md").read_text(encoding="utf-8")
    assert index.count(f"- [[{_only_note(vault).stem}]]") == 1
    stub = (vault / "Topics" / "dogs.md").read_text(encoding="utf-8")
    assert stub.count(f"- [[{_only_note(vault).stem}]]") == 1
    # Created-vs-updated is reported where it is actually knowable: the stub already
    # existed, so the second call says so.
    assert "Created topic stubs: Dogs" in first
    assert "Created topic stubs: none" in second
    # The note line is byte-identical on both calls, and it is the *only* difference
    # between the two results. That is the point of the dead branch's removal: it
    # could not have changed either string, so nothing lost by deleting it.
    # `strict=True` because the very next line asserts the two are the same length:
    # a zip that silently stopped at the shorter one would report "no differences"
    # and turn this into the vacuous assertion it replaced.
    differing = [
        a
        for a, b in zip(first.splitlines(), second.splitlines(), strict=True)
        if a != b
    ]
    assert len(first.splitlines()) == len(second.splitlines()) == 4
    assert len(differing) == 1 and differing[0].startswith("Created topic stubs:")


def test_a_dropped_topic_is_reported_rather_than_silently_missed(monkeypatch, tmp_path):
    """A topic with no usable file name is dropped **and named in the result**.

    The behaviour, chosen so that degradation is loud in both directions: it is not
    written (a stub called ``...md`` is indistinguishable from a real topic later
    and is not a thing anyone can link to), and it is not silently skipped either
    (a topic that vanished would read to the agent as "filed" when nothing was).
    ``Linked topics`` counts only the topics that were really filed, and
    ``Dropped topics`` carries the rest verbatim so the failure is diagnosable from
    the tool's own output.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)

    result = save_summary_to_second_brain(
        title="Drop Probe", summary_content="- x", topics="Dogs, .., ., ///, -"
    )

    assert "Linked topics (1): Dogs" in result
    assert "Dropped topics (4): .., ., ///, -" in result
    assert [p.name for p in _stub_files(vault)] == ["dogs.md"]
    body = _only_note(vault).read_text(encoding="utf-8")
    assert "[[Dogs]]" in body
    for dropped in ("[[..]]", "[[.]]", "[[///]]", "[[-]]"):
        assert dropped not in body
    # The dropped topic leaves no trace in the frontmatter either.
    assert _frontmatter_of(_only_note(vault))[0]["tags"] == ["Dogs"]


def test_no_dropped_line_when_every_topic_is_usable(monkeypatch, tmp_path):
    """The normal path keeps the original four-line result shape."""
    _point_vault_at(monkeypatch, tmp_path)

    result = save_summary_to_second_brain(
        title="Normal", summary_content="- x", topics="Dogs, Checkout"
    )

    assert "Dropped topics" not in result
    assert result.count("\n") == 3


def test_wikilink_syntax_in_a_topic_cannot_break_out_of_the_link(monkeypatch, tmp_path):
    """``[[``/``]]``/``|`` in a topic must not re-shape the ``## Related`` section.

    A wikilink target is markdown, not YAML, so it is *not* JSON-quoted -- doing
    that would emit ``[[\"Dogs\"]]`` and break every link in the vault while securing
    none. Obsidian's own link metacharacters are replaced instead, which is a no-op
    for every ordinary topic.
    """
    vault = _point_vault_at(monkeypatch, tmp_path)

    save_summary_to_second_brain(
        title="Link Probe", summary_content="- x", topics="Dogs]]\n- [[Injected"
    )

    related = _only_note(vault).read_text(encoding="utf-8").split("## Related\n", 1)[1]
    # The bracket and the newline are gone, so the topic cannot close the link and
    # open another one; what is left is a single (mangled, but inert) target.
    assert related.count("[[") == 1
    assert related.strip() == "- [[Dogs -  Injected]]"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ""),
        (".", ""),
        ("..", ""),
        ("...", ""),
        ("/", ""),
        ("///", ""),
        ("-", ""),
        ("   ", ""),
        (None, ""),
        ("Dogs", "dogs"),
        ("Mobile App", "mobile-app"),
        ("a/b", "a-b"),
        ("a\\b", "a-b"),
        ("../../../../tmp/PWNED", "tmp-pwned"),
        ("../../outside/PWNED", "outside-pwned"),
        ("..\\..\\windows", "windows"),
        ("Customer Feedback 2026-09-25", "customer-feedback-2026-09-25"),
        ("a..b", "a-b"),
        ("5/3/2026", "5-3-2026"),
        ("trailing---", "trailing"),
        ("x" * 80, "x" * 60),
    ],
)
def test_safe_name_reduces_to_a_filename_allowlist(value, expected):
    """The rules, one case each, including every shape that must reduce to ``""``.

    A topic that reduces to ``""`` is dropped by :func:`_split_topics`; a *name*
    that reduces to ``""`` would be an empty path component, so the two matter
    separately and both are pinned.
    """
    assert second_brain._safe_name(value) == expected


def test_safe_name_survives_a_non_string_argument():
    """A model that emits ``topics: 5`` must not raise a TypeError into the turn.

    ADK does not coerce JSON arguments to the declared annotation, so the
    sanitiser -- the first thing every argument now passes through -- has to accept
    whatever arrived.
    """
    assert second_brain._safe_name(42) == "42"
    assert second_brain._safe_name(3.5) == "3-5"
