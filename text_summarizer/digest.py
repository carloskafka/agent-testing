"""Digest mode: what the agent learned on a given day, read straight off the vault.

Why this reads the vault instead of asking the agent
----------------------------------------------------
Every fact a digest reports was already written down. ``save_summary_to_second_brain``
persisted the summary, its topics and its provenance; ``log_conversation`` persisted
the exchange. Asking the agent to re-read and re-narrate its own output would cost a
model call for information that is sitting on disk as plain text -- and this project
spends real quota on every turn, with a free-tier fallback that is throttled often
enough to be a poor foundation for a reporting command. Four consequences follow, and
they are the reason this is a filesystem reader and not a prompt:

* **no quota, no latency.** The whole digest is a directory listing and a handful of
  reads. Measured on a 50-note vault it is the same order as the cache lookup that
  already runs on the first model call of every turn.
* **it works when the model does not.** The primary backend is rate-limited, the
  fallback is a *free* tier that returns HTTP 429 often enough that a Gemini quota
  error is as likely to end in a rate-limit error as in a served turn, and the
  provider can be mid-outage. A report on what was already learned cannot depend on
  any of that.
* **it is deterministic.** No sampling, no temperature, no re-interpretation. Two runs
  over an unchanged vault produce byte-identical output, which is what makes the
  ``--json`` form usable as the input to a script and the two renderings diffable.
  The alternative is a digest whose wording drifts on every invocation and that nobody
  can assert anything about.
* **it cannot hallucinate a fact about the vault**, because it never leaves the vault.

The converse is the one real limitation, and it is worth stating: this reports what was
*recorded*, not what was true. A note whose body is wrong is reported faithfully wrong.
The provenance fields are what let a reader judge it -- which is also why they are
never invented here (below).

Provenance is omitted, never invented
-------------------------------------
A note's ``generated_by_model`` / ``generated_in_vault`` are reported **only if the
frontmatter actually recorded them**. Notes written before those fields existed carry
neither, and the honest rendering of that is the absence of a field -- exactly the
convention ``save_summary_to_second_brain`` follows when it writes, and the same
principle that makes ``sources.render_sources`` emit :data:`sources.UNKNOWN` rather
than guess. Back-filling from whichever model happens to be serving now would make the
digest assert something the vault never said. The digest-level vault *name* is the one
place a value has to be shown even when unresolvable, and it reuses ``sources.UNKNOWN``
for that case rather than inventing a directory name.

Nothing here is optional
------------------------
This module imports only the standard library plus :mod:`text_summarizer.sources` and
:mod:`text_summarizer.second_brain`, both of which are themselves standard-library
only. It never imports ``google.adk`` or ``google.genai``, and a digest therefore
costs nothing to run in a context that cannot reach a model. That is a test
(``test_the_import_graph_reaches_neither_adk_nor_genai``), not a claim: the package
``__init__`` exports ``root_agent`` and so *does* pull both in, which is why that test
measures this module's own graph with the parent package stubbed out.

One consequence of those constants is worth stating, because it is a limitation rather
than a choice: the package initializer prints, and ``second_brain`` prints when it
resolves a vault it cannot create. Both land on stdout, and both happen when the
*parent package* is imported -- which is before this module's body runs, so no amount of
care in here can keep them off the data stream. ``--json`` is therefore piped into
something that skips the first lines, and the durable fix is for
:mod:`text_summarizer.observability` and :mod:`text_summarizer.second_brain` to print
diagnostics to stderr, which is where a diagnostic belongs. Neither file belongs to
this module, so neither is touched here.

Layout, and the fields read
---------------------------
:func:`build_digest` reads what :mod:`text_summarizer.second_brain` writes and nothing
else: the dated notes under ``Second Brain/``, the ``## Backlinks`` stubs under
``Topics/``, the hub note ``Second Brain Index.md`` and the day's chat log. It reuses
that module's directory constants and its ``_split_frontmatter`` reader rather than
re-declaring the layout, because two descriptions of one on-disk format is how a
refactor of the writer silently stops being reflected in its reader.

Malformed input degrades, it does not drop
-----------------------------------------
A note with broken or missing frontmatter still appears, carrying whatever fields could
be read, and a ``has_frontmatter`` flag says which case it was. That is the difference
between "this note has no recorded provenance" and "this note was not found", and a
digest that silently dropped damaged notes would report the same day as a quieter one.
Because the fields are read one key at a time, a single unparseable ``tags:`` line
costs the topics and nothing else.

A note belongs to a day if **either** its filename prefix (``<date> - <slug>.md``, what
the writer names files) **or** its frontmatter ``date:`` says so, so a note a human
renamed in Obsidian is not lost. A note with neither -- dropped into ``Second Brain/``
by hand -- belongs to no day, and saying so requires inventing a date; it is reported
on the day that is asked for only when it is genuinely ambiguous, and otherwise left
out. The two can disagree, and the reported ``date`` is the note's own declared one, so
a discrepancy is visible rather than smoothed over.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date

from . import second_brain, sources
from .second_brain import (
    BRAIN_DIR,
    CHAT_LOG_DIR,
    GENERATED_BY_MODEL_KEY,
    GENERATED_IN_VAULT_KEY,
    INDEX_NAME,
    TOPICS_DIR,
    _safe_name,
    _split_frontmatter,
)
from .sources import UNKNOWN, strip_sources_block

#: ``_safe_name`` is the writer's own sanitiser, so a topic is looked up under the
#: filename the writer would have given it rather than one guessed here.
#: ``_split_frontmatter`` is the writer's own frontmatter reader, so a note this cannot
#: parse is a note the vault cache cannot replay either -- one definition of the
#: format, not two. ``strip_sources_block`` is the writer's own Sources-block remover,
#: so a digest line can never show the ``[obsidian][ck][...][...]`` markers that mean
#: nothing outside a rendered answer.

__all__ = [
    "ChatEntry",
    "Digest",
    "DigestNote",
    "TopicTally",
    "build_digest",
    "default_vault_root",
    "main",
    "render_md",
    "render_text",
    "to_dict",
    "to_json",
]

DEFAULT_FORMAT = "text"
FORMATS = ("text", "md")

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+\S")
#: The writer's own section heading, matched loosely (any level, any case, optional
#: colon) because a human can retype it. Only this heading ends the summary: the legacy
#: ``## From the vault`` block in pre-provenance notes is *not* it, and those lines are
#: the note's own text. ``MULTILINE`` is load-bearing -- the heading is mid-body.
_RELATED_RE = re.compile(
    r"^[ \t]{0,3}#{1,6}[ \t]*Related[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE
)
#: The topic stub's own heading, which is *not* ``## Related`` -- these are two
#: different files written by two different lines of ``save_summary_to_second_brain``.
_BACKLINKS_RE = re.compile(
    r"^[ \t]{0,3}#{1,6}[ \t]*Backlinks[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE
)
_FENCE_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)[ \t]*:(.*)$")
_ITEM_RE = re.compile(r"^[ \t]*-(?:\s+(.*)|[ \t]*)$")
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\n]+?)\]\]")
_HUB_LINK_RE = re.compile(
    r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+\[\[([^\[\]\n]+?)\]\][ \t]*$", re.MULTILINE
)
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_CHAT_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*(\d{1,2}:\d{2})[ \t]*$", re.MULTILINE)
#: The writer emits ``**User:**`` and ``**Agent:**`` -- the bold markers wrap the colon
#: too, which a reader anchored on the front of the line alone does not match. Leading
#: and trailing ``*`` are both optional so a hand-edited log still parses.
_CHAT_SPEAKER_RE = re.compile(
    r"^[ \t]{0,3}\**[ \t]*(User|Agent)[ \t]*\**[ \t]*:[ \t]*\**[ \t]*$", re.IGNORECASE
)


# --- the vault root -------------------------------------------------------------


def default_vault_root() -> str:
    """The active vault, read from :mod:`second_brain` at call time.

    Deliberately an attribute lookup rather than ``from .second_brain import
    VAULT_ROOT``. ``VAULT_ROOT`` is resolved at *import* time, so binding it here would
    make this module a third copy of the same path that tests have to repoint
    individually -- and :file:`AGENTS.md` records a real defect caused by exactly that
    duplication, where one call site read a different vault than the other and a note's
    frontmatter said ``ck`` while the answer said ``vaults``. Reading the one binding
    means there is nothing here to keep in sync, and ``--vault`` overrides it per call.
    """
    return second_brain.VAULT_ROOT


def _resolve_root(vault_root: str | None) -> str:
    root = vault_root if vault_root is not None else default_vault_root()
    return os.path.abspath(os.path.expanduser(str(root or "")))


# --- frontmatter reading --------------------------------------------------------


def _unquote(value: str) -> str:
    """Strip one layer of YAML quoting, tolerating a scalar that is not quoted.

    The writer quotes with ``json.dumps`` (:func:`second_brain._yaml_str`), so the
    double-quoted form is unescaped exactly; the single-quoted form is YAML's own and
    has no escapes, so the quotes simply come off. A value that is not quoted at all is
    returned as-is, which is what a hand-written note looks like and is why this is not
    a YAML parser: a scalar that fails to decode is still a scalar a human typed.
    """
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            return json.loads(text)
        except ValueError:
            return text[1:-1].strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'").strip()
    return text


def _split_inline_list(value: str) -> list[str] | None:
    """Parse ``[a, "b, c"]`` into ``["a", "b, c"]``, or ``None`` if it is not a list.

    The comma split has to be quote-aware. The writer emits topics as *block* items
    (``  - "Dogs, cats and birds"``), which are comma-safe, but a hand-edited note with
    the inline spelling would otherwise yield the single topic ``Dogs`` and the single
    topic ``cats and birds`` -- two invented topics in place of one real one. Returning
    ``None`` for an unterminated ``[`` is deliberate: reporting ``"[unclosed"`` as a
    topic would be inventing one too, and the caller falls back to the ``## Related``
    section instead.
    """
    text = (value or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return []

    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in inner:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char == ",":
            items.append("".join(current))
            current = []
            continue
        current.append(char)
    if quote is not None:
        return None
    items.append("".join(current))
    return [_unquote(item) for item in items if _unquote(item)]


def _as_list(value: object) -> list[str]:
    """A list-typed field, however it was spelled.

    A note may write ``tags: Dogs`` (one topic) as readily as the block form, and
    iterating a bare string would yield one "topic" per character. Anything that is
    neither a string nor a sequence is unresolved rather than iterated, which is the
    same rule the rest of the reader follows.
    """
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _frontmatter_fields(frontmatter: str, wanted: tuple[str, ...]) -> dict[str, object]:
    """Read a named subset of a frontmatter block, one key at a time.

    A real YAML parser would be shorter and would be wrong here for a specific reason:
    one malformed line makes the *whole* document unparseable, so a note whose ``tags:``
    is damaged would also lose its ``generated_by_model`` -- the exact "one broken field
    costs all the others" behaviour this digest is required not to have. Reading per
    key means an unreadable field is simply absent.

    Only the two shapes the writer emits are understood, plus the inline list Obsidian
    itself accepts. Anything else is left unresolved rather than guessed.
    """
    fields: dict[str, object] = {}
    lines = (frontmatter or "").splitlines()
    index = 0
    while index < len(lines):
        match = _FENCE_KEY_RE.match(lines[index])
        index += 1
        if not match:
            continue
        key, raw = match.group(1), match.group(2).strip()
        if key not in wanted or key in fields:
            continue
        if raw:
            inline = _split_inline_list(raw) if raw.startswith("[") else None
            if inline is not None:
                fields[key] = inline
            elif not raw.startswith("["):
                fields[key] = _unquote(raw)
            continue

        items: list[str] = []
        while index < len(lines):
            item = _ITEM_RE.match(lines[index])
            if not item:
                break
            value = _unquote(item.group(1) or "")
            if value:
                items.append(value)
            index += 1
        fields[key] = items
    return fields


# --- the vault layout -----------------------------------------------------------


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _note_date_from_name(name: str) -> str:
    match = _ISO_DATE_RE.match(name[:10]) if len(name) >= 10 else None
    return match.group(0) if match else ""


def _summary_of(body: str) -> tuple[str, list[str]]:
    """The note's own summary, and the list-item lines within it.

    The ``## Related`` heading is where the writer puts the topic links, which are
    reported as ``topics``; leaving them in would print every topic twice. What is left
    is the note's content whatever shape it has -- bullets, prose, or the closing
    sentence the model likes to add -- and it is *not* reduced to its bullets alone,
    because a note whose content is prose has no bullets to report and dropping them
    would report it as empty. The bullets are returned alongside, verbatim with their
    markers, for callers that want to count or re-render them.
    """
    cut = _RELATED_RE.search(body)
    text = body[: cut.start()] if cut else body
    text = strip_sources_block(text).strip()
    return text, [line.strip() for line in text.splitlines() if _BULLET_RE.match(line)]


def _related_topics(body: str) -> list[str]:
    """Topic wikilinks from a note's ``## Related`` section.

    The fallback for a note whose ``tags:`` did not parse. The section is the writer's
    own rendering of the same topics, so reading it recovers the field from the one
    place it is written twice -- which is what keeps a note with damaged frontmatter
    from also reading as untopiced.
    """
    cut = _RELATED_RE.search(body)
    if not cut:
        return []
    seen: list[str] = []
    for match in _WIKILINK_RE.finditer(body[cut.end() :]):
        target = match.group(1).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


@dataclass(frozen=True)
class DigestNote:
    """One note written on the digested day."""

    title: str
    alias: str
    path: str
    date: str
    summary: str
    bullets: list[str]
    topics: list[str]
    provenance: dict[str, str]
    index_position: int | None
    has_frontmatter: bool
    #: The note's recorded ``source_fingerprint`` -- the key
    #: ``second_brain.source_fingerprint`` produces and the vault cache is looked up
    #: by. Carried because ``--json`` is how a script groups the day's notes by what
    #: was actually asked, and re-reading the file to recover it would be absurd; it is
    #: simply not rendered, because 64 hex characters help nobody reading a digest.
    fingerprint: str


def _load_note(path: str, root: str, day: str, index: dict[str, int]) -> DigestNote:
    text = _read(path)
    frontmatter, body = _split_frontmatter(text)
    title = os.path.splitext(os.path.basename(path))[0]

    fields = _frontmatter_fields(
        frontmatter,
        (
            "tags",
            "aliases",
            "alias",
            "date",
            GENERATED_BY_MODEL_KEY,
            GENERATED_IN_VAULT_KEY,
            "source_fingerprint",
        ),
    )

    name_date = _note_date_from_name(title)
    declared = str(fields.get("date") or "").strip()
    note_date = declared if _ISO_DATE_RE.match(declared) else name_date

    summary, bullets = _summary_of(body)

    topics = _as_list(fields.get("tags")) or _related_topics(body)
    aliases = _as_list(fields.get("aliases")) or _as_list(fields.get("alias"))

    provenance: dict[str, str] = {}
    for key in (GENERATED_BY_MODEL_KEY, GENERATED_IN_VAULT_KEY):
        value = _unquote(str(fields.get(key) or ""))
        if value:
            provenance[key] = value

    fingerprint = _unquote(str(fields.get("source_fingerprint") or "")).strip()
    if not _FINGERPRINT_RE.match(fingerprint):
        # A value that is not the 64-hex shape the writer emits is not a cache key,
        # and reporting it as one would let a script compare against a key that can
        # never match a lookup.
        fingerprint = ""

    return DigestNote(
        title=title,
        alias=aliases[0] if aliases else "",
        path=os.path.relpath(path, root),
        date=note_date,
        summary=summary,
        bullets=bullets,
        topics=topics,
        provenance=provenance,
        index_position=index.get(title),
        has_frontmatter=bool(frontmatter.strip()),
        fingerprint=fingerprint,
    )


def _hub_index(root: str) -> dict[str, int]:
    """Map each title the hub note lists to its 1-based position among those links.

    Positions are counted over the index's own link lines, not its raw line numbers, so
    the number means "the Nth note in the hub" -- which is the thing a reader compares
    against. A title listed twice keeps its first position, matching how Obsidian
    resolves a duplicate link to one note.
    """
    text = _read(os.path.join(root, f"{INDEX_NAME}.md"))
    return {
        match.group(1).strip(): position
        for position, match in enumerate(_HUB_LINK_RE.finditer(text), 1)
    }


def _topic_stub(root: str, topic: str) -> str:
    """The topic stub to count backlinks in, or ``""`` when there is none.

    Prefers the slugged name, which is what the current writer produces
    (:func:`second_brain._safe_name`), and falls back to the readable name that
    earlier versions wrote unslugged. A vault holding both -- which this one does, for
    topics like ``Space exploration`` -- is a duplicate left by that change; the
    slugged stub is the one current code maintains, so it is the one counted and the
    path is reported so a reader can check.
    """
    for name in (_safe_name(topic), str(topic or "").strip()):
        if not name:
            continue
        candidate = os.path.join(root, TOPICS_DIR, f"{name}.md")
        if os.path.isfile(candidate):
            return candidate
    return ""


def _backlink_count(path: str) -> int:
    """Wikilink entries under a topic stub's ``## Backlinks`` heading.

    Scoped to that section because the heading itself carries a ``#`` and a later
    section could hold links that are not backlinks; a stub with no such heading -- one
    a human wrote, or one from a writer that used a different name -- falls back to the
    whole file rather than to zero, which would report a well-linked topic as untouched.
    """
    text = _read(path)
    if not text.strip():
        return 0
    cut = _BACKLINKS_RE.search(text)
    body = text[cut.end() :] if cut else text
    return sum(1 for line in body.splitlines() if _WIKILINK_RE.search(line))


@dataclass(frozen=True)
class TopicTally:
    """A topic the day's notes touched, and how many notes link to its stub."""

    topic: str
    backlinks: int
    stub: str
    notes: list[str]


@dataclass(frozen=True)
class ChatEntry:
    """One logged user<->agent exchange."""

    time: str
    user: str
    agent: str


@dataclass(frozen=True)
class Digest:
    """Everything the day holds, already resolved and ready to render."""

    date: str
    vault_root: str
    vault_name: str
    vault_name_source: str
    notes: list[DigestNote]
    topics: list[TopicTally]
    chat: list[ChatEntry] | None


def _parse_chat(path: str) -> list[ChatEntry]:
    """Parse ``log_conversation``'s output into entries.

    The writer separates exchanges with a ``## HH:MM`` heading and labels the two sides
    ``**User:**`` / ``**Agent:**``. An entry whose labels are missing is still reported,
    with the side it did have -- a chat log the digest silently shortened would read as
    a quieter day, which is the one thing this command must never do. The Sources block
    is stripped from both sides: those lines are this project's own provenance markers
    (``[obsidian][ck][…][[…]]``) and mean nothing outside a rendered answer.
    """
    text = _read(path)
    if not text.strip():
        return []

    entries: list[ChatEntry] = []
    bounds = [(m.group(1), m.start(), m.end()) for m in _CHAT_HEADING_RE.finditer(text)]
    for position, (time, _, start) in enumerate(bounds):
        end = bounds[position + 1][1] if position + 1 < len(bounds) else len(text)
        block = text[start:end]

        sides: dict[str, list[str]] = {}
        current: str | None = None
        for line in block.splitlines():
            speaker = _CHAT_SPEAKER_RE.match(line)
            if speaker:
                current = speaker.group(1).lower()
                sides.setdefault(current, [])
                continue
            if current:
                sides[current].append(line)
        entries.append(
            ChatEntry(
                time=time,
                user=strip_sources_block("\n".join(sides.get("user", []))).strip(),
                agent=strip_sources_block("\n".join(sides.get("agent", []))).strip(),
            )
        )
    return entries


def build_digest(
    day: str | None = None,
    *,
    vault_root: str | None = None,
    include_chat: bool = False,
) -> Digest:
    """Read the vault and return everything it holds for ``day``.

    ``day`` is an ISO ``YYYY-MM-DD`` string and defaults to today. A day with no notes
    is a valid, empty :class:`Digest` rather than an error: "the agent learned nothing
    on Tuesday" is a fact worth being able to ask for, and a command that failed on it
    could not be used from a cron job that reports only when there is something to say.

    ``chat`` is ``None`` unless ``include_chat`` -- the chat log is a different kind of
    record from a summary (raw prompts and answers, not distilled findings), and putting
    it in the default output would bury the part that is a digest.
    """
    day = day or date.today().isoformat()
    root = _resolve_root(vault_root)
    identity = sources.resolve_vault_name(vault_root=root)

    notes: list[DigestNote] = []
    index = _hub_index(root)
    brain_dir = os.path.join(root, BRAIN_DIR)
    if os.path.isdir(brain_dir):
        for name in sorted(os.listdir(brain_dir)):
            if not name.endswith(".md") or name.startswith("."):
                continue
            path = os.path.join(brain_dir, name)
            if not os.path.isfile(path):
                continue
            note = _load_note(path, root, day, index)
            # Either source of the date counts, so a note renamed in Obsidian is not
            # lost and a note whose filename disagrees with its own frontmatter is
            # reported on the day it declares.
            if _note_date_from_name(name) == day or note.date == day:
                notes.append(note)

    # Sorted on the filename title, which is unique within one directory and always
    # present, unlike the alias. Sorting on the alias would make the order depend on
    # model-chosen text and could not break a tie.
    notes.sort(key=lambda note: (note.title.casefold(), note.title))

    touched: dict[str, list[str]] = {}
    for note in notes:
        for topic in note.topics:
            seen = touched.setdefault(topic, [])
            if note.title not in seen:
                seen.append(note.title)
    topics = []
    for topic in sorted(touched, key=lambda name: (name.casefold(), name)):
        stub = _topic_stub(root, topic)
        topics.append(
            TopicTally(
                topic=topic,
                backlinks=_backlink_count(stub) if stub else 0,
                stub=os.path.relpath(stub, root) if stub else "",
                notes=touched[topic],
            )
        )

    chat = _parse_chat(os.path.join(root, CHAT_LOG_DIR, f"{day}.md")) if include_chat else None

    return Digest(
        date=day,
        vault_root=root,
        # `resolve_vault_name` already answers with UNKNOWN rather than a guess; the
        # `or` is the belt-and-braces that keeps the field a non-empty string whatever
        # that helper is given.
        vault_name=identity.name or UNKNOWN,
        vault_name_source=identity.source,
        notes=notes,
        topics=topics,
        chat=chat,
    )


# --- machine-readable form ------------------------------------------------------


def to_dict(digest: Digest) -> dict:
    """The stable structure ``--json`` emits.

    Field order is the reading order, and every list is sorted (notes by title, topics
    case-insensitively, chat entries as they appear in the log). A consumer can
    therefore diff two digests of the same day and see only what actually changed, and
    an unresolvable provenance key is *absent* rather than ``null`` or ``"unknown"`` --
    the same omission the writer makes and the reason a pre-provenance note can never
    be mistaken for one that recorded a model.
    """
    return {
        "date": digest.date,
        "vault": {
            "root": digest.vault_root,
            "name": digest.vault_name,
            "name_source": digest.vault_name_source,
        },
        "note_count": len(digest.notes),
        "topic_count": len(digest.topics),
        "notes": [
            {
                "title": note.title,
                "alias": note.alias,
                "path": note.path,
                "date": note.date,
                "summary": note.summary,
                "bullets": note.bullets,
                "topics": note.topics,
                "provenance": dict(note.provenance),
                "fingerprint": note.fingerprint,
                "index_position": note.index_position,
                "has_frontmatter": note.has_frontmatter,
            }
            for note in digest.notes
        ],
        "topics": [
            {
                "topic": tally.topic,
                "backlinks": tally.backlinks,
                "stub": tally.stub,
                "notes": tally.notes,
            }
            for tally in digest.topics
        ],
        "chat": (
            None
            if digest.chat is None
            else [
                {"time": entry.time, "user": entry.user, "agent": entry.agent}
                for entry in digest.chat
            ]
        ),
    }


def to_json(digest: Digest) -> str:
    """The ``--json`` rendering, with a trailing newline for shell friendliness."""
    return json.dumps(to_dict(digest), indent=2, ensure_ascii=False) + "\n"


# --- the two human renderings ---------------------------------------------------


def _meta(note: DigestNote) -> str:
    """The provenance-and-position line, or ``""`` when the note says nothing.

    A note with no recorded provenance and no hub entry produces no line at all rather
    than a line reading ``model: -``. The absence is the information; a placeholder
    would invite a reader to treat "unknown" as a value.
    """
    parts: list[str] = []
    model = note.provenance.get(GENERATED_BY_MODEL_KEY, "")
    vault = note.provenance.get(GENERATED_IN_VAULT_KEY, "")
    if model:
        parts.append(f"model: {model}")
    if vault:
        parts.append(f"in vault: {vault}")
    if note.index_position is not None:
        parts.append(f"index #{note.index_position}")
    return " | ".join(parts)


def _md_provenance(note: DigestNote) -> str:
    """The provenance line for the note rendering, in the vault's own code style.

    Separate from :func:`_meta` because the two renderings label things differently:
    the terminal one has no ``**Provenance:**`` prefix, so it spells out ``model:`` and
    ``in vault:`` itself. The hub position is deliberately *not* included -- the note
    rendering gives it a line of its own, with a link, and printing it in both places
    would state it twice.
    """
    parts: list[str] = []
    model = note.provenance.get(GENERATED_BY_MODEL_KEY, "")
    vault = note.provenance.get(GENERATED_IN_VAULT_KEY, "")
    if model:
        parts.append(f"`{_md_cell(model)}`")
    if vault:
        parts.append(f"in `{_md_cell(vault)}`")
    return " ".join(parts)


def _one_line(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def render_text(digest: Digest) -> str:
    """The terminal rendering: flat, labelled, numbered, no markdown.

    A digest is normally read at a prompt, so this stays inside a plain-text
    convention -- no ``#``, no tables, no wikilink brackets to decode -- and gives
    every field an explicit label so nothing has to be inferred from punctuation.
    """
    lines = [
        f"Digest for {digest.date}  -  {len(digest.notes)} notes, "
        f"{len(digest.topics)} topics touched",
        f"vault {digest.vault_name} at {digest.vault_root}",
        "",
    ]

    if not digest.notes:
        lines.append(f"no notes for {digest.date}")
        lines.append("")
        return "\n".join(lines)

    for position, note in enumerate(digest.notes, 1):
        lines.append(f"  [{position}] {note.alias or note.title}")
        if note.alias:
            lines.append(f"      note: {note.title}")
        for line in note.summary.splitlines() or ["(no summary text)"]:
            lines.append(f"      {line.strip()}" if line.strip() else "")
        if note.topics:
            lines.append(f"      topics: {', '.join(note.topics)}")
        meta = _meta(note)
        if meta:
            lines.append(f"      {meta}")
        lines.append("")

    if digest.topics:
        lines.append(f"Topics touched ({len(digest.topics)})")
        width = max(len(_one_line(tally.topic, 40)) for tally in digest.topics)
        for tally in digest.topics:
            plural = "backlink" if tally.backlinks == 1 else "backlinks"
            lines.append(
                f"  {_one_line(tally.topic, 40):<{width}}  "
                f"{tally.backlinks:>3} {plural}  {', '.join(tally.notes)}"
            )
        lines.append("")

    if digest.chat is not None:
        lines.append(f"Chat log ({len(digest.chat)} entries)")
        for entry in digest.chat or []:
            lines.append(
                f"  {entry.time}  {_one_line(entry.user, 60) or '(no user text)'} "
                f"-> {_one_line(entry.agent, 60) or '(no agent text)'}"
            )
        lines.append("")

    return "\n".join(lines)


def _md_cell(value: str) -> str:
    """A value safe inside a markdown table cell: only ``|`` and newlines are special."""
    return " ".join((value or "").split()).replace("|", r"\|")


def render_md(digest: Digest) -> str:
    """The note rendering: headings, wikilinks and a table, ready to paste into Obsidian.

    Genuinely a different document rather than the text one with other separators: it
    uses the vault's own conventions (``[[wikilinks]]``, ``##`` headings) so what is
    pasted becomes part of the graph instead of a wall of prose *about* it. A topic
    table is worth the width here and not in the terminal, where it would wrap into
    nonsense.
    """
    lines = [
        f"# Second Brain digest - {digest.date}",
        "",
        f"> {len(digest.notes)} notes, {len(digest.topics)} topics touched. "
        f"Vault `{_md_cell(digest.vault_name)}` at `{_md_cell(digest.vault_root)}`.",
        "",
    ]

    if not digest.notes:
        lines.append(f"no notes for {digest.date}")
        lines.append("")
        return "\n".join(lines)

    for note in digest.notes:
        lines.append(f"## {note.alias or note.title}")
        lines.append("")
        lines.append(note.summary or "_no summary text_")
        lines.append("")
        if note.topics:
            lines.append("**Topics:** " + ", ".join(f"[[{_md_cell(t)}]]" for t in note.topics))
        provenance = _md_provenance(note)
        if provenance:
            lines.append(f"**Provenance:** {provenance}")
        if note.index_position is not None:
            lines.append(f"**Hub index:** [[{note.title}]] (#{note.index_position})")
        lines.append("")

    if digest.topics:
        lines.append("## Topics touched")
        lines.append("")
        lines.append("| Topic | Backlinks | Stub | Notes today |")
        lines.append("| --- | --- | --- | --- |")
        for tally in digest.topics:
            notes = ", ".join(f"[[{n}]]" for n in tally.notes)
            stub = f"[[{_md_cell(tally.stub)}]]" if tally.stub else "_missing_"
            lines.append(f"| [[{_md_cell(tally.topic)}]] | {tally.backlinks} | {stub} | {notes} |")
        lines.append("")

    if digest.chat is not None:
        lines.append("## Chat log")
        lines.append("")
        if not digest.chat:
            lines.append(f"no chat entries for {digest.date}")
            lines.append("")
        for entry in digest.chat or []:
            lines.append(f"### {entry.time}")
            lines.append("")
            if entry.user:
                lines.append(f"**User:** {entry.user}")
                lines.append("")
            if entry.agent:
                lines.append(f"**Agent:** {entry.agent}")
                lines.append("")

    return "\n".join(lines)


_RENDERERS = {"text": render_text, "md": render_md}


def render(digest: Digest, output_format: str = DEFAULT_FORMAT) -> str:
    """Render ``digest`` in one of :data:`FORMATS`."""
    try:
        return _RENDERERS[output_format](digest)
    except KeyError:
        raise ValueError(
            f"unknown format {output_format!r}; expected one of {', '.join(FORMATS)}"
        ) from None


# --- the CLI --------------------------------------------------------------------


def iso_date(value: str) -> str:
    """Argparse type for ``--date``: strictly ``YYYY-MM-DD``, and a real date.

    ``date.fromisoformat`` alone would also accept ``20260930`` and the ISO week forms,
    so a caller who typed the wrong shape would get a digest for a date they did not
    ask for and no indication of it. The regex pins the spelling this command prints,
    and ``fromisoformat`` then still rejects ``2026-13-01``, which the regex happily
    matches.
    """
    text = (value or "").strip()
    if not _ISO_DATE_RE.match(text):
        raise argparse.ArgumentTypeError(f"expected a date in YYYY-MM-DD form, got {value!r}")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a real calendar date in YYYY-MM-DD form, got {value!r}: {exc}"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m text_summarizer.digest",
        description=(
            "Report what the summarizer learned on a given day by reading the vault. "
            "No agent turn, no model call, no quota."
        ),
    )
    parser.add_argument(
        "--date",
        type=iso_date,
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="the day to report on (default: today)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FORMATS,
        default=DEFAULT_FORMAT,
        help="text for a terminal, md for pasting into a note (default: %(default)s)",
    )
    parser.add_argument(
        "--vault",
        default=None,
        metavar="PATH",
        help="the vault to read, i.e. the directory holding 'Second Brain' "
        "(default: the active vault, from SECOND_BRAIN_VAULT)",
    )
    parser.add_argument(
        "--include-chat",
        action="store_true",
        help="also report the day's entries from 'Chat Log/<date>.md' (off by default)",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit the machine-readable structure instead of a rendering; wins over --format",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = _resolve_root(args.vault)
    if not os.path.isdir(root):
        # Distinct from an empty day, which is a legitimate answer: a path that is not
        # a directory is a typo or a missing mount, and reporting "no notes" for it
        # would read as "the agent learned nothing there" rather than "you pointed me
        # at nothing".
        print(
            f"digest: not a directory: {root}\n        pass --vault PATH to read another vault",
            file=sys.stderr,
        )
        return 2

    digest = build_digest(args.date, vault_root=root, include_chat=args.include_chat)
    sys.stdout.write(to_json(digest) if args.as_json else render(digest, args.output_format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
