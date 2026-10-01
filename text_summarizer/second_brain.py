"""Second-brain persistence for the agent: save summaries as linked Obsidian notes.

The agent container mounts the vault's *parent* at /vaults (see docker-compose.yml)
and :mod:`text_summarizer.vaults` picks the active vault out of it, so this tool
writes notes directly to disk. The obsidian-mcp sidecar resolves the same vault
independently at startup and watches the same host directory, indexing new/changed
files automatically.

Provenance
----------
New notes record who wrote them:

    generated_by_model:  the model that actually served the turn
    generated_in_vault:  the resolved vault name

Both are read from the live invocation, never from a literal:

* the model comes from ``Event.model_version`` (see ``sources.py``), which is
  why the field is written by the tool itself rather than passed in by the
  model -- the model is told nothing about its own identity;
* the vault name comes from ``sources.resolve_vault_name``.

If neither can be resolved, the field is **omitted** rather than guessed. Notes
written before these fields existed therefore have no recorded provenance, and
``sources.render_sources`` renders that case as the literal ``unknown``.

Untrusted input
---------------
``title``, ``summary_content`` and ``topics`` are *model output*, which is to say
attacker-influenced: anything the user pasted lands in the title, and a jailbreak
lands in the topics. Two consequences are designed for rather than discovered:

* ``title`` and every topic reach the filesystem, so both are reduced by
  :func:`_safe_name` to an allowlist of ``[A-Za-z0-9 _-]`` and, independently,
  every path this module writes is asserted to resolve inside the vault
  (:func:`_assert_in_vault`). The sanitiser makes an escape unrepresentable; the
  assert is there so that if it ever stops doing that, the failure is a loud
  exception rather than a file outside the vault.
* ``title`` and every topic also reach the note's frontmatter, which YAML -- not
  the note reader -- interprets, so their values are quoted (:func:`_yaml_str`).
  Unquoted interpolation meant a title containing a newline could close the
  ``---`` block and add keys of its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

from . import vaults
from .sources import (
    mcp_vault_name_from_events,
    resolve_vault_name,
    served_model_from_events,
)

BRAIN_DIR = "Second Brain"
TOPICS_DIR = "Topics"
INDEX_NAME = "Second Brain Index"
CHAT_LOG_DIR = "Chat Log"

GENERATED_BY_MODEL_KEY = "generated_by_model"
GENERATED_IN_VAULT_KEY = "generated_in_vault"


def resolve_vault_root(configured: str | None = None) -> str:
    """Return the directory to read and write for this run.

    The whole selection rule lives in :mod:`text_summarizer.vaults`; this is the
    thin wrapper that supplies the configured parent and the ``OBSIDIAN_VAULT_NAME``
    choice, then reports where a vault that does not exist yet will be created.

    Under Docker the vault's *parent* is bind-mounted, never the vault itself.
    Mounting the vault (``.../ck-vault/ck:/vault``) flattens its name to ``vault``
    and the host directory name becomes unrecoverable from inside the container,
    which is why the parent is mounted instead.

    An ambiguous mount is not guessed: with several vaults and no
    ``OBSIDIAN_VAULT_NAME`` the selection error's message is printed and the
    unresolved parent is returned, so a read-only turn still returns notes while
    writes land nowhere useful. That is the deliberate asymmetry with
    ``resolve-vault.sh``, which exits non-zero on the same input -- the MCP server
    cannot meaningfully run against a wrong path, but the agent degrading to
    "unknown" provenance is better than not answering at all.
    """
    parent = (
        configured if configured is not None else os.environ.get("SECOND_BRAIN_VAULT", "")
    ).strip()
    if not parent:
        parent = "/vault"

    name = os.environ.get(vaults.VAULT_NAME_ENV, "").strip()
    try:
        selection = vaults.select_vault(parent, name=name or None)
    except vaults.VaultSelectionError as exc:
        print(f"[vault] {exc}")
        print(f"[vault] candidates:\n{vaults.format_candidates(vaults.discover_vaults(parent))}")
        return parent

    if selection.needs_create:
        # Lazy: a read-only turn against a fresh mount must not create anything.
        try:
            os.makedirs(os.path.join(selection.vault.path, BRAIN_DIR), exist_ok=True)
        except OSError as exc:  # pragma: no cover - unwritable mount
            print(f"[vault] could not create {selection.vault.path}: {exc}")
            return parent
    return selection.vault.path


VAULT_ROOT = resolve_vault_root()


#: Leading phrases that introduce the text to summarise rather than being part of
#: it. Stripped before fingerprinting, and before the relative-time scan so the two
#: examine the same words. Order matters: the longer forms come first, since
#: ``"summarize "`` would otherwise match the start of ``"summarize the following: "``.
_SUMMARIZE_PREFIXES = (
    "summarize the following: ",
    "please summarize: ",
    "can you summarize: ",
    "summarize: ",
    "summarize ",
)


def _strip_summarize_prefix(normalized: str) -> str:
    """Remove one leading summarise-introducing phrase from ``normalized`` text."""
    for prefix in _SUMMARIZE_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


#: Words naming a time relative to now. This exists because a relative time
#: reference is *not* part of the text being summarised: "summarize today news"
#: is the same request every day and a different answer, so keying it only on the
#: text made the cache serve the first day's note forever.
_TIME_WORDS = frozenset(
    {
        "today",
        "tomorrow",
        "yesterday",
        "tonight",
        "overnight",
        "this week",
        "this month",
        "this year",
        "past week",
        "past month",
        "past year",
        "last week",
        "last month",
        "last night",
        "last year",
        "latest",
        "recent",
        "recently",
        "breaking",
        "current",
        "currently",
        "right now",
        "at the moment",
        "news",
        "headlines",
    }
)

#: Units the time-word match runs on: letters and digits only, so hyphens and
#: punctuation separate and ``"todays"`` does not match ``"today"``.
_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def has_relative_time(text: str) -> bool:
    """True when ``text`` names a time relative to now.

    Matches whole words against :data:`_TIME_WORDS`, so ``"latest"`` qualifies and
    ``"greatest"`` does not. Multi-word entries are matched as adjacent unigrams
    (``this week`` = ``this`` + ``week``), which keeps one matching path instead of
    a separate phrase matcher.

    Only words in the part of the message *after* any ``summarize:``-style prefix
    are considered -- which is what :func:`source_fingerprint` normalises away
    before hashing, so the two must look at the same words. Without that,
    ``"Now summarize the cats."`` matched on its sentence-initial ``"now"`` and the
    prompt became day-scoped for no reason; that exact string is in
    ``test_adk_wiring``, which is how it was found.
    """
    words = _WORD_SPLIT_RE.split(
        _strip_summarize_prefix(" ".join((text or "").strip().lower().split()))
    )
    words = [w for w in words if w]
    if not words:
        return False

    single = {w for w in _TIME_WORDS if " " not in w}
    phrases = [p.split() for p in _TIME_WORDS if " " in p]

    for index, word in enumerate(words):
        if word in single:
            return True
        for phrase in phrases:
            span = len(phrase)
            if words[index : index + span] == phrase:
                return True
    return False


def cache_key_text_for(source_text: str, *, today: str | None = None) -> str:
    """The text the cache key is computed from, with relative times pinned to a day.

    Returns ``source_text`` unchanged unless it names a relative time, in which case
    today's date is appended. ``"summarize today news"`` therefore keys differently
    on each day while a same-day repeat still hits the cache, and a prompt with no
    relative time stays byte-identical to before -- so every note already in the
    vault keeps matching its own fingerprint and nothing is orphaned.

    ``today`` is injectable for the tests; it is ISO-8601, as ``date.today()`` is.
    """
    if not has_relative_time(source_text):
        return source_text or ""
    stamp = today or date.today().isoformat()
    return f"{source_text or ''} [as of {stamp}]"


def source_fingerprint(text: str) -> str:
    """Deterministic, whitespace/prefix-insensitive fingerprint of user source text.

    Deliberately unaware of the calendar: it fingerprints exactly the text it is
    given. Callers that hold a live user message go through
    :func:`cache_key_text_for` first, so both the write and the lookup agree.
    """
    normalized = _strip_summarize_prefix(" ".join((text or "").strip().lower().split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_name(value: str, max_len: int = 60) -> str:
    """Reduce arbitrary model output to a name that is safe as a path component.

    Everything here is an allowlist, not a blocklist. The rules, in order:

    1. any run of path separators (``/``, ``\\``) becomes a space;
    2. any run of dots becomes a space -- which is what kills ``..`` *and* makes a
       bare ``.``/``..``/``...`` reduce to ``""`` rather than to a name that is
       itself a parent reference;
    3. anything outside ``[A-Za-z0-9 _-]`` becomes a space (a character, not a
       deletion, so ``a/b`` reads as ``a-b`` and not ``ab``);
    4. whitespace and underscores collapse to ``-``, case is lowered, the result is
       truncated at ``max_len`` and any dangling ``-`` trimmed.

    Rule 3 is why the ASCII-only class is the whole defence: it removes Unicode
    lookalikes (``／``, U+2024 one-dot leader, RTL overrides) by construction, so
    there is no separator variant left to enumerate. The cost is that a
    non-ASCII title loses its accents -- deliberate, since these names become
    filenames, and Obsidian resolves wikilinks case-insensitively but not
    accent-insensitively.

    The cost of reducing two different strings to the same name -- ``a/b`` and
    ``a b`` both become ``a-b`` -- is merging, never loss: a repeated topic
    appends a backlink to the stub that already exists instead of making a second
    one, and a same-day title collision overwrites, which is the tool's documented
    behaviour anyway.
    """
    text = "" if value is None else str(value)
    text = re.sub(r"[/\\]+", " ", text)
    text = re.sub(r"\.+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 _-]+", " ", text)
    text = re.sub(r"[\s_]+", "-", text.strip().lower())
    return text[:max_len].strip("-")


def _slug(value: str, max_len: int = 60) -> str:
    """Kept under its old name for callers that predate :func:`_safe_name`.

    The writer calls :func:`_safe_name` directly, so this is an alias rather than a
    second implementation -- two sluggers is how the old one got a weaker one.
    """
    return _safe_name(value, max_len=max_len)


def _assert_in_vault(path: str) -> None:
    """Refuse to write anything that does not resolve inside the vault.

    Called from :func:`_write` -- the single place a path becomes a syscall --
    rather than from each caller, because a check that has to be remembered at
    every call site is a check that is eventually forgotten. Both sides are
    ``resolve()``d, so a symlinked ``VAULT_ROOT`` (``/tmp`` -> ``/private/tmp`` on
    macOS) compares like with like.

    This is a second line of defence, not the first: :func:`_safe_name` makes an
    escaping path unrepresentable, so this only fires if that has regressed. It
    raises rather than warns, because the only safe response to "this write would
    land outside the vault" is to not perform it.
    """
    root = Path(VAULT_ROOT).resolve()
    target = Path(path).resolve()
    if target != root and not target.is_relative_to(root):
        raise ValueError(f"refusing to write outside the vault: {path} (vault: {root})")


def _yaml_str(value: str) -> str:
    """Render a string as a YAML scalar that cannot escape the line it is on.

    ``json.dumps`` because its output *is* a valid YAML 1.2 double-quoted scalar:
    every escape it can emit (``\\"``, ``\\\\``, ``\\n``, ``\\t``, ``\\uXXXX``) is in
    YAML's own double-quoted escape set, so a value containing a newline becomes
    the two characters ``\\n`` inside the scalar instead of a line break that
    closes the frontmatter block and lets the rest of the value be read as new
    keys. ``ensure_ascii=False`` keeps accented text readable in the vault, which
    is the point of writing a note a human will open.
    """
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def _wikilink_target(topic: str) -> str:
    """Make a topic safe to sit between ``[[`` and ``]]``.

    Deliberately *not* :func:`_yaml_str`: this is markdown, not YAML, and quoting
    it would break every link in the vault rather than secure one. Obsidian treats
    ``[``, ``]``, ``|``, ``#``, ``^`` and newlines as link syntax, so those are
    replaced with a space; a plain topic like ``Mobile App`` is returned unchanged,
    which is what keeps the ``## Related`` section human-readable.
    """
    return re.sub(r"[\[\]|#^\n\r]+", " ", str(topic or "")).strip()


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def _write(path: str, content: str) -> None:
    _assert_in_vault(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _link_line(note_title: str) -> str:
    return f"- [[{note_title}]]\n"


def _session_events(tool_context) -> list:
    """Best-effort access to the live invocation's events from inside a tool.

    ``ToolContext`` is injected by ADK (``FunctionTool`` treats a parameter named
    ``tool_context`` as framework-supplied and hides it from the model's schema).
    Returns ``[]`` when the tool is called directly, e.g. from a unit test.
    """
    if tool_context is None:
        return []
    try:
        session = tool_context.session
    except Exception:  # pragma: no cover - context shape varies by ADK version
        return []
    return list(getattr(session, "events", None) or [])


def _part_texts(content) -> list[str]:
    if content is None:
        return []
    parts = getattr(content, "parts", None) or []
    return [str(p.text) for p in parts if getattr(p, "text", None) is not None]


def user_text_from_context(tool_context) -> str:
    """The verbatim user message of the turn this tool call belongs to.

    This is the **authoritative** cache key input, and it is read from the live
    invocation rather than taken from the model on purpose. See
    :func:`cache_key_text` for why.

    ``ToolContext.user_content`` is the field ADK documents for exactly this
    ("the user content that started this invocation"), and it is the same field
    ``agent._user_content_text`` reads on the lookup side, so the two keys cannot
    drift apart. The session's event list is only a fallback, and it is not
    equivalent: in a resumed multi-turn session ``tool_context.session`` can be a
    snapshot taken before the current user message was appended, in which case the
    most recent ``user`` event on it still belongs to the *previous* turn. That
    failure is silent -- it produces a plausible fingerprint of the wrong prompt
    -- which is exactly the class of bug this function exists to remove.
    """
    if tool_context is not None:
        try:
            text = "".join(_part_texts(getattr(tool_context, "user_content", None)))
        except Exception:  # pragma: no cover - context shape varies by ADK version
            text = ""
        if text.strip():
            return text

    for event in reversed(_session_events(tool_context)):
        if (getattr(event, "author", "") or "") == "user":
            return "".join(_part_texts(getattr(event, "content", None)))
    return ""


def cache_key_text(tool_context=None, fallback: str = "") -> str:
    """The text whose fingerprint keys the vault cache for this turn.

    **The model must never be trusted to supply this.** It used to be: the tool
    took a ``source_text`` argument, fingerprinted it, and the cache looked the
    turn up by the fingerprint of the user's prompt. Those are different strings,
    so the two could never agree and the cache never hit -- once per run, at the
    cost of the whole turn (three model round-trips). Worse, the model paraphrases
    ``source_text`` differently every time, and on a tool-driven turn it tends to
    pass *its own summary* -- so the recorded key was the output, not the input,
    and two identical questions produced two different keys.

    Reading the user message from the session makes key == lookup key by
    construction. If the session cannot be read (a direct call from a test, or a
    context shape this helper does not recognise) the caller's ``fallback`` is
    used, which keeps a plain function call working at the cost of a possible
    miss. A miss is the safe direction: it re-runs the model, whereas a wrong hit
    would replay an unrelated note.

    The raw message is then passed through :func:`cache_key_text_for`, which pins a
    prompt naming a *relative* time to the day it was asked. Without it,
    ``"summarize today news"`` fingerprinted identically on every day and the
    lookup served the first day's note forever. Both sides must go through here --
    the write below and the read in ``agent.cache_hit_before_model`` -- or they
    disagree and the cache silently never hits again.
    """
    return cache_key_text_for(user_text_from_context(tool_context) or (fallback or ""))


def note_provenance(tool_context=None) -> dict[str, str]:
    """Resolve the provenance recorded in a new note's frontmatter.

    Returns only the keys that could actually be resolved. An empty dict means
    "unknown" -- callers must omit those keys rather than invent a value, which
    is also how pre-existing notes (written before provenance was recorded) are
    treated downstream.
    """
    events = _session_events(tool_context)
    provenance: dict[str, str] = {}

    model = served_model_from_events(events)
    if model:
        provenance[GENERATED_BY_MODEL_KEY] = model

    vault = resolve_vault_name(
        vault_root=VAULT_ROOT,
        mcp_reported=mcp_vault_name_from_events(events),
    )
    if vault.name:
        provenance[GENERATED_IN_VAULT_KEY] = vault.name

    return provenance


def _provenance_lines(provenance: dict[str, str]) -> str:
    """Render provenance as frontmatter lines, YAML-quoted and omitted if empty."""
    lines = []
    for key in (GENERATED_BY_MODEL_KEY, GENERATED_IN_VAULT_KEY):
        value = (provenance.get(key) or "").strip()
        if not value:
            continue
        lines.append(f'{key}: "{value}"\n')
    return "".join(lines)


def _split_topics(topics: str) -> tuple[list[str], list[str]]:
    """Split the model's topic list into usable topics and ones with no file name.

    The split keeps the *original* string in both halves. The slug is computed
    only to answer "can this be a file?", and only the file on disk uses it: the
    stub's ``# heading`` and the ``## Related`` wikilink stay the human-readable
    text, because they are the copy a person reads and clicks.

    A topic that is non-empty but has no usable file name (``..``, ``.``, ``///``,
    a lone ``-``) is **dropped and reported** rather than written as a stub called
    ``...md`` or emitted as a tag the vault cannot link to. Degrading loudly is
    the point: silently vanishing would read to the agent as "that topic was
    filed" when nothing was written, and a stub with a meaningless name is worse
    than an absent one because it is indistinguishable from a real topic later.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for raw in (topics or "").split(","):
        topic = raw.strip()
        if not topic:
            continue
        (kept if _safe_name(topic) else dropped).append(topic)
    return kept, dropped


def save_summary_to_second_brain(
    title: str,
    summary_content: str,
    topics: str,
    tool_context=None,
) -> str:
    """Persist a summary into the Obsidian vault as a linked, graph-friendly note.

    A dated note is written under 'Second Brain/', topic stub notes receive
    backlinks, and every new note is appended to the hub note 'Second Brain Index'.
    The note's ``source_fingerprint`` -- the key the vault cache is looked up by --
    is computed here from the live user message, so the model is never asked to
    supply it (see :func:`cache_key_text`).
    The model and vault that produced the summary are recorded alongside
    (see ``note_provenance``); unresolvable provenance is omitted, not guessed.

    All three arguments are model output, so all three are treated as untrusted:
    ``title`` and ``topics`` are reduced by :func:`_safe_name` before they name
    anything on disk, every write is confined by :func:`_assert_in_vault`, and
    every value that lands in frontmatter is quoted by :func:`_yaml_str`.

    Args:
        title: Short descriptive title for the summary (e.g. "Customer Feedback 2026-09-25").
        summary_content: Markdown body with the bullet-point summary.
        topics: Comma-separated list of topics/keywords to link (e.g. "Mobile App, Checkout").
        tool_context: Injected by ADK; used to read live provenance and the cache
            key. Not for the model.
    """
    today = date.today().isoformat()
    slug = _safe_name(title)
    note_title = f"{today} - {slug}" if slug else f"{today} - summary"
    note_path = os.path.join(VAULT_ROOT, BRAIN_DIR, f"{note_title}.md")
    index_path = os.path.join(VAULT_ROOT, f"{INDEX_NAME}.md")
    topic_names, dropped_topics = _split_topics(topics)

    tags = "\n".join(f"  - {_yaml_str(t)}" for t in topic_names)
    key_text = cache_key_text(tool_context)
    fp_line = f"source_fingerprint: {source_fingerprint(key_text)}\n" if key_text else ""
    provenance_line = _provenance_lines(note_provenance(tool_context))
    frontmatter = (
        f"---\ntags:\n{tags}\ndate: {today}\n{fp_line}{provenance_line}"
        f"aliases:\n  - {_yaml_str(title)}\n---\n\n"
    )
    links = "\n".join(f"- [[{_wikilink_target(t)}]]" for t in topic_names)
    body = f"{frontmatter}{summary_content}\n\n## Related\n{links}\n"

    _write(note_path, body)

    index_body = _read(index_path)
    if not index_body.strip():
        index_body = f"# {INDEX_NAME}\n\nNone yet.\n"
    if _link_line(note_title) not in index_body:
        index_body = index_body.rstrip() + "\n" + _link_line(note_title)
    _write(index_path, index_body)

    created_topics = []
    for topic in topic_names:
        # Only the *filename* is slugged. The link in the body above and the
        # stub's heading keep the readable original, so a topic is still
        # recognisable to whoever opens the note.
        topic_path = os.path.join(VAULT_ROOT, TOPICS_DIR, f"{_safe_name(topic)}.md")
        topic_body = _read(topic_path)
        if not topic_body.strip():
            topic_body = f"# {topic}\n\n## Backlinks\n{_link_line(note_title)}"
            _write(topic_path, topic_body)
            created_topics.append(topic)
        elif _link_line(note_title) not in topic_body:
            if "## Backlinks" in topic_body:
                topic_body = topic_body.rstrip() + "\n" + _link_line(note_title)
            else:
                topic_body = topic_body.rstrip() + "\n\n## Backlinks\n" + _link_line(note_title)
            _write(topic_path, topic_body)

    dropped_line = (
        f"Dropped topics ({len(dropped_topics)}): {', '.join(dropped_topics)}\n"
        if dropped_topics
        else ""
    )
    # No created/updated field for the note itself: the old one was
    # ``if note_title not in index_body or index_body``, whose right-hand operand is
    # the non-empty index body this function has just written, so it was always true
    # and the list it guarded was always ``[note_path]`` -- and was then never read.
    # "Saved note" is the accurate wording for both cases (the file was written
    # either way, overwriting a same-day note of the same name), and the genuinely
    # new information is already reported by "Created topic stubs".
    result = (
        f"Saved note: {note_path}\n"
        f"Linked topics ({len(topic_names)}): {', '.join(topic_names) or 'none'}\n"
        f"{dropped_line}"
        f"Created topic stubs: {', '.join(created_topics) or 'none'}\n"
        f"Updated index: {index_path}"
    )
    return result


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a note into ``(frontmatter, body)``, recognising the fence by *line*.

    A regex cannot do this correctly, and neither can a reader that assumes the
    frontmatter ends at the first ``---`` it finds: a quoted scalar may still
    *contain* that string, and stopping inside it hands the rest of the
    frontmatter back as body. The value is inert to a YAML reader and very much not
    inert to a regex, so only a line that is exactly the fence closes the block.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1:])
    return "", text


def find_cached_summary(source_text: str) -> str | None:
    """Return the stored summary note for an identical source text, if any.

    Scans the Second Brain notes' frontmatter for a ``source_fingerprint`` matching
    ``source_text``. Pure filesystem lookup -- no LLM call involved.

    ``source_text`` is the *user's message*, the same string
    :func:`save_summary_to_second_brain` fingerprinted via
    :func:`cache_key_text`. Both sides must keep using the same input or the cache
    silently never hits.

    Reads the fence with :func:`_split_frontmatter` rather than a ``^---.*?---``
    regex, so a note whose title contained ``---`` replays its summary instead of
    the tail of its own frontmatter. That is the same injection as the write side,
    read back.

    Known cost: O(n) reads over the whole Second Brain directory on every model
    call. Measured at well under a millisecond for a few dozen notes, so it is not
    worth an index file until a vault is large enough for that to show up.
    """
    digest = source_fingerprint(source_text)
    brain_dir = os.path.join(VAULT_ROOT, BRAIN_DIR)
    if not os.path.isdir(brain_dir):
        return None
    frontmatter_spec = rf"source_fingerprint:\s*([0-9a-f]{{64}})\s*\n"
    for name in os.listdir(brain_dir):
        if not name.endswith(".md"):
            continue
        frontmatter, body = _split_frontmatter(_read(os.path.join(brain_dir, name)))
        m = re.search(frontmatter_spec, frontmatter)
        if m and m.group(1) == digest:
            # Both ends: the regex this replaced ended its match with ``\s*``, so the
            # caller saw the body without the blank line the writer puts after the
            # closing fence.
            return body.split("## Related", 1)[0].strip()
    return None


def log_conversation(user_message: str, agent_response: str) -> str:
    """Append one user<->agent exchange to today's chat log note in the vault.

    Everything in the chat log is dated and time-stamped so the agent can later
    reconstruct what was discussed by reading (note_read) the relevant day's note.
    Reuses save_summary_to_second_brain's topic convention so new day notes are
    discoverable too.

    Args:
        user_message: What the user said.
        agent_response: What the agent replied (markdown, bullets included).
    """
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M")
    chat_path = os.path.join(VAULT_ROOT, CHAT_LOG_DIR, f"{today}.md")
    index_path = os.path.join(VAULT_ROOT, f"{INDEX_NAME}.md")

    entry_title = f"{today} - chat log"
    body = _read(chat_path)
    if not body.strip():
        body = f"# Chat Log -- {today}\n\n"
    body = body.rstrip() + "\n\n"
    body += f"## {now}\n\n**User:**\n\n{user_message.strip()}\n\n**Agent:**\n\n{agent_response.strip()}\n"
    _write(chat_path, body)

    index_body = _read(index_path)
    if not index_body.strip():
        index_body = f"# {INDEX_NAME}\n\nNone yet.\n"
    if _link_line(entry_title) not in index_body:
        index_body = index_body.rstrip() + "\n" + _link_line(entry_title)
    _write(index_path, index_body)

    return f"Logged conversation to {chat_path}\nUpdated index: {index_path}"