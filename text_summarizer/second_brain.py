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
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import date, datetime

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


def source_fingerprint(text: str) -> str:
    """Deterministic, whitespace/prefix-insensitive fingerprint of user source text."""
    normalized = " ".join((text or "").strip().lower().split())
    for prefix in (
        "summarize: ",
        "summarize the following: ",
        "please summarize: ",
        "can you summarize: ",
        "summarize ",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _slug(value: str, max_len: int = 60) -> str:
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[\s_]+", "-", value)
    return value[:max_len].rstrip("-")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def _write(path: str, content: str) -> None:
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
    """
    return user_text_from_context(tool_context) or (fallback or "")


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

    Args:
        title: Short descriptive title for the summary (e.g. "Customer Feedback 2026-09-25").
        summary_content: Markdown body with the bullet-point summary.
        topics: Comma-separated list of topics/keywords to link (e.g. "Mobile App, Checkout").
        tool_context: Injected by ADK; used to read live provenance and the cache
            key. Not for the model.
    """
    today = date.today().isoformat()
    slug = _slug(title)
    note_title = f"{today} - {slug}" if slug else f"{today} - summary"
    note_path = os.path.join(VAULT_ROOT, BRAIN_DIR, f"{note_title}.md")
    index_path = os.path.join(VAULT_ROOT, f"{INDEX_NAME}.md")
    topic_names = [t.strip() for t in topics.split(",") if t.strip()]

    tags = "\n".join(f"  - {t}" for t in topic_names)
    key_text = cache_key_text(tool_context)
    fp_line = f"source_fingerprint: {source_fingerprint(key_text)}\n" if key_text else ""
    provenance_line = _provenance_lines(note_provenance(tool_context))
    frontmatter = (
        f"---\ntags:\n{tags}\ndate: {today}\n{fp_line}{provenance_line}aliases:\n  - {title}\n---\n\n"
    )
    links = "\n".join(f"- [[{t}]]" for t in topic_names)
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
        topic_path = os.path.join(VAULT_ROOT, TOPICS_DIR, f"{topic}.md")
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

    created = []
    if note_title not in index_body or index_body:
        created.append(note_path)
    result = (
        f"Saved note: {note_path}\n"
        f"Linked topics ({len(topic_names)}): {', '.join(topic_names) or 'none'}\n"
        f"Created topic stubs: {', '.join(created_topics) or 'none'}\n"
        f"Updated index: {index_path}"
    )
    return result


def find_cached_summary(source_text: str) -> str | None:
    """Return the stored summary note for an identical source text, if any.

    Scans the Second Brain notes' frontmatter for a ``source_fingerprint`` matching
    ``source_text``. Pure filesystem lookup -- no LLM call involved.

    ``source_text`` is the *user's message*, the same string
    :func:`save_summary_to_second_brain` fingerprinted via
    :func:`cache_key_text`. Both sides must keep using the same input or the cache
    silently never hits.

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
        body = _read(os.path.join(brain_dir, name))
        m = re.search(frontmatter_spec, body)
        if m and m.group(1) == digest:
            stripped = re.sub(r"^---.*?---\s*", "", body, flags=re.DOTALL)
            stripped = stripped.split("## Related", 1)[0].rstrip()
            return stripped
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