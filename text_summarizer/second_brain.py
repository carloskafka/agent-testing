"""Second-brain persistence for the agent: save summaries as linked Obsidian notes.

The agent container mounts the vault at /vault (see docker-compose.yml), so this
tool writes notes directly to disk. The obsidian-mcp sidecar watches the same
host directory and indexes new/changed files automatically.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import date, datetime

VAULT_ROOT = os.environ.get("SECOND_BRAIN_VAULT", "/vault")
BRAIN_DIR = "Second Brain"
TOPICS_DIR = "Topics"
INDEX_NAME = "Second Brain Index"
CHAT_LOG_DIR = "Chat Log"


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


def save_summary_to_second_brain(
    title: str,
    summary_content: str,
    topics: str,
    source_text: str = "",
) -> str:
    """Persist a summary into the Obsidian vault as a linked, graph-friendly note.

    A dated note is written under 'Second Brain/', topic stub notes receive
    backlinks, and every new note is appended to the hub note 'Second Brain Index'.
    When source_text is provided, its fingerprint is stored in the note frontmatter
    so repeated identical inputs can be served from the vault without an LLM call.

    Args:
        title: Short descriptive title for the summary (e.g. "Customer Feedback 2026-09-25").
        summary_content: Markdown body with the bullet-point summary.
        topics: Comma-separated list of topics/keywords to link (e.g. "Mobile App, Checkout").
        source_text: The original text the user asked to summarize (for dedup).
    """
    today = date.today().isoformat()
    slug = _slug(title)
    note_title = f"{today} - {slug}" if slug else f"{today} - summary"
    note_path = os.path.join(VAULT_ROOT, BRAIN_DIR, f"{note_title}.md")
    index_path = os.path.join(VAULT_ROOT, f"{INDEX_NAME}.md")
    topic_names = [t.strip() for t in topics.split(",") if t.strip()]

    tags = "\n".join(f"  - {t}" for t in topic_names)
    fp_line = f"source_fingerprint: {source_fingerprint(source_text)}\n" if source_text else ""
    frontmatter = (
        f"---\ntags:\n{tags}\ndate: {today}\n{fp_line}aliases:\n  - {title}\n---\n\n"
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

    Scans the Second Brain notes' frontmatter for a source_fingerprint matching
    source_text. Pure filesystem lookup -- no LLM call involved.
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