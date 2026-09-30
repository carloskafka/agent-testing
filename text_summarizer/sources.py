"""Deterministic ``**Sources**`` rendering for the text summarizer.

Why this module exists
----------------------
The agent is wired to a ``FallbackModel`` (Gemini first, then free OpenRouter
models), so *which* backend actually answers a given call is unknowable when the
prompt is built. Asking the model to report its own name would therefore yield a
guess, not the truth. Both identifiers are injected in code *after* the model has
run:

* **served model** -- read from ``Event.model_version``. ``Event`` subclasses
  ``LlmResponse``, and ADK merges every non-``None`` ``LlmResponse`` field into
  the model-response event (``google/adk/flows/llm_flows/base_llm_flow.py``,
  ``_finalize_model_response_event``). ``model_version`` is populated by both
  backends (``google/adk/models/google_llm.py`` via ``LlmResponse.create`` ->
  ``generate_content_response.model_version``, and
  ``google/adk/models/lite_llm.py`` -> ``response.model``), so it reflects
  whichever backend served the request.
* **vault name** -- resolved at runtime by :func:`resolve_vault_name`; see
  ``VAULT_NAME`` in ``.env.example`` and the "Vault name resolution" section of
  ``AGENTS.md`` for the container bind-mount caveat.

The model therefore only emits the *shape* of a source line, carrying two
sentinel tokens that this module replaces. Rendering normalizes whatever spelling
the model used (missing heading, duplicate headings, legacy ``[vault] [[Note]]``
lines, leading bullets) into exactly one canonical block, so a model that ignores
the format still cannot produce duplicates.

Rendered shape::

    **Sources**
    - [obsidian][<vault_name>][<model_name>][[Note Title]]: why it is relevant
    - [obsidian][<vault_name>][unknown][[Older Note]]: provenance not recorded

Every source line is a markdown list item. Bare lines would be joined into a
single paragraph by every markdown renderer (and by the dev UI's message
component), which reads as one unreadable run-on block once the note titles get
long -- so the bullet is part of the format, not decoration. The model is not
asked for it: the renderer emits the list itself.

Clickable titles
----------------
The note title becomes a real markdown link when the title can be resolved to a
file in the vault, so a click lands on the source instead of on nothing::

    - [obsidian][ck][gemini-3.5-flash-lite][[Email]](/vault/Topics/Email.md): why

``[[Email]]`` alone is Obsidian wikilink syntax, which no browser and not even
the ADK dev UI's markdown renderer understands -- it renders as literal
``[Email]``. Wrapping it as ``[[Email]](href)`` is the one spelling that both
survives that renderer as a link *and* still shows the brackets, so the line
keeps its Obsidian-readable shape while being clickable.

The href is served by the same FastAPI app that serves the dev UI
(``/vault/<path relative to the vault root>``), because the browser cannot read
a path inside the container and ``obsidian://`` does not resolve when Obsidian
is itself containerised.

Resolution is a **title lookup, never a guess**: the title the model wrote is
matched against real files under the vault root, and a title that matches
nothing is rendered as a plain ``[[wikilink]]`` exactly as before. The model is
never asked for a path -- it only ever saw a title, and a path it invented
would produce a link that 404s.

Notes written before ``generated_by_model`` existed in the frontmatter have no
recorded provenance. That renders as the literal ``unknown`` -- never guessed,
never back-filled from whichever model happens to be serving now.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from urllib.parse import quote

__all__ = [
    "MODEL_TOKEN",
    "SOURCES_HEADING",
    "SOURCE_BULLET",
    "SOURCE_KIND",
    "UNKNOWN",
    "VAULT_TOKEN",
    "VAULT_WEB_PREFIX",
    "SourceEntry",
    "VaultIdentity",
    "mcp_vault_name_from_events",
    "note_href",
    "render_sources",
    "resolve_vault_name",
    "served_model_from_events",
    "summary_only",
]

# --- wire format -------------------------------------------------------------

#: Sentinel the model copies verbatim in place of the real vault name. Doubled
#: ``@`` plus SCREAMING_SNAKE is not markdown-significant and is not something a
#: model produces in prose, so it cannot collide with real content.
VAULT_TOKEN = "@@ADK_VAULT@@"

#: Sentinel the model copies verbatim in place of the real served model.
MODEL_TOKEN = "@@ADK_MODEL@@"

_TOKEN_RE = re.compile(f"{re.escape(VAULT_TOKEN)}|{re.escape(MODEL_TOKEN)}")

SOURCES_HEADING = "**Sources**"

#: One source per markdown list item. See the module docstring.
SOURCE_BULLET = "- "

#: Leading kind marker, kept stable so the block stays machine-greppable.
SOURCE_KIND = "obsidian"

#: Rendered for provenance that was never recorded. Never a guess.
UNKNOWN = "unknown"

#: URL prefix under which the dev UI app serves the vault read-only. It has to
#: match the mount point in ``serve.py``; see the module docstring.
VAULT_WEB_PREFIX = "/vault"

# ``## Sources`` / ``### Sources`` / ``**Sources**`` / ``Sources:`` on its own line.
_HEADING_RE = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]*)?\**\s*sources\s*\**\s*:?[ \t]*$",
    re.IGNORECASE,
)
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\n]+?)\]\]")
# The linked spelling the renderer emits, e.g. ``[\[\[Note\]\](/vault/Topics/Note.md)``:
# a link whose *text* is the escaped wikilink, so the brackets survive to the
# screen instead of being eaten as link syntax. The capture excludes backslashes
# so it cannot run past the closing escape into the URL.
_LINKED_WIKILINK_RE = re.compile(
    r"\[\\\[\\\[([^\[\]\\\n]+?)\\\]\\\]\]\(([^)\s]*)\)"
)
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_REASON_SEP_RE = re.compile(r"^(?::|\||[-—–])[ \t]*")
_KIND_RE = re.compile(rf"^\[?{re.escape(SOURCE_KIND)}\]?", re.IGNORECASE)


@dataclass(frozen=True)
class SourceEntry:
    """One consulted note: the wikilink target plus an optional reason."""

    note: str
    reason: str = ""


@dataclass(frozen=True)
class VaultIdentity:
    """A vault name plus the runtime source it was resolved from."""

    name: str
    source: str


# --- vault name resolution ----------------------------------------------------


def _sanitize_identifier(value: str) -> str:
    """Make a value safe to drop inside the ``[...]`` tokens of a source line.

    Brackets would break the grammar and whitespace would split the token, so
    both are folded away. Names such as ``qwen/qwen3.8-27b:free`` and ``ck``
    pass through unchanged.
    """
    cleaned = re.sub(r"[\[\]]", "", value or "").strip()
    return re.sub(r"\s+", "-", cleaned)


def _vault_name_from_root(root: str) -> str:
    """Basename of the vault root, or "" when the root is not a real directory name.

    ``/vaults/ck`` yields ``ck`` -- correct for a local run and under Docker,
    because the parent directory is bind-mounted so the host name survives in the
    path. Requires the path to exist: a configured path that is not a directory
    (typo, missing mount) must fall through to ``unknown`` rather than rendering
    whatever the last path segment happens to be. See ``AGENTS.md``.
    """
    normalized = os.path.normpath((root or "").strip())
    if not normalized or normalized in (os.sep, ".", ".."):
        return ""
    if not os.path.isdir(normalized):
        return ""
    return os.path.basename(normalized)


def _vault_name_from_mcp_payload(payload, *, _depth: int = 0) -> str | None:
    """Pull ``vault_name`` out of an obsidian-mcp ``vault_info`` tool result.

    Accepts the raw ``function_response`` payload (a dict, or a JSON string) in
    the shapes ADK and obsidian-mcp produce. Returns ``None`` when no vault name
    is present.
    """
    if payload is None or _depth > 4:
        return None
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", "replace")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None

    name = payload.get("vault_name")
    if isinstance(name, str) and name.strip():
        return name.strip()

    # ADK wraps tool output as {"result": <text>}; obsidian-mcp returns the
    # vault_info object as that text.
    for key in ("result", "output", "structuredContent", "content"):
        if key in payload:
            nested = _vault_name_from_mcp_payload(payload[key], _depth=_depth + 1)
            if nested:
                return nested
    return None


def _events_list(events) -> list:
    return list(events or [])


def _parts(event) -> list:
    content = getattr(event, "content", None)
    return list(getattr(content, "parts", None) or [])


def _function_responses(event) -> Iterator:
    for part in _parts(event):
        response = getattr(part, "function_response", None)
        if response is not None:
            yield response


def mcp_vault_name_from_events(events) -> str | None:
    """Return the vault name reported by an MCP ``vault_info`` call, if any.

    Opportunistic: only populated when the turn actually called ``vault_info``.
    The tool is exposed (see the filter in ``obsidian_tools.py``) but the agent
    is never required to call it, so the primary resolution path is
    ``VAULT_NAME`` / ``basename(vault_root)``.
    """
    for event in reversed(_events_list(events)):
        for response in _function_responses(event):
            if (getattr(response, "name", "") or "") != "vault_info":
                continue
            name = _vault_name_from_mcp_payload(getattr(response, "response", None))
            if name:
                return name
    return None


def resolve_vault_name(
    *,
    env: dict | None = None,
    vault_root: str | None = None,
    mcp_reported: str | None = None,
) -> VaultIdentity:
    """Resolve the vault name at runtime, in priority order.

    1. ``basename(vault_root)`` -- authoritative. Callers pass the *resolved*
       vault root (see ``second_brain.resolve_vault_root``), which under Docker is
       the single child of the parent directory that gets bind-mounted, so this is
       the real host directory name with nothing configured.
    2. ``VAULT_NAME`` -- explicit override, for the case where the vault is mounted
       directly and its name genuinely is not in the path.
    3. ``mcp_reported`` -- the ``vault_name`` from the obsidian-mcp ``vault_info``
       tool, when the turn called it. Derived from the path the server was launched
       with, so it agrees with step 1 whenever the mount preserves the name.
    4. :data:`UNKNOWN` with source ``"unknown"``.
    """
    env = os.environ if env is None else env
    if vault_root is None:
        vault_root = env.get("SECOND_BRAIN_VAULT", "") or "/vault"

    from_root = _sanitize_identifier(_vault_name_from_root(vault_root))
    if from_root:
        return VaultIdentity(from_root, "vault-path")

    configured = _sanitize_identifier(env.get("VAULT_NAME", ""))
    if configured:
        return VaultIdentity(configured, "VAULT_NAME")

    reported = _sanitize_identifier(mcp_reported or "")
    if reported:
        return VaultIdentity(reported, "mcp:vault_info")

    return VaultIdentity(UNKNOWN, "unknown")


# --- note path resolution ----------------------------------------------------

#: Bytes of a note read when looking for its frontmatter. The alias list sits
#: near the top; reading the whole file for every note in the vault on every
#: render would be wasteful, and the walk already touches each file.
_FRONTMATTER_PREFIX_BYTES = 8192

_ALIASES_BLOCK_RE = re.compile(
    r"^aliases?[ \t]*:[ \t]*(?:\[(?P<inline>[^\]\n]*)\]|\n(?P<items>(?:[ \t]*-[^\n]*\n?)*))?",
    re.MULTILINE,
)
_ALIAS_LINE_RE = re.compile(r"^[ \t]*-[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def _frontmatter_aliases(path: str) -> list[str]:
    """The ``aliases`` (or ``alias``) values from a note's YAML frontmatter.

    Obsidian resolves ``[[Title]]`` against an alias as readily as against the
    filename, and it has to: every note this agent writes carries
    ``aliases: [<the title the model chose>]`` while the file on disk is named
    ``<date> - <slug>`` (``save_summary_to_second_brain``). The model cites the
    alias, because that is the readable name it was shown, so an index built
    from filenames alone would leave most citations unlinked.

    Returns ``[]`` for a note with no frontmatter, no aliases, or a body that
    cannot be read -- all of which mean "no aliases", never "guess some".
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(_FRONTMATTER_PREFIX_BYTES)
    except OSError:
        return []
    if not head.startswith("---"):
        return []
    end = head.find("\n---", 3)
    if end == -1:
        return []
    block = head[:end]
    match = _ALIASES_BLOCK_RE.search(block)
    if not match:
        return []
    if match.group("inline") is not None:
        # `aliases: [One, Two]` -- valid YAML, and Obsidian accepts it.
        raw = "\n".join(f"- {part}" for part in match.group("inline").split(","))
    else:
        raw = match.group("items") or ""
    return [
        m.group(1).strip().strip("\"'")
        for m in _ALIAS_LINE_RE.finditer(raw)
        if m.group(1).strip()
    ]


def _build_title_index(vault_root: str) -> dict[str, str]:
    """Map every name a note answers to, to its vault-relative path.

    Two kinds of key, in Obsidian's own precedence: the filename stem first, then
    the frontmatter aliases. Filenames are indexed in a full pass before any
    alias, so a filename always beats an alias that happens to collide with it.

    Built once per render call by walking the vault. A few hundred notes is a
    few milliseconds -- the same order as the cache lookup that already runs on
    the first model call of every turn -- and caching across turns would mean
    invalidating on writes this module cannot see.

    Titles collide in Obsidian: two notes may share a name in different folders.
    The first match wins, in a deterministic order (see :func:`_vault_walk_order`),
    so the same vault always yields the same link.
    """
    index: dict[str, str] = {}
    notes = list(_vault_walk_order(vault_root))
    for abs_path, rel_path in notes:
        index.setdefault(os.path.splitext(os.path.basename(abs_path))[0], rel_path)
    for abs_path, rel_path in notes:
        for alias in _frontmatter_aliases(abs_path):
            if alias:
                index.setdefault(alias, rel_path)
    return index


def _vault_walk_order(vault_root: str) -> Iterator[tuple[str, str]]:
    """Yield ``(absolute path, vault-relative path)`` for every note, stably.

    Sorted at each level rather than left in ``os.walk`` order, so which note
    wins a duplicate title does not depend on the filesystem's enumeration.
    Symlinked directories are not followed: a link out of the vault would both
    escape the mount and make the walk unbounded.
    """
    if not vault_root or not os.path.isdir(vault_root):
        return
    for dirpath, dirnames, filenames in os.walk(vault_root, followlinks=False):
        dirnames[:] = sorted(
            d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
        )
        for name in sorted(filenames):
            if not name.endswith(".md") or name.startswith("."):
                continue
            abs_path = os.path.join(dirpath, name)
            yield abs_path, os.path.relpath(abs_path, vault_root)


def note_href(title: str, vault_root: str | None) -> str | None:
    """Return the dev-UI URL for the note called ``title``, or ``None``.

    ``None`` means "no such note": the caller then leaves the title as a plain
    ``[[wikilink]]``, which is what the renderer did before links existed. A
    link is only ever emitted for a file that is actually there, because a dead
    link is worse than an unlinked title -- it looks like the note is there and
    is not.

    Titles are matched exactly, and then case-insensitively, so a title the
    model cased differently still resolves. Nothing is fuzzy-matched: picking
    the closest note would be a guess about which one was meant.

    Building the index is a whole-vault walk, so a caller resolving several
    titles at once should build it once and use :func:`_href_from_index`. That
    is what :func:`render_sources` does.
    """
    if not title or not vault_root:
        return None
    return _href_from_index(title, _build_title_index(vault_root))


def _href_from_index(title: str, index: dict[str, str]) -> str | None:
    """Resolve one title against a prebuilt index. See :func:`note_href`."""
    rel = index.get(title)
    if rel is None:
        folded = title.casefold()
        for name, candidate in index.items():
            if name.casefold() == folded:
                rel = candidate
                break
    if rel is None:
        return None
    # Quote per segment: a title with a space or a '#' in it would otherwise
    # produce a URL the browser truncates at the fragment.
    return VAULT_WEB_PREFIX + "/" + "/".join(
        quote(segment) for segment in rel.split(os.sep)
    )


# --- served model resolution -------------------------------------------------


def _has_model_output(event) -> bool:
    """True when the event carries model output (text or a function call).

    A tool call counts: ``save_summary_to_second_brain`` runs *before* the agent
    writes its answer, and the event that asked for the tool call is the one
    carrying ``model_version`` at that moment.
    """
    for part in _parts(event):
        if getattr(part, "function_call", None) is not None:
            return True
        if (getattr(part, "text", None) or "").strip():
            return True
    return False


def served_model_from_events(events) -> str | None:
    """Return the model that actually served this turn, or ``None`` if unknown.

    Walks the session backwards and takes ``model_version`` off the most recent
    complete model-authored event. Tool results are skipped (they are authored
    by the agent and carry no ``model_version``), and so is a cached replay --
    a replayed ``LlmResponse`` has no model_version, which is exactly the signal
    that the text was not freshly generated.

    For the response itself there is no need to search: the rewriting callback
    holds the ``LlmResponse`` the model just produced, so it reads
    ``model_version`` off that. This helper exists for the paths that have no
    response in hand -- ``save_summary_to_second_brain`` writing note frontmatter
    from inside a tool call (``second_brain.note_provenance``).
    """
    for event in reversed(_events_list(events)):
        if getattr(event, "partial", False):
            continue
        if (getattr(event, "author", "") or "") == "user":
            continue
        if not _has_model_output(event):
            continue
        model_version = getattr(event, "model_version", None)
        if model_version and str(model_version).strip():
            return _sanitize_identifier(str(model_version))
    return None


# --- parsing -----------------------------------------------------------------


def _parse_source_line(line: str) -> SourceEntry | None:
    """Parse one source line into a :class:`SourceEntry`.

    Accepts the canonical token form as well as legacy spellings the model may
    still produce (``[vault] [[Note]]``, ``- [[Note]]: reason``), and the linked
    spelling the renderer itself emits, so a rendered block re-parses into the
    same entries -- that is what keeps rendering idempotent. The note is the
    first ``[[wikilink]]``; the reason is whatever follows it.
    """
    match = _LINKED_WIKILINK_RE.search(line)
    if match:
        # The href is the renderer's own, so it is dropped and the href is
        # recomputed from the vault on the next pass. Keeping it would let a
        # stale path survive a note being renamed.
        note = match.group(1).strip()
        reason = _REASON_SEP_RE.sub("", line[match.end():].strip()).strip()
        return SourceEntry(note=note, reason=reason) if note else None

    match = _WIKILINK_RE.search(line)
    if not match:
        return None
    note = match.group(1).strip()
    if not note:
        return None
    reason = _REASON_SEP_RE.sub("", line[match.end():].strip()).strip()
    return SourceEntry(note=note, reason=reason)


def _starts_source_block(line: str) -> bool:
    """True for a line that opens a Sources block without a heading."""
    if (_WIKILINK_RE.search(line) or _LINKED_WIKILINK_RE.search(line)) and (
        _TOKEN_RE.search(line)
    ):
        return True
    return bool(_KIND_RE.match(line.strip()))


def _is_block_line(line: str) -> bool:
    """True for a line that belongs to a Sources block (list item, wikilink, kind tag)."""
    if not line.strip():
        return False
    return (
        bool(_BULLET_RE.match(line))
        or bool(_WIKILINK_RE.search(line))
        or bool(_LINKED_WIKILINK_RE.search(line))
        or bool(_KIND_RE.match(line.strip()))
    )


def _scrub(line: str) -> str:
    """Remove leftover sentinel tokens from prose without reflowing it."""
    if not _TOKEN_RE.search(line):
        return line
    cleaned = re.sub(r"[ \t]{2,}", " ", _TOKEN_RE.sub("", line))
    return cleaned.rstrip()


# --- rendering ---------------------------------------------------------------


def _dedupe(entries: Iterable[SourceEntry]) -> list[SourceEntry]:
    seen: set[str] = set()
    out: list[SourceEntry] = []
    for entry in entries:
        key = entry.note.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _format_block(
    entries: Iterable[SourceEntry],
    vault: str,
    model: str,
    index: dict[str, str] | None = None,
) -> list[str]:
    lines = [SOURCES_HEADING]
    for entry in entries:
        title = f"[[{entry.note}]]"
        href = _href_from_index(entry.note, index) if index else None
        if href:
            # Backslash-escaped so the title still *displays* as [[Note]] while
            # being a link; see _LINKED_WIKILINK_RE.
            title = rf"[\[\[{entry.note}\]\]]({href})"
        line = f"{SOURCE_BULLET}[{SOURCE_KIND}][{vault}][{model}]{title}"
        if entry.reason:
            line += f": {entry.reason}"
        lines.append(line)
    return lines


def render_sources(
    text: str,
    *,
    vault_name: str | None = None,
    model_name: str | None = None,
    vault_root: str | None = None,
) -> str:
    """Re-emit the model's Sources block with the real vault and model names.

    Any ``**Sources**`` / ``## Sources`` heading the model produced is discarded
    and exactly one canonical block is written in its place, so a model that
    ignores the format can never yield duplicates. Each source is emitted as its
    own list item (:data:`SOURCE_BULLET`) regardless of the spelling it arrived
    in, and identifiers that could not be resolved render as :data:`UNKNOWN`;
    they are never invented.

    ``vault_root`` enables clickable titles: a note title that resolves to a
    real file under it is linked to the dev UI's ``/vault`` route, and one that
    does not is left as a plain ``[[wikilink]]``. Omit it and every title stays
    unlinked, which is the behaviour when the vault is not readable.

    Returns the input unchanged (modulo stray sentinel tokens) when it contains
    no Sources block.
    """
    text = text or ""
    if not text.strip():
        return text

    vault = _sanitize_identifier(vault_name or "") or UNKNOWN
    model = _sanitize_identifier(model_name or "") or UNKNOWN

    lines = text.split("\n")
    out: list[str] = []
    entries: list[SourceEntry] = []
    anchor: int | None = None
    in_block = False

    for line in lines:
        if in_block:
            if _is_block_line(line):
                parsed = _parse_source_line(line)
                if parsed is not None:
                    entries.append(parsed)
                continue
            in_block = False

        if _HEADING_RE.match(line) or _starts_source_block(line):
            if anchor is None:
                anchor = len(out)
            in_block = True
            parsed = _parse_source_line(line)
            if parsed is not None:
                entries.append(parsed)
            continue

        out.append(_scrub(line))

    if anchor is None:
        # No Sources block at all -- just drop any stray sentinel tokens.
        return "\n".join(out)

    deduped = _dedupe(entries)
    if not deduped:
        # A heading with no usable source line: drop it rather than render an
        # empty section.
        return "\n".join(out).rstrip()

    # One walk of the vault for the whole block, not one per source line.
    index = _build_title_index(vault_root) if vault_root else None
    out[anchor:anchor] = _format_block(deduped, vault, model, index)
    return "\n".join(out)


def strip_sources_block(text: str) -> str:
    """Remove the Sources block (heading plus its lines) and keep everything else.

    Used by the quality heuristics so source lines are never counted as summary
    bullets, while text that follows the block is preserved. A single blank
    separator is collapsed at the splice point so removing the block does not
    leave a double gap behind.
    """
    text = text or ""
    lines = text.split("\n")
    out: list[str] = []
    in_block = False
    just_ended = False
    for line in lines:
        if in_block:
            if _is_block_line(line):
                continue
            in_block = False
            just_ended = True
        if just_ended:
            just_ended = False
            if not line.strip() and out and not out[-1].strip():
                continue
        if _HEADING_RE.match(line) or _starts_source_block(line):
            in_block = True
            continue
        out.append(_scrub(line))
    return "\n".join(out)


def summary_only(text: str) -> str:
    """The response as the quality metrics should see it: bullets, no Sources."""
    return strip_sources_block(text)
