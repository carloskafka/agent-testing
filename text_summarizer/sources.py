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
from urllib.parse import quote, urlsplit, urlunsplit

from .chart import (
    _position_of as chart_position_of,
    render_svg as chart_render_svg,
    store_svg as chart_store_svg,
    svg_fingerprint as chart_svg_fingerprint,
)

__all__ = [
    "CHART_FENCE",
    "CHART_IMAGE_PREFIX",
    "CHART_TOKEN",
    "MODEL_TOKEN",
    "SOURCES_HEADING",
    "SOURCE_BULLET",
    "SOURCE_KIND",
    "UNKNOWN",
    "VAULT_TOKEN",
    "VAULT_WEB_PREFIX",
    "WEB_KIND",
    "WEB_TOKEN",
    "SourceEntry",
    "VaultIdentity",
    "mcp_vault_name_from_events",
    "normalised_url",
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

#: Sentinel standing in for the retrieval provider, for a ``[web]`` line. Same
#: collision-proof construction as the other two.
WEB_TOKEN = "@@ADK_WEB@@"

# One alternation over every sentinel. Built from a tuple rather than written out,
# so adding a token cannot leave a place that still only knows about two.
_TOKENS = (VAULT_TOKEN, MODEL_TOKEN, WEB_TOKEN)
_TOKEN_RE = re.compile("|".join(re.escape(token) for token in _TOKENS))

SOURCES_HEADING = "**Sources**"

#: One source per markdown list item. See the module docstring.
SOURCE_BULLET = "- "

#: Leading kind marker, kept stable so the block stays machine-greppable.
SOURCE_KIND = "obsidian"

#: A web page reached through the search tier. The second bracket then carries the
#: provider rather than a vault name -- slot two means "where this came from", and
#: both kinds answer that question the same way.
WEB_KIND = "web"

#: Every kind the grammar recognises. ``_KIND_RE`` is built from this, so a line
#: the renderer emits is always a line the parser can read back.
SOURCE_KINDS = (SOURCE_KIND, WEB_KIND)

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
_KIND_RE = re.compile(
    rf"^\[?(?:{'|'.join(re.escape(kind) for kind in SOURCE_KINDS)})\]?", re.IGNORECASE
)

# The web target, in the spelling the renderer emits: an autolink. Chosen over a
# bare URL and over ``[title](url)`` because it needs no bracket escaping, gives
# the re-parse one unambiguous delimiter, and survives the dev UI's `marked`
# sanitizer, whose URL pattern accepts the ``https:`` scheme form.
_WEB_TARGET_RE = re.compile(r"<((?:https?)://[^>\n]+)>|(https?://[^\s<>]+)")


@dataclass(frozen=True)
class SourceEntry:
    """One consulted source: its target plus an optional reason.

    ``note`` is a vault title for an ``obsidian`` entry and a URL for a ``web``
    one; ``kind`` says which, and both fields are defaulted so every existing
    construction site and test keeps working unchanged.
    """

    note: str
    reason: str = ""
    kind: str = SOURCE_KIND


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


def _line_kind(line: str) -> str:
    """Which kind this line claims to be, defaulting to ``obsidian``.

    Precedence matters, and it is not "URL wins":

    1. an explicit ``[kind]`` tag, so a rendered block re-parses with the kind it
       was written with;
    2. a ``[[wikilink]]``, because a wikilink is a vault target *by definition* --
       a note is titled ``[[https://example.com/p]]`` is a vault note about a URL,
       and treating it as a web citation would drop it from the block entirely
       when the URL was not in this turn's permitted set;
    3. a bare/autolinked URL, so a model that omits the tag still round-trips.
    """
    match = re.search(r"^\s*\[*\s*\[([a-z]+)\]", line, re.IGNORECASE)
    if match:
        candidate = match.group(1).casefold()
        if candidate in SOURCE_KINDS:
            return candidate
    if _WIKILINK_RE.search(line) or _LINKED_WIKILINK_RE.search(line):
        return SOURCE_KIND
    if _WEB_TARGET_RE.search(line):
        return WEB_KIND
    return SOURCE_KIND


def _parse_source_line(line: str) -> SourceEntry | None:
    """Parse one source line into a :class:`SourceEntry`.

    Accepts the canonical token form as well as legacy spellings the model may
    still produce (``[vault] [[Note]]``, ``- [[Note]]: reason``), and the linked
    spelling the renderer itself emits, so a rendered block re-parses into the
    same entries -- that is what keeps rendering idempotent. The note is the
    first ``[[wikilink]]``; the reason is whatever follows it.
    """
    kind = _line_kind(line)

    match = _LINKED_WIKILINK_RE.search(line)
    if match:
        # The href is the renderer's own, so it is dropped and the href is
        # recomputed from the vault on the next pass. Keeping it would let a
        # stale path survive a note being renamed.
        note = match.group(1).strip()
        reason = _REASON_SEP_RE.sub("", line[match.end() :].strip()).strip()
        return SourceEntry(note=note, reason=reason, kind=kind) if note else None

    match = _WIKILINK_RE.search(line)
    if match:
        note = match.group(1).strip()
        if not note:
            return None
        reason = _REASON_SEP_RE.sub("", line[match.end() :].strip()).strip()
        return SourceEntry(note=note, reason=reason, kind=kind)

    # A web source is a URL, and a URL has no [[wikilink]]. Without this branch
    # _parse_source_line returned None and the entry was dropped *silently* --
    # the answer would render with no citation and nothing in the trace to say
    # why. Both spellings are accepted: the autolink the renderer emits, and a
    # bare URL in case the model writes one.
    match = _WEB_TARGET_RE.search(line)
    if match:
        target = (match.group(1) or match.group(2) or "").strip()
        if match.group(1):
            # The angle-bracket form is unambiguous: everything up to `>` is the URL,
            # spaces included, so it round-trips verbatim.
            reason = _REASON_SEP_RE.sub("", line[match.end() :].strip()).strip()
        else:
            # A bare URL runs into the reason separator with no whitespace between
            # them ("https://x/p: because"), so `: ` is the only reliable cut. A URL
            # containing ": " cannot be written bare; the bracket form is the one the
            # renderer emits and the one that matters for idempotency.
            head, separator, tail = target.partition(": ")
            if separator:
                target, reason = head, tail.strip()
            else:
                reason = _REASON_SEP_RE.sub("", line[match.end() :].strip()).strip()
            target = target.rstrip(":|")
        if target:
            return SourceEntry(note=target, reason=reason, kind=WEB_KIND)
    return None


def _starts_source_block(line: str) -> bool:
    """True for a line that opens a Sources block without a heading."""
    if (
        _WIKILINK_RE.search(line) or _LINKED_WIKILINK_RE.search(line) or _WEB_TARGET_RE.search(line)
    ) and _TOKEN_RE.search(line):
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
        or bool(_WEB_TARGET_RE.search(line))
        or bool(_KIND_RE.match(line.strip()))
    )


def _scrub(line: str) -> str:
    """Remove leftover sentinel tokens from prose without reflowing it."""
    if not _TOKEN_RE.search(line):
        return line
    cleaned = re.sub(r"[ \t]{2,}", " ", _TOKEN_RE.sub("", line))
    return cleaned.rstrip()


# --- rendering ---------------------------------------------------------------


def normalised_url(url: str, *, keep_query: bool = True) -> str:
    """The canonical form of ``url``, or ``""`` when it is not an http(s) URL.

    Scheme and host lowercased, an empty path written as ``/``, a trailing ``/``
    dropped from a longer path, and the fragment dropped. A query parameter is kept
    by default: SearXNG's ``url`` and a tracking-free citation can differ by one,
    so both spellings are accepted (see :func:`_url_permitted`).

    Used on **both** sides of the permission check. Comparing one normalised form
    against a raw one is how a real citation gets silently dropped, which is
    exactly the failure this rule exists to make safe.
    """
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return ""
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return ""
    try:
        host = (parts.hostname or "").lower()
    except ValueError:
        return ""
    if not host:
        return ""
    port = parts.port
    netloc = f"{host}:{port}" if port else host
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query if keep_query else "", ""))


def _url_permitted(url: str, permitted: set[str]) -> bool:
    """Whether ``url`` is one the search tier returned.

    Compared on the normalised form, and additionally without the query string,
    because a tracking parameter does not change which page was read -- so
    ``example.com/p?utm_source=x`` still matches a permitted ``example.com/p``.
    """
    candidate = normalised_url(url)
    if not candidate:
        return False
    if candidate in permitted:
        return True
    return normalised_url(url, keep_query=False) in permitted


def _dedupe(entries: Iterable[SourceEntry]) -> list[SourceEntry]:
    """Drop repeats of the same source.

    Keyed on ``(kind, target)``, not the target alone: a URL and a note title can
    legitimately be the same string, and collapsing them would lose a source. The
    URL is compared case-insensitively because the host is, but the path may not
    be -- so the whole target is folded rather than just its scheme.
    """
    seen: set[tuple[str, str]] = set()
    out: list[SourceEntry] = []
    for entry in entries:
        key = (entry.kind, entry.note.casefold())
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
    web_provider: str = UNKNOWN,
) -> list[str]:
    """One heading, one bullet per source, one kind tag per line.

    A ``web`` entry is emitted as ``[web][<provider>][<model>]<url>``: slot two
    carries the provider instead of a vault name, because slot two means "where
    this came from" and that is the honest answer for a web page.

    Web URLs deliberately do **not** go through ``_href_from_index``/``note_href``.
    Those derive a vault path from a *title* and are pinned against traversal by
    four spellings in ``test_serve.py``; a web citation's URL is its own, there is
    nothing to recompute, and a title-shaped URL would otherwise be turned into a
    link to a note that does not exist.
    """
    lines = [SOURCES_HEADING]
    for entry in entries:
        if entry.kind == WEB_KIND:
            target = f"<{entry.note}>"
            origin = web_provider
            kind = WEB_KIND
        else:
            target = f"[[{entry.note}]]"
            href = _href_from_index(entry.note, index) if index else None
            if href:
                # Backslash-escaped so the title still *displays* as [[Note]] while
                # being a link; see _LINKED_WIKILINK_RE.
                target = rf"[\[\[{entry.note}\]\]]({href})"
            origin = vault
            kind = SOURCE_KIND
        line = f"{SOURCE_BULLET}[{kind}][{origin}][{model}]{target}"
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
    web_provider: str | None = None,
    allowed_web_urls: Iterable[str] | None = None,
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

    ``web_provider`` names the retrieval tier for a ``[web]`` line, resolved in
    code like the vault and the model. ``allowed_web_urls`` is the set of URLs the
    search tier actually returned this turn; a ``[web]`` line citing anything else
    is dropped. That is the same rule already applied to vault notes -- *"a link
    is never emitted for a note that is not there, because a dead link asserts the
    note exists when it does not"* -- and without it this feature would print
    invented URLs on every hallucinated claim. Pass ``None`` to accept every web
    line; pass an empty collection to accept none.

    Returns the input unchanged (modulo stray sentinel tokens) when it contains
    no Sources block.
    """
    text = text or ""
    if not text.strip():
        return text

    vault = _sanitize_identifier(vault_name or "") or UNKNOWN
    model = _sanitize_identifier(model_name or "") or UNKNOWN
    provider = _sanitize_identifier(web_provider or "") or UNKNOWN
    permitted = (
        None
        if allowed_web_urls is None
        else {normalised_url(url) for url in allowed_web_urls} - {""}
    )

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

    if permitted is not None:
        entries = [e for e in entries if e.kind != WEB_KIND or _url_permitted(e.note, permitted)]

    deduped = _dedupe(entries)
    if not deduped:
        # A heading with no usable source line: drop it rather than render an
        # empty section.
        return "\n".join(out).rstrip()

    # One walk of the vault for the whole block, not one per source line.
    index = _build_title_index(vault_root) if vault_root else None
    out[anchor:anchor] = _format_block(deduped, vault, model, index, provider)
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
    """Summary text with charts and sources removed."""
    return strip_chart_block(strip_sources_block(text))
