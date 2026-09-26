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
    [obsidian][<vault_name>][<model_name>][[Note Title]]: why it is relevant
    [obsidian][<vault_name>][unknown][[Older Note]]: provenance not recorded

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

__all__ = [
    "MODEL_TOKEN",
    "SOURCES_HEADING",
    "SOURCE_KIND",
    "UNKNOWN",
    "VAULT_TOKEN",
    "SourceEntry",
    "VaultIdentity",
    "mcp_vault_name_from_events",
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

#: Leading kind marker, kept stable so the block stays machine-greppable.
SOURCE_KIND = "obsidian"

#: Rendered for provenance that was never recorded. Never a guess.
UNKNOWN = "unknown"

# ``## Sources`` / ``### Sources`` / ``**Sources**`` / ``Sources:`` on its own line.
_HEADING_RE = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]*)?\**\s*sources\s*\**\s*:?[ \t]*$",
    re.IGNORECASE,
)
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\n]+?)\]\]")
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
    still produce (``[vault] [[Note]]``, ``- [[Note]]: reason``). The note is
    the first ``[[wikilink]]``; the reason is whatever follows it.
    """
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
    if _WIKILINK_RE.search(line) and _TOKEN_RE.search(line):
        return True
    return bool(_KIND_RE.match(line.strip()))


def _is_block_line(line: str) -> bool:
    """True for a line that belongs to a Sources block (list item, wikilink, kind tag)."""
    if not line.strip():
        return False
    return (
        bool(_BULLET_RE.match(line))
        or bool(_WIKILINK_RE.search(line))
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
    entries: Iterable[SourceEntry], vault: str, model: str
) -> list[str]:
    lines = [SOURCES_HEADING]
    for entry in entries:
        line = f"[{SOURCE_KIND}][{vault}][{model}][[{entry.note}]]"
        if entry.reason:
            line += f": {entry.reason}"
        lines.append(line)
    return lines


def render_sources(
    text: str,
    *,
    vault_name: str | None = None,
    model_name: str | None = None,
) -> str:
    """Re-emit the model's Sources block with the real vault and model names.

    Any ``**Sources**`` / ``## Sources`` heading the model produced is discarded
    and exactly one canonical block is written in its place, so a model that
    ignores the format can never yield duplicates. Identifiers that could not be
    resolved render as :data:`UNKNOWN`; they are never invented.

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

    out[anchor:anchor] = _format_block(deduped, vault, model)
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
