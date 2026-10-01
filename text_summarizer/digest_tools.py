"""The vault digest, as a tool the model can call -- and why it is its own module.

:mod:`text_summarizer.digest` reads the vault and knows nothing about agents: it is the
standard library plus two sibling modules, costs no quota, and a test
(``test_the_import_graph_reaches_neither_adk_nor_genai``) pins that. This module is
where ``google.adk`` is allowed in, and it exists for exactly that reason. Folding the
tool into ``digest.py`` would mean importing ``FunctionTool`` there, which would either
break that test or force its deletion -- and the property it protects is the whole basis
for the digest existing at all: a report on what was already learned must keep working
when the model cannot be reached.

Three deliberate differences from the CLI, because the two consumers are not the same:

* **Bounded.** ``--format text`` is read by a person at a terminal and the day is as
  long as the day was; a tool result lands in a context window that also has to hold the
  conversation, the retrieved notes and the answer. The lists are capped and the cap is
  *reported* -- see :func:`_payload` -- because a truncated list that does not say so
  reads as a quiet day, which is the single failure this module exists to avoid.
* **Errors are data.** ``web_search._error``'s reasoning applies verbatim: the model has
  to be able to read a failure and fall through to another tool. Raising kills the turn
  and reports nothing about why.
* **No fingerprint.** ``to_dict`` carries ``source_fingerprint`` because a *script*
  grouping a day's notes by what was actually asked wants it. Sixty-four hex characters
  help nobody reading a digest, and the model has no use for them, so they are dropped
  rather than paid for in context on every call.

What is *not* different: the date is validated by :func:`digest.iso_date`, the same
definition ``--date`` uses, and the vault is read through the same
:func:`digest.default_vault_root` attribute lookup. One definition of a valid day and
one binding of the vault root, or the tool and the command would disagree about both --
and the failure of that disagreement is an empty digest for a day nobody asked about,
which is indistinguishable from a day the agent learned nothing.
"""

from __future__ import annotations

import argparse
import os

from google.adk.tools import FunctionTool

from . import digest as digest_module

#: Notes, topics and chat entries returned per call. Counted, not measured in bytes,
#: so the bound is predictable and assertable; the real payload per note is a summary
#: plus a handful of bullets, so these land in the low thousands of tokens.
MAX_NOTES = 25
MAX_TOPICS = 25
MAX_CHAT = 10


def digest_enabled() -> bool:
    """Whether there is a vault to read.

    Gated, unlike the clock. The clock is ungated because a missing clock is a
    *silently wrong answer* -- "today's news" answered from a training cutoff -- while a
    missing vault is a missing capability, and every other toolset here already returns
    ``[]`` when its subject is not configured. The gate is also the honest one: with no
    vault on disk this tool can only ever report an empty day, and offering the model a
    tool whose only possible answer is "nothing" is worse than not offering it.

    The check is on the *resolved* root, which is the same binding
    ``save_summary_to_second_brain`` writes through, so this cannot disagree with the
    writer about which vault is in play.
    """
    return os.path.isdir(digest_module.default_vault_root())


def build_digest_tools() -> list:
    """The digest tool, or ``[]`` when there is no vault to read.

    Same shape as ``build_obsidian_tools``, ``build_gmail_tools`` and
    ``build_web_search_tools``: the gate lives entirely in here and the caller spreads
    the result unconditionally.
    """
    if not digest_enabled():
        return []
    return [FunctionTool(read_day_digest)]


def _payload(result: digest_module.Digest) -> dict:
    """The tool's view of a digest: :func:`digest.to_dict`, bounded and trimmed.

    The cap is reported rather than applied silently. ``note_count`` and ``topic_count``
    stay the *true* totals from :func:`digest.to_dict`, so a caller that gets 25 of 40
    notes can still say 40, and ``truncated`` names which list was cut and by how much.
    Losing the distinction between "25 notes were written" and "25 notes were written and
    15 more" is the difference between a digest and a lie.
    """
    payload = digest_module.to_dict(result)
    notes, dropped_notes = payload["notes"][:MAX_NOTES], len(payload["notes"]) - MAX_NOTES
    topics, dropped_topics = (
        payload["topics"][:MAX_TOPICS],
        len(payload["topics"]) - MAX_TOPICS,
    )
    payload["notes"], payload["topics"] = notes, topics

    # Counted *before* the slice, or the count is the number that survived.
    dropped_chat = 0
    if payload["chat"] is not None:
        dropped_chat = max(len(payload["chat"]) - MAX_CHAT, 0)
        payload["chat"] = payload["chat"][:MAX_CHAT]

    for note in notes:
        note.pop("fingerprint", None)

    truncated = {
        key: dropped
        for key, dropped in (
            ("notes", dropped_notes),
            ("topics", dropped_topics),
            ("chat", dropped_chat),
        )
        if dropped > 0
    }
    if truncated:
        payload["truncated"] = truncated
    return payload


def read_day_digest(day: str = "", include_chat: bool = False) -> dict:
    """Report what the summarizer wrote into the vault on one day.

    Reads the dated notes under "Second Brain/", the topic stubs they link to, the hub
    note's index position, and each note's recorded provenance (which model generated
    it, and in which vault). Pass `day` as YYYY-MM-DD; omit it for today. Nothing here
    costs a model call, so this is the cheap way to answer "what did you learn on
    Tuesday" -- and, because it reads the vault rather than your own notes of a
    conversation, it also covers summaries the agent wrote in sessions you never saw.

    A day with no notes is a real answer, reported as `note_count: 0`, not an error.
    Use `include_chat` only when the day's raw exchanges are wanted too: the chat log
    holds the user's own prompts and full answers, which is a different kind of record
    from a distilled summary and is off by default.

    A note recorded before the agent tracked provenance appears without a `provenance`
    field. That absence means the vault never said which model wrote it; do not fill it
    in with whichever model is running now.
    """
    # Only a genuinely absent value means "today". A blank one is a value the model got
    # wrong, and quietly substituting today's date for it is precisely the silent
    # substitution this check exists to prevent -- so it goes to digest.iso_date, which
    # strips and then rejects it, exactly as `--date "  "` does.
    raw = "" if day is None else str(day)
    requested = ""
    if raw != "":
        try:
            # digest.iso_date, not a second date parser. The CLI's `--date` and this
            # tool's `day` have to accept exactly the same strings: a `day` this
            # rejected would report nothing at all, and a date shape it accepted that
            # the CLI rejected would report a day the user never asked about. Both are
            # silent, so both are worth preventing structurally.
            requested = digest_module.iso_date(raw)
        except argparse.ArgumentTypeError as exc:
            return {
                "error": str(exc),
                "hint": (
                    "call current_datetime for today's date, then pass it as YYYY-MM-DD; "
                    "this tool does not understand 'yesterday' or 'last week'"
                ),
            }

    try:
        result = digest_module.build_digest(
            requested or None, include_chat=include_chat
        )
    except OSError as exc:
        # Same reasoning as web_search._error: the model must be able to read this and
        # fall through to another tool rather than have the turn die here.
        return {"error": f"could not read the vault: {exc}"}

    return _payload(result)