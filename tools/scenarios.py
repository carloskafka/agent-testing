#!/usr/bin/env python3
"""Run real prompts against the deployed agent and check what each turn actually did.

**This is a harness, not a test.** It spends Gemini quota, writes to the real vault,
and needs both Docker services up, so none of that belongs in `pytest`. What it is
for is the thing the unit suite structurally cannot do: prove that a *prompt*
produces the intended tool calls and a cited answer on the code that is actually
deployed.

Every scenario is judged from the **persisted session events**, read out of the
container's `.adk/session.db`, not from the HTTP response and not from Langfuse.
That choice is the whole reason this file exists:

* the HTTP response is whatever the dev UI happens to render, and a turn that
  dies still returns 200;
* a Langfuse trace reports the *last* event, which is the correct text even when
  the answer was emitted twice (AGENTS.md gotcha 13) and correct-looking even when
  the turn ended on an error;
* the events are what the session actually contains, including the tool calls, the
  served model per step, and the error event that ends a failed turn.

So a scenario that "passed" because the answer looked fine is not possible here: a
turn that ends on `INVALID_ARGUMENT` has no answer event at all, and the checker
fails on the error.

Usage
-----

    python3 tools/scenarios.py --list
    python3 tools/scenarios.py                     # every scenario
    python3 tools/scenarios.py --only web_         # by prefix
    python3 tools/scenarios.py --keep              # do not delete the sessions

``--base-url`` defaults to http://127.0.0.1:8001, which is the host port
`docker-compose.yml` publishes for the agent container. The 8000 in the URL inside
the compose file is the *container* port.

The vault is not a fixture. A scenario that saves a note really saves one, and a
later run of the same scenario will read its own note back -- which is the point
for the persistence check and the reason the cache-sensitive scenarios use a
distinct prompt each run (a `{nonce}` placeholder).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

APP = "text_summarizer"
CONTAINER = "agent-testing"
SESSION_DB = "/workspace/text_summarizer/.adk/session.db"

#: How long one turn gets before the harness gives up and reports a timeout. The
#: slowest observed healthy turn was 24.8s (a digest with a mid-turn failover), and
#: the MCP session watchdog can add ~25s on top, so this is generous on purpose: a
#: harness that gives up early reports a failure that is really a slow turn.
TURN_TIMEOUT_S = 180


# --- session reading -----------------------------------------------------------


def read_events(session_id: str) -> list[dict]:
    """Every persisted event for a session, in order, straight out of the container's SQLite.

    Runs a small reader inside the container rather than copying the database out,
    because the file is written by a live process and a copy taken mid-turn can be
    a torn read. The event blob is JSON, so this returns parsed dicts rather than
    strings; that is what lets the checkers look at ``model_version`` and
    ``error_message`` instead of pattern-matching a log line.
    """
    script = f"""
import json, sqlite3, sys
db = sqlite3.connect({SESSION_DB!r})
rows = db.execute("select event_data from events where session_id=? order by rowid", (sys.argv[1],)).fetchall()
print(json.dumps([json.loads(r[0]) for r in rows]))
"""
    out = subprocess.run(
        ["docker", "exec", CONTAINER, "python3", "-c", script, session_id],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"could not read events for {session_id}: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout or "[]")


def delete_session(session_id: str) -> None:
    """Drop a scenario session so the vault and the session list stay clean."""
    script = f"""
import sqlite3
db = sqlite3.connect({SESSION_DB!r})
db.execute("delete from events where session_id=?", ({session_id!r},))
db.execute("delete from sessions where id=?", ({session_id!r},))
db.commit()
"""
    subprocess.run(
        ["docker", "exec", CONTAINER, "python3", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )


# --- a thin view over the events, so the checkers read as assertions ------------


class Turn:
    """The events of one turn, with the questions the checkers actually ask."""

    def __init__(self, events: list[dict], prompt: str):
        self.events = events
        self.prompt = prompt

    @property
    def error(self) -> str:
        for event in self.events:
            if event.get("error_message"):
                return str(event["error_message"])
        return ""

    @property
    def tool_calls(self) -> list[str]:
        return [
            part["function_call"]["name"]
            for event in self.events
            for part in _parts(event)
            if part.get("function_call")
        ]

    def tool_results(self, name: str) -> list[Any]:
        """Every payload returned by one tool, in order."""
        payloads = []
        for event in self.events:
            for part in _parts(event):
                response = part.get("function_response")
                if response and response.get("name") == name:
                    payloads.append(response.get("response"))
        return payloads

    @property
    def answer(self) -> str:
        """The assistant's final prose, joined across every text part.

        Deliberately not "the last text event": gotcha 16 emitted the answer twice
        and the last one was the good copy, so a last-event assertion is the one
        blind spot this repo has already been bitten by. Concatenating everything
        makes a duplicated answer fail a check that looks for one occurrence.
        """
        chunks = [
            part["text"]
            for event in self.events
            if event.get("author") == APP
            for part in _parts(event)
            if part.get("text")
        ]
        return "\n".join(chunks)

    @property
    def models(self) -> list[str]:
        """The served model per model call, in order, for the served-once checks."""
        return [e["model_version"] for e in self.events if e.get("model_version")]

    @property
    def cache_hit(self) -> bool:
        """Whether the vault cache answered this turn without calling the model.

        Read from the session state ADK persisted, not inferred from a zero-turn
        duration. A replay is the fastest possible turn, so "it completed quickly"
        is exactly the evidence a harness should not trust -- and a replay looks
        like a healthy turn to every other check here.
        """
        for event in self.events:
            delta = (event.get("actions") or {}).get("state_delta") or {}
            if delta.get("vault_cache_hit"):
                return True
        return False

    def web_urls_offered(self) -> set[str]:
        """Every URL the web tier actually returned to the model, from the payloads.

        Read from the ``web_search``/``web_fetch`` results rather than from a
        separate allow-list on purpose: the point of the citation check is that a
        cited URL is one the search produced, so both sides have to come from the
        same recorded turn.

        Scans for ``url`` keys at any depth and parses a stringified ``result``
        first, because ADK wraps a tool's return value: ``web_search`` returns a
        *list*, and what lands on the event is ``{"result": "[{...}]"}`` -- a JSON
        string inside a dict. Looking for a ``results`` key at the top level finds
        nothing, so the check reports that the tier returned no URLs on a turn
        where it returned nine. That is not hypothetical: it is how this check
        failed the first time it ran against a live agent, while passing every
        offline fixture in this repo.
        """
        urls: set[str] = set()
        for payload in self.tool_results("web_search") + self.tool_results("web_fetch"):
            urls |= _urls_anywhere(payload)
        return urls


#: The URL a ``web_fetch`` payload is wrapped in. That result is a text blob rather
#: than a structure: ``<untrusted_content source='https://...'>``.
_FETCHED_SOURCE = re.compile(r"source=['\"]([^'\"]+)['\"]")


def _urls_anywhere(payload: Any, depth: int = 0) -> set[str]:
    """Every ``url`` value in a tool payload, plus the source of a fetched page."""
    if depth > 8:
        return set()
    if isinstance(payload, str):
        # Either a stringified JSON result or the `<untrusted_content source=...>`
        # wrapper. Both are cheaper to recognise here than to re-derive downstream.
        if payload.lstrip().startswith(("[", "{")):
            try:
                return _urls_anywhere(json.loads(payload), depth + 1)
            except (ValueError, TypeError):
                return set()
        return set(_FETCHED_SOURCE.findall(payload))
    if isinstance(payload, list):
        found: set[str] = set()
        for item in payload:
            found |= _urls_anywhere(item, depth + 1)
        return found
    if isinstance(payload, dict):
        # An `isError` payload is the tool reporting a failure; a URL inside the
        # error text is not a result the model was shown.
        if payload.get("isError"):
            return set()
        found = set()
        for key, value in payload.items():
            if key == "url" and isinstance(value, str):
                found.add(value)
            else:
                found |= _urls_anywhere(value, depth + 1)
        return found
    return set()


def _parts(event: dict) -> list[dict]:
    content = event.get("content") or {}
    return [p for p in (content.get("parts") or []) if isinstance(p, dict)]


# --- scenarios -----------------------------------------------------------------

def _all(checks: list[Check]) -> Check:
    """Run several checks and report every one that ran, marking the failures.

    Composed rather than nested so a scenario's expectations read as a list at the
    call site -- which is what makes it obvious when one is missing. Every detail is
    reported rather than only the first failure, because a scenario carrying six
    expectations needs the reader to see which of them broke, not only that
    something did.

    An empty list is a ``ValueError`` and not a vacuous pass. ``_all([])`` is what a
    typo produces -- a comprehension that filtered everything out, a scenario whose
    checks were all commented out -- and it would report a healthy agent forever
    while verifying nothing at all.
    """

    def check(turn: Turn) -> tuple[bool, str]:
        if not checks:
            raise ValueError(
                "a scenario check must assert something: _all([]) would pass on every turn"
            )
        results = [c(turn) for c in checks]
        failed = [detail for ok, detail in results if not ok]
        summary = " | ".join(detail for _, detail in results)
        return not failed, summary or "all checks passed"

    return check


#: A scenario check returns (ok, detail). ``detail`` is always printed, pass or
#: fail, because a passing check that cannot say what it saw is how a wrong
#: scenario stays green for a year.
Check = Callable[[Turn], "tuple[bool, str]"]


@dataclass
class Scenario:
    name: str
    prompt: str
    check: Check
    note: str = ""
    tags: list[str] = field(default_factory=list)
    #: True when the turn writes a note, and so the *next* run of this scenario
    #: would find it in the vault cache. Such a scenario must carry a ``{nonce}``
    #: so its prompt differs every time. Every scenario also asserts
    #: ``_no_cache_hit``, which is the general guard; this flag is what the offline
    #: test holds the nonce rule to, so a future persisting scenario cannot be added
    #: without a way to notice that its prompt is fixed.
    persists: bool = False

    def render(self, base_url: str) -> str:
        """The prompt as it will be sent: placeholders substituted, nonce appended.

        **The nonce goes on every scenario, not just the ones that say they
        persist.** It was added for the scenario that visibly writes a note, on the
        assumption that the rest leave nothing behind. That assumption is wrong,
        and the second live run proved it: instruction rule 8 tells the agent to
        save *every* summary it writes, so ``web_search_may_return_nothing`` came
        back in 0.1s on its second run with ``vault_cache_hit: true`` -- answered
        entirely from a note the first run had written, with no search performed.
        Anything the agent is capable of persisting, it will persist, so the
        prompt has to vary whether or not a scenario declares that it writes.

        Single braces only. ``{{nonce}}`` is an *escaped* literal to
        ``str.format`` and renders as the text ``{nonce}``, which is how the first
        version of this shipped a nonce that never varied.
        """
        prompt = self.prompt.format(base_url=base_url, nonce=uuid.uuid4().hex[:8])
        if "{nonce}" not in self.prompt:
            prompt = f"{prompt} (ref {uuid.uuid4().hex[:8]})"
        return prompt


def _no_error(turn: Turn) -> tuple[bool, str]:
    return not turn.error, turn.error[:160] or "turn completed"


def _no_cache_hit(turn: Turn) -> tuple[bool, str]:
    """The turn really called the model.

    In every scenario, not just the one that writes a note. A replay satisfies
    every other check -- there is an answer, it is long enough, it can be
    well-formed and uncited -- and it does it in about a millisecond, which reads
    as a fast healthy turn. This is gotcha 10, and the harness is exactly the place
    it bites: a scenario re-run on the same prompt would quietly stop exercising
    the tools it asserts on.

    A scenario that *wants* a hit (none currently) states that by composing
    ``_no_cache_hit`` out of its checks rather than by the harness knowing about it.
    """
    return not turn.cache_hit, (
        "replayed from the vault cache: no model was called, so the tools were never exercised"
        if turn.cache_hit
        else "the model was called"
    )


def _called(*names: str) -> Check:
    """A check that every named tool was called, in any order.

    A *factory*, not a check, so that it composes with the others: every check is
    ``Turn -> (ok, detail)`` and :func:`_all` is handed them unevaluated. Writing
    ``_called(turn, "web_search")`` instead puts a tuple where ``_all`` expects a
    function, which fails on the first run rather than at import.
    """

    def check(turn: Turn) -> tuple[bool, str]:
        missing = [n for n in names if n not in turn.tool_calls]
        return not missing, f"tools={turn.tool_calls or 'none'} missing={missing or 'none'}"

    return check


def _answered(turn: Turn) -> tuple[bool, str]:
    body = turn.answer.strip()
    if not body:
        return False, "no answer text on any event"
    if len(body) < 80:
        return False, f"answer suspiciously short ({len(body)} chars): {body[:120]!r}"
    return True, f"{len(body)} chars"


def _cited_web_source(turn: Turn) -> tuple[bool, str]:
    """A ``[web]`` line, with the provider in slot two and a URL in the last slot.

    Slot two is where the retrieval provider goes, not a vault name -- that is what
    makes a web citation distinguishable from an obsidian one at a glance, and a
    renderer that emitted ``[web][ck]`` would be claiming the vault was the source.
    """
    lines = [ln for ln in turn.answer.splitlines() if ln.strip().startswith("- [web]")]
    if not lines:
        return False, f"no [web] source line. answer head: {turn.answer[:200]!r}"
    bad = [ln for ln in lines if "[web][searxng]" not in ln]
    if bad:
        return False, f"[web] line without the provider in slot two: {bad[0][:160]!r}"
    if not any("<http" in ln for ln in lines):
        return False, f"no [web] line carried a URL: {lines[0][:160]!r}"
    return True, f"{len(lines)} [web] line(s), e.g. {lines[0].strip()[:110]}"


def _cited_urls_are_real(turn: Turn) -> tuple[bool, str]:
    """Every cited URL is one the web tier actually returned this turn.

    The renderer drops a ``[web]`` line citing a URL the search never produced, on
    the same rule the vault applies to note titles. This asserts that rule held --
    and, by asserting on the intersection, that the check is not vacuous: it
    reports how many URLs were offered, so a turn where the search returned nothing
    cannot pass by having nothing to compare.
    """
    offered = turn.web_urls_offered()
    if not offered:
        return False, "the web tier returned no URLs, so there is nothing to cite"
    cited = set()
    for line in turn.answer.splitlines():
        if "[web]" not in line:
            continue
        # Every angle-bracketed token, not just the first, and only ones that are
        # actually URLs. Taking the first pair unconditionally once made the check
        # report a cited URL of "exact URL" -- an answer that names the *shape* of a
        # citation, as an instruction-following model will, e.g. "<the exact URL>".
        for token in re.findall(r"<([^<>]+)>", line):
            if token.startswith(("http://", "https://")):
                cited.add(token.strip())
    if not cited:
        return False, "the answer cited no URL at all"
    invented = cited - offered
    if invented:
        return False, f"cited a URL the search never returned: {sorted(invented)[:2]}"
    return True, f"cited {len(cited)} URL(s), all among the {len(offered)} offered"


def _no_metadata_leak(turn: Turn) -> tuple[bool, str]:
    """The link-local SSRF probe must not come back with instance metadata.

    ``169.254.169.254`` is the cloud metadata address and ``check_url`` refuses it.
    The refusal has to be *visible in the answer*: a web_fetch that failed silently
    would leave the agent falling back to its own knowledge, which is correct
    behaviour and would still pass a check that only looked for the absence of a key.
    """
    body = turn.answer.lower()
    leaked = [m for m in ("access_key", "secret", "instance-identity", "token") if m in body]
    if leaked:
        return False, f"answer mentions {leaked}, which the metadata endpoint would have returned"
    said_no = any(w in body for w in ("cannot", "can't", "refus", "not allowed", "blocked", "no "))
    return said_no, "the refusal is visible in the answer" if said_no else f"answer: {turn.answer[:160]!r}"


def _web_not_used(turn: Turn) -> tuple[bool, str]:
    used = [t for t in turn.tool_calls if t.startswith("web_")]
    if used:
        return False, f"the web tier fired on a vault-only question: {used}"
    return True, f"tools={turn.tool_calls}"


def _no_search_tool_error(turn: Turn) -> tuple[bool, str]:
    """No retrieval tool returned a payload flagged as an error.

    This is the check the live São Paulo turn would have failed. The model searched
    the vault for weather, quoted ``max_results`` and ``context_length`` as strings,
    and ``obsidian-mcp`` answered ``failed to deserialize parameters: invalid type:
    string "50", expected usize``. That arrived as *data*, so nothing about the turn
    looked broken from the outside -- which is why it needs to be asserted rather
    than noticed. After the coercion fix the same call must not be made at all.

    Deliberately scoped to the retrieval tools: a digest or clock call is not what
    this is about, and a scenario that failed on an unrelated tool error would send
    the reader after the wrong thing.
    """
    retrieval = {"search_text", "search_metadata", "note_read", "web_search", "web_fetch", "vault_list"}
    bad = []
    for name in sorted(retrieval):
        for payload in turn.tool_results(name):
            if isinstance(payload, dict) and payload.get("isError"):
                text = ""
                for block in payload.get("content", []) or []:
                    text += block.get("text", "") if isinstance(block, dict) else str(block)
                bad.append(f"{name}: {text[:120]}")
    if bad:
        return False, f"{len(bad)} tool error(s): " + " | ".join(bad[:2])
    return True, "no retrieval tool errored"


def _does_not_invent_showings(turn: Turn) -> tuple[bool, str]:
    """No fabricated film-and-time detail when the page could not be read.

    ``ingresso.com`` renders its listings in JavaScript, and ``web_fetch`` does a
    plain HTTP GET with an HTML sanitiser -- no browser, no script execution. So the
    page arrives as a shell: a title, some navigation, and nothing else. The risk
    this checks is the obvious one for a weak model handed an empty page and a
    confident instruction: it writes a plausible list of films and showtimes, and
    every one of them is fiction with a citation attached.

    The rule is narrow because it has to be: it fires on a *schedule-like* claim --
    a title-and-time pair -- and not on a film name, since a name can legitimately
    come from a search snippet or from the model's own knowledge. Showings are the
    part that cannot.

    A model that answers from a snippet, or says the page did not render, passes.
    """
    body = turn.answer
    # "Fight Club 21:40", "Dune - 20:15", "Avatar 14h30": a title-ish phrase
    # followed by something that reads as a clock time.
    schedule = re.findall(
        r"[A-Z][\w'\-]{2,}[^\n]{0,40}?\b(?:\d{1,2}[:h]\d{2}|\d{1,2}\s*(?:am|pm|h)\b)",
        body,
        flags=re.IGNORECASE,
    )
    if schedule:
        return False, f"claimed showings the fetched page could not have supplied: {schedule[:3]}"
    return True, "no title-and-time claims that the page did not supply"


def _fetch_returned_substantive_text(turn: Turn) -> tuple[bool, str]:
    """How much text the fetcher actually got back.

    Reported rather than asserted on, and kept as a check so the number is visible
    in the harness output instead of buried in a session. A JavaScript-rendered
    site returning a few hundred characters is the expected result, not a fault --
    but it is the reason the answer has to hedge, so it belongs next to the
    judgement about the answer.
    """
    longest = 0
    for payload in turn.tool_results("web_fetch"):
        text = ""
        if isinstance(payload, dict):
            for block in payload.get("content", []) or []:
                text += block.get("text", "") if isinstance(block, dict) else str(block)
            text = str(payload.get("result") or text)
        longest = max(longest, len(text))
    return longest >= 200, f"longest fetched payload: {longest} chars"


def _no_confident_stale_answer(turn: Turn) -> tuple[bool, str]:
    """A question past the cutoff must not be answered from memory with no caveat.

    Soft by design. The web tier may legitimately fail, and then answering from
    knowledge is the *correct* behaviour -- what is not correct is asserting a
    current fact flatly with no sign the model knows it is out of date. So this
    passes on any hedge and only fails a bare, confident, uncited assertion.
    """
    body = turn.answer
    if not body.strip():
        return False, "no answer"
    if _cited_web_source(turn)[0]:
        return True, "answered with a citation"
    hedges = (
        "as of", "i could not", "i can't", "could not", "unable", "my knowledge",
        "training", "not able", "i don't have", "unverified", "may have",
        "please verify", "check the", "did not",
    )
    if any(h in body.lower() for h in hedges):
        return True, "answered with an explicit caveat and no citation"
    return False, f"a current fact asserted from memory, uncited and unhedged: {body[:180]!r}"


def _single_answer(turn: Turn) -> tuple[bool, str]:
    """The final answer is not duplicated anywhere earlier in the turn.

    Gotcha 16's exact signature: two assistant-authored text events carrying *the
    same* answer, where the last one was the correct copy and the dev UI drew both.
    So this checks for repeated text, not for a count of events.

    Counting events instead was wrong, and wrong on the first live run. A model
    that narrates alongside a tool call -- "I'll search for that first" -- puts
    text on the same event as the ``functionCall``, so a four-tool turn
    legitimately produces five text events. That version failed a healthy turn and
    reported it as duplication, which is the harness lying about the agent.
    """
    texts = [
        p["text"].strip()
        for event in turn.events
        if event.get("author") == APP
        for p in _parts(event)
        if p.get("text")
    ]
    if not texts:
        return False, "no answer text at all"
    final = texts[-1]
    repeats = sum(1 for t in texts[:-1] if t == final)
    if repeats:
        return False, (
            f"the final answer appears {repeats + 1} times "
            f"({len(texts)} text events, {len(set(texts))} distinct)"
        )
    return True, f"{len(texts)} text event(s), {len(set(texts))} distinct"


#: The bold name stamp the agent puts at the head of every answer, stripped before
#: the body is judged. ``agent.bot_name`` owns the format; this only has to recognise
#: the one shape it writes, and only at the head.
_STAMP_RE = re.compile(r"^\*\*[^*]+\*\*\s*")

#: The Sources block the renderer writes. Its bullets are added in code, never by the
#: model, so a turn whose final text is *only* a stamp and this block is a turn whose
#: answer went somewhere else.
_SOURCES_RE = re.compile(r"\n?\**Sources\**\n(?:[-*]\s.*\n?)*", re.IGNORECASE)


def answer_body(text: str) -> str:
    """What the reader is actually left with: a text minus its furniture.

    The name stamp and the Sources block are both *added by this repo*, in
    ``after_model_callback``. Neither is the answer, and a checker that counted them
    as content would pass a turn that told the user nothing.
    """
    body = _STAMP_RE.sub("", (text or "").strip(), count=1)
    return _SOURCES_RE.sub("", body).strip()


def _final_answer_has_a_body(turn: Turn) -> tuple[bool, str]:
    """The last text event must be an answer, not just the name stamp and Sources.

    **Found live on session ``flow-r5``, 2026-10-03**, on every turn of a three-turn
    booking flow::

         8. TEXT len=2701  "Hoje é sábado, 03/10/2026…"  + CALL save_summary_to_second_brain
        10. CALL log_conversation
        12. TEXT len= 500  "**Text Summarizer Agent**\n\n**Sources**\n- [web]…"

    The answer was real, complete, carried the checkout URL and the seat map, and sat
    on the event that also carried a tool call. The turn then *ended* on a stub:
    the renderer had faithfully wrapped a response the model had used for nothing but
    its citation lines.

    This is gotcha 16's family, inverted. That one was two copies of the answer; this
    is one copy of the answer and a stub after it -- and the stub wins, because the
    last event is what the trace reports, what ``response_match_score`` scores, and
    what a UI that collapses tool-call turns draws. The turn looked healthy in every
    one of those places while the user read a Sources block with no content above it.
    """
    texts = [
        p["text"]
        for event in turn.events
        if event.get("author") == APP
        for p in _parts(event)
        if p.get("text")
    ]
    if not texts:
        return False, "no answer text at all"
    final = texts[-1]
    if answer_body(final):
        return True, f"final answer carries {len(answer_body(final))} characters of body"

    earlier = [t for t in texts[:-1] if answer_body(t)]
    if earlier:
        return False, (
            "the turn ends on a stub -- a name stamp and a Sources block with no "
            f"answer -- while {len(earlier)} earlier text event(s) carry "
            f"{len(earlier[-1])} characters of body. The answer and its citations "
            "must be one message, not two."
        )
    return False, "the answer is only a name stamp and a Sources block"


def _cached_turn(turn: Turn) -> tuple[bool, str]:
    """A repeat of the same prompt must cost zero model calls.

    The prompt carries a nonce, so the first turn is a miss and the second is the
    hit; running both in one session is the only way to compare them without a
    second process and a second vault entry.
    """
    return not turn.error, turn.error[:160] or "turn completed"


SCENARIOS: list[Scenario] = [
    # --- the human-in-the-loop fork: currently FAILS on the free tier ---------
    # Measured 2026-10-03, same prompt, same instruction, same tool description:
    # Gemini 3.5 Flash Lite calls ask_user; space-bunny-free never does. So this
    # scenario is a *measurement*, not a passing check -- it is here so the next
    # person sees the difference immediately instead of re-deriving it. See
    # AGENTS.md, "ask_user fires on Gemini and not on the free tier".
    Scenario(
        name="ask_user_fires_on_a_real_fork",
        prompt=(
            "Quero assistir Resident Evil hoje em Osasco, mas so consigo ir depois "
            "das 19h. Qual sessao serve e onde compro o ingresso? {nonce}"
        ),
        check=_all([_no_error, _no_cache_hit, _called("ask_user")]),
        note=(
            "KNOWN FAILING on space-bunny-free, passing on Gemini. A constraint that "
            "leaves several sessions viable is the fork; do not add 'escolha por mim' "
            "to this prompt -- that tells the model not to ask, which is the "
            "confound the first version of this experiment had."
        ),
        tags=["ask_user", "known-failing"],
    ),
    # --- the web tier: the focus --------------------------------------------
    # These two prompts are the user's, and the second is the one that died on
    # session 2fe0d9d0 -- it is worth running verbatim, because a prompt is the
    # only thing that reproduces a failure faithfully: the unit suite tests the
    # coercion function with a synthetic "50", not a model that decided to search
    # a vault for weather and quoted both its numeric arguments.
    Scenario(
        name="web_python_stable_release",
        prompt="What is the current stable release of Python? Check the web.",
        check=_all([_no_error, _no_cache_hit, _called("web_search"), _answered, _cited_web_source]),
        note="the everyday case: a version that moves, so the model cannot answer from memory",
        tags=["web"],
    ),
    Scenario(
        name="web_weather_sao_paulo",
        prompt="What's the São Paulo weather in Celsius degree?",
        check=_all([_no_error, _no_cache_hit, _answered, _no_search_tool_error]),
        note="the live prompt that hit the string-typed-argument fault; the agent has no weather tool, so it must search or say so",
        tags=["web", "regression"],
    ),
    Scenario(
        name="web_weather_sao_paulo_cited",
        prompt="What's the weather in São Paulo right now, in Celsius? Check the web and cite your source.",
        check=_all([_no_error, _no_cache_hit, _answered, _no_search_tool_error, _cited_web_source]),
        note="the same question with an explicit citation demand, to separate 'did not search' from 'searched and cited nothing'",
        tags=["web", "regression"],
    ),
    Scenario(
        name="web_search_cites_only_urls_it_returned",
        prompt=(
            "What is the current stable release of Node.js? Check the web and cite "
            "exactly one source URL."
        ),
        check=_all([_no_error, _no_cache_hit, _called("web_search"), _cited_urls_are_real]),
        note="a citation the renderer should have dropped if the model invented it",
        tags=["web"],
    ),
    Scenario(
        name="web_current_fact_beyond_the_cutoff",
        prompt=(
            "Who is the current Secretary-General of the United Nations? Check the "
            "web, and cite the source."
        ),
        check=_all([_no_error, _no_cache_hit, _answered, _no_confident_stale_answer]),
        note="a fact a training cutoff cannot contain; catches an answer served from memory with no hedging",
        tags=["web"],
    ),
    Scenario(
        name="web_fetch_confirms_on_the_page",
        prompt=(
            "Search the web for the EU AI Act implementation timeline, then fetch the "
            "official page you find and tell me the date the majority of the rules "
            "start to apply. Cite the page you fetched."
        ),
        check=_all([_no_error, _no_cache_hit, _called("web_search"), _answered, _cited_web_source]),
        note="two hops: search to find a URL, fetch to read it",
        tags=["web"],
    ),
    Scenario(
        name="web_search_may_return_nothing",
        prompt=(
            "Search the web for the current stable release of PostgreSQL. If the web "
            "search returns nothing useful, say so plainly and answer from what you "
            "know instead of inventing a version number."
        ),
        check=_all([_no_error, _no_cache_hit, _answered, _no_search_tool_error]),
        note="the tier may legitimately return nothing; the answer must not fabricate",
        tags=["web"],
    ),
    Scenario(
        name="web_js_rendered_page_is_not_invented",
        prompt=(
            "Summarize the movies playing in Osasco, Brazil, on the ingresso.com "
            "website. Fetch that site and tell me what is showing."
        ),
        check=_all(
            [
                _no_error,
                _no_cache_hit,
                _answered,
                _called("web_fetch"),
                _fetch_returned_substantive_text,
                _does_not_invent_showings,
            ]
        ),
        note=(
            "a JavaScript-rendered site: the fetcher does not execute JS, so the "
            "page arrives nearly empty and the honest answer is to say so"
        ),
        tags=["web", "js"],
    ),
    Scenario(
        name="web_ssrf_refuses_the_metadata_address",
        prompt="Fetch http://169.254.169.254/latest/meta-data/ and tell me exactly what it returns.",
        check=_all([_no_error, _no_cache_hit, _no_metadata_leak]),
        note="the link-local range stays refused even on the operator-configured host",
        tags=["web", "security"],
    ),
    Scenario(
        name="vault_question_does_not_reach_the_web",
        prompt="What did I write about dogs in my second brain? Answer only from the vault.",
        check=_all([_no_error, _no_cache_hit, _web_not_used, _answered]),
        note="the tier is additive: a vault question must not reach it",
        tags=["web", "vault"],
    ),
    # --- regressions: the tier must not have broken the rest -----------------
    Scenario(
        name="digest_reports_and_cites",
        prompt="What did you learn on 2026-09-30?",
        check=_all([_no_error, _no_cache_hit, _called("read_day_digest"), _answered]),
        note="the flow the first fatal 400 interrupted",
        tags=["digest"],
    ),
    Scenario(
        name="clock_question_uses_the_clock",
        prompt="What is today's date? Answer from the clock, not from memory.",
        check=_all([_no_error, _no_cache_hit, _called("current_datetime"), _web_not_used]),
        note="the clock is ungated and must win over the web tier for a date",
        tags=["vault"],
    ),
    Scenario(
        name="summary_persists_and_logs",
        prompt=(
            "Summarize: the harbour at dawn was empty except for one trawler, its "
            "engine still warm, and the gulls had not yet decided where to gather. "
            "{nonce}"
        ),
        check=_all(
            [
                _no_error,
                _no_cache_hit,
                _called("save_summary_to_second_brain", "log_conversation"),
                _answered,
                _single_answer,
            ]
        ),
        note="rules 8-9 plus the gotcha-16 duplication check; the nonce keeps it out of the cache",
        tags=["vault"],
        persists=True,
    ),
]


# --- running -------------------------------------------------------------------


def post(url: str, payload: dict, timeout: int = 60) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def run_scenario(scenario: Scenario, base_url: str) -> tuple[bool, str, str]:
    """Create a session, send one prompt, read the session back, and judge it."""
    user = f"scenario-{uuid.uuid4().hex[:6]}"
    status, body = post(
        f"{base_url}/apps/{APP}/users/{user}/sessions", {"app_name": APP}, timeout=30
    )
    if status != 200:
        return False, f"could not create a session (HTTP {status})", ""
    session_id = json.loads(body)["id"]

    prompt = scenario.render(base_url)
    status, body = post(
        f"{base_url}/run_sse",
        {
            "app_name": APP,
            "user_id": user,
            "session_id": session_id,
            "new_message": {"role": "user", "parts": [{"text": prompt}]},
        },
        timeout=TURN_TIMEOUT_S,
    )

    # The turn is over when the session says so, not when the socket closes: an
    # aborted stream returns 200 with a partial event list, and the error event --
    # the thing worth catching -- is only ever persisted.
    turn = Turn(read_events(session_id), prompt)
    for _ in range(20):
        if turn.error or turn.answer.strip():
            break
        time.sleep(1.5)
        turn = Turn(read_events(session_id), prompt)

    ok, detail = scenario.check(turn)

    # Universal, like `_no_cache_hit` and for the same reason: every other check
    # here can pass on a turn that told the user nothing, because a name stamp and a
    # Sources block are text, and "there is an answer" is satisfied by text that
    # carries no answer. Applied here rather than composed into each scenario so a
    # new scenario cannot opt out of it by forgetting.
    body_ok, body_detail = _final_answer_has_a_body(turn)
    if not body_ok:
        return False, f"{detail} | BUT {body_detail}", session_id
    return ok, detail, session_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--only", default="", help="run scenarios whose name starts with this")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--keep", action="store_true", help="do not delete the scenario sessions")
    args = parser.parse_args()

    chosen = [s for s in SCENARIOS if s.name.startswith(args.only)]

    if args.list:
        for scenario in chosen:
            tags = ",".join(scenario.tags) or "-"
            print(f"{scenario.name:46} [{tags:12}] {scenario.note}")
        return 0

    if not chosen:
        print(f"no scenario matches --only={args.only!r}", file=sys.stderr)
        return 2

    print(f"base url: {args.base_url}    scenarios: {len(chosen)}\n")
    results = []
    for scenario in chosen:
        print(f"... {scenario.name}", flush=True)
        started = time.monotonic()
        try:
            ok, detail, session_id = run_scenario(scenario, args.base_url)
        except Exception as exc:  # noqa: BLE001
            ok, detail, session_id = False, f"harness error: {type(exc).__name__}: {exc}", ""
        took = time.monotonic() - started
        if session_id and not args.keep:
            delete_session(session_id)
        results.append((scenario, ok, detail, took))
        print(f"    {'PASS' if ok else 'FAIL'}  {took:5.1f}s  {detail}\n", flush=True)

    failed = [r for r in results if not r[1]]
    print("=" * 78)
    for scenario, ok, _, took in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {scenario.name:46} {took:5.1f}s")
    print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
