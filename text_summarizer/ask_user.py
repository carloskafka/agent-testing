"""Ask the user a question the agent genuinely cannot answer for itself.

**Why this exists.** Asked *"summarize the movies in Osasco with session times"*
against a site with two cinemas, the agent had a fork it could not resolve: every
session time it could report belonged to one cinema or the other, and the choice
changes every fact in the answer. It picked one and did not say so. The general
failure is worse than this example -- when a question has a real fork, guessing
produces a confident, well-cited, entirely wrong answer, and nothing in the
output distinguishes it from a considered choice.

So this is a **tool**, not an instruction. The alternative -- a rule telling the
model to ask in prose -- costs a full extra model call, renders the question as
chat text with no way to answer it structurally, and leaves the resumed turn
dependent on the model correctly reading a free-text reply. Here the pause is an
ADK primitive: the turn genuinely stops, and the answer arrives as typed data on
the tool call it interrupted.

**The bar this deliberately sets high: ask only when it changes the answer.**
An agent that asks about everything is worse than one that never asks, because
every question costs a turn and trains the user to reflexively type "yes". The
gate is in :func:`worth_asking`, and it is a function rather than a docstring
because a model cannot be relied on to apply a judgement rule to its own
reasoning -- it can be relied on to call a tool whose name says what it is for.

Two things this must not become:

* **Not a menu for everything.** ``worth_asking`` returns false whenever a
  default is obvious, so the common case never reaches the model at all.
* **Not a stall.** If the tool is called and the user never answers, the agent
  has to finish *somehow*: :func:`ask_user` falls back to the stated default and
  says which it used, because a turn that never returns is worse than a turn that
  guesses. That is also why the default is a required argument -- a fallback that
  had to invent one would be guessing with extra steps.
"""

from __future__ import annotations

from typing import Any

#: How many options are worth showing. Past this the question has stopped being a
#: decision and become a form, and a long list in a chat bubble is worse than a
#: sensible default. Five cinemas or five payment methods is a real fork; twenty
#: is a search box.
MAX_OPTIONS = 5

#: Longest option string kept. An option is a label a human reads in a chat
#: bubble, not a paragraph; anything longer is a sign the agent is asking the
#: wrong question.
MAX_OPTION_CHARS = 80


def worth_asking(
    *,
    options: list[str],
    default: str,
    consequence: str,
) -> tuple[bool, str]:
    """Whether this question is worth a turn, and why not when it is not.

    Returns ``(allowed, reason)``. ``reason`` is returned either way so the caller
    can report it: a skipped question that says *why* it was skipped is auditable,
    and one that silently proceeds is indistinguishable from never having
    considered it.

    The three refusals, each for a different reason:

    * **No real fork** -- fewer than two distinct options, or the default is
      already among them and nothing else is implied. There is no decision here.
    * **Not consequential** -- ``consequence`` is empty. A question whose answer
      changes nothing is a question about the agent's confidence, not the user's
      preference, and the agent should answer it itself.
    * **Too many options** -- past :data:`MAX_OPTIONS`. See its docstring.
    """
    distinct = [o.strip() for o in options if o and o.strip()]
    # Case-insensitive, because "Cinemark" and "cinemark" are one cinema and a
    # question that offers both reads as a bug in the agent.
    unique = list(dict.fromkeys(o.lower() for o in distinct))

    if len(unique) < 2:
        return False, "only one option, so there is no decision to make"
    if not consequence.strip():
        return False, "the answer would not change anything, so guessing is safe"
    if len(unique) > MAX_OPTIONS:
        return (
            False,
            f"{len(unique)} options is too many to choose between in a chat "
            f"message; pick the most likely and say so",
        )
    if default.strip().lower() not in unique:
        return (
            False,
            f"the default {default.strip()!r} is not one of the options, so "
            "there is nothing to fall back to",
        )
    return True, "the answer changes what gets reported"


def _clean(options: Any, default: Any, consequence: Any, question: Any) -> dict:
    """Normalise the model's arguments, keeping the prompt the user will read.

    Errors are returned as **data** (``web_search._error``'s discipline): the
    model has to be able to read a failure and correct it, rather than have the
    turn die on a malformed call.
    """
    if not isinstance(options, (list, tuple)):
        return {"error": "options must be a list of strings"}
    if not isinstance(default, str) or not isinstance(consequence, str):
        return {"error": "default and consequence must both be strings"}
    if not isinstance(question, str):
        return {"error": "question must be a string"}

    cleaned = [str(o).strip()[:MAX_OPTION_CHARS] for o in options if str(o).strip()]
    # Deduplicate case-insensitively but keep the model's own capitalisation: the
    # option text is what the user reads back.
    seen: set[str] = set()
    unique = []
    for option in cleaned:
        if option.lower() in seen:
            continue
        seen.add(option.lower())
        unique.append(option)

    allowed, reason = worth_asking(
        options=unique, default=default, consequence=consequence
    )
    return {
        "question": question.strip(),
        "options": unique,
        "default": default.strip(),
        "consequence": consequence.strip(),
        "allowed": allowed,
        "reason": reason,
    }


def ask_user(
    question: str,
    options: list[str],
    default: str,
    consequence: str,
    tool_context: Any = None,
) -> dict:
    """Ask the user to choose between options that change the answer.

    Use this ONLY when you have found a real fork: two or more options you
    cannot choose between yourself, where picking wrong makes the whole answer
    wrong. Say in `consequence` what differs between them -- that is what the
    user needs in order to decide, and it is what you report if they decline.

    Do NOT use it for anything with an obvious default, to check whether the
    user is still there, or to confirm something you are already confident of.
    If you find yourself about to call this tool just to be safe, answer instead.

    Args:
        question: What to ask, in one sentence. Shown to the user verbatim.
        options: The choices, in the order they should be offered.
        default: What to use if the user does not answer. Must be one of
            `options` -- this tool always finishes, and it finishes with this.
        consequence: What actually changes between the options. This is the part
            that makes the question worth asking; if you cannot fill it in, do
            not ask.

    Returns:
        The user's choice, or the default with `answered_by` saying which.
    """
    prepared = _clean(options, default, consequence, question)

    if "error" in prepared:
        return prepared

    if not prepared["allowed"]:
        # Refusing here rather than in the prompt: a model told "ask when it
        # matters" will decide it matters. The gate is code, so it holds.
        return {
            "status": "not_asked",
            "reason": prepared["reason"],
            "used": prepared["default"],
            "options": prepared["options"],
        }

    confirmation = getattr(tool_context, "tool_confirmation", None)

    if confirmation is None:
        if tool_context is None:
            # No framework to pause on (a direct call, or a unit test). Answering
            # with the default is the honest outcome: the alternative is to return
            # a question the caller cannot act on.
            return {
                "status": "answered",
                "answered_by": "default_no_turn_context",
                "choice": prepared["default"],
                "question": prepared["question"],
                "options": prepared["options"],
            }
        # Ask. ADK turns this into an `adk_request_confirmation` event carrying
        # `originalFunctionCall`, which is the one shape the bundled dev UI knows
        # how to render -- verified by running it, not by reading the docs.
        tool_context.request_confirmation(
            hint=prepared["question"],
            # The payload is what comes back as `ToolConfirmation.payload`, so it
            # has to be the options rather than the question.
            payload={"options": prepared["options"], "default": prepared["default"]},
        )
        return {
            "status": "asked",
            "question": prepared["question"],
            "options": prepared["options"],
            "default": prepared["default"],
            "consequence": prepared["consequence"],
        }

    # Resumed: the user answered. `payload` is whatever the UI sent, so it is
    # treated as untrusted shape and matched against the options we offered --
    # a client that echoes something else gets the default, not an invented value.
    payload = confirmation.payload
    if isinstance(payload, dict):
        raw_choice = payload.get("choice") or payload.get("selection")
        if raw_choice is None:
            # The bundled UI sends the whole options list unless the user types
            # one, so take the first offered option in that shape.
            offered = payload.get("options")
            if isinstance(offered, list) and offered:
                raw_choice = offered[0]
    elif isinstance(payload, str):
        raw_choice = payload
    else:
        raw_choice = None

    choice = str(raw_choice).strip() if raw_choice is not None else ""
    matched = next(
        (o for o in prepared["options"] if o.lower() == choice.lower()),
        "",
    )

    if confirmation.confirmed is False:
        # Declined outright: the default, and say so, rather than pressing on.
        return {
            "status": "answered",
            "answered_by": "default_declined",
            "choice": prepared["default"],
            "question": prepared["question"],
            "consequence": prepared["consequence"],
        }

    if not matched:
        # An answer we did not offer. The default is the only safe reading: it is
        # a value the model chose knowingly and stated up front.
        return {
            "status": "answered",
            "answered_by": "default_unrecognised_answer",
            "choice": prepared["default"],
            "unrecognised": choice[:MAX_OPTION_CHARS],
            "question": prepared["question"],
            "consequence": prepared["consequence"],
        }

    return {
        "status": "answered",
        "answered_by": "user",
        "choice": matched,
        "question": prepared["question"],
        "consequence": prepared["consequence"],
    }


def build_ask_user_tool():
    """The tool, or ``[]`` when ADK is not importable.

    Same gating shape as every other toolset in this package: the agent is
    ``FunctionTool``-ed from the result unconditionally, so an unconfigured or
    unimportable environment yields no tools rather than an import error at
    agent construction.
    """
    try:
        from google.adk.tools import FunctionTool
    except ImportError:  # pragma: no cover - google-adk is a hard dependency
        return []
    return [FunctionTool(ask_user)]