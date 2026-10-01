"""Tests for the digest *tool*: the vault digest as something the model can call.

``test_digest.py`` covers the reader. This module covers the four things the wrapper
adds on top of it, each of which is a way the tool could quietly *lie* rather than fail:

* **it is actually offered.** A tool that is written, gated correctly and then not added
  to ``root_agent.tools`` is invisible, and nothing else in the suite would notice.
* **a date it cannot parse is an error the model can read**, not an empty digest. An
  empty digest is a legitimate answer, which is exactly why it must never be produced by
  accident: ``"last tuesday"`` matching no filename yields a clean zero-note result that
  reads as "you learned nothing that day".
* **the cap is reported.** ``note_count`` stays the true total and ``truncated`` says how
  much was withheld, so a caller that gets 25 of 40 notes can still say 40.
* **a vault that cannot be read is data, not an exception**, because the turn has to
  survive it.

The date-validation test compares the tool against ``--date`` on the same spellings
rather than restating what a valid date is. That is the point of the wrapper reusing
``digest.iso_date``: two definitions of one format is how the tool and the command come
to disagree, and a disagreement is silent in both directions.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from datetime import date

import pytest
from _helpers import _point_vault_at
from text_summarizer import digest, digest_tools, second_brain

DAY = "2026-09-30"
FP = "a" * 64


def _plant(vault, stem: str, *, day: str = DAY, model: str | None = "gemini-3.5-flash-lite") -> str:
    """One note in the writer's own on-disk format, with provenance switchable.

    ``model=None`` plants a pre-provenance note -- the shape a note written before
    ``generated_by_model`` existed has, which is the one case where the tool must report
    an absence rather than a value.
    """
    brain = pathlib.Path(vault) / second_brain.BRAIN_DIR
    brain.mkdir(parents=True, exist_ok=True)
    frontmatter = ["---", "tags:", "  - Dogs", f"date: {day}", f"source_fingerprint: {FP}"]
    if model:
        frontmatter.append(f"{second_brain.GENERATED_BY_MODEL_KEY}: {model}")
        frontmatter.append(f"{second_brain.GENERATED_IN_VAULT_KEY}: ck")
    frontmatter.append("---")
    path = brain / f"{stem}.md"
    path.write_text("\n".join(frontmatter) + f"\n\n- {stem} learned something.\n", encoding="utf-8")
    return str(path)


def _plant_chat(vault, day: str, count: int) -> str:
    folder = pathlib.Path(vault) / second_brain.CHAT_LOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"## 10:{index:02d}\n\n**User:**\n\nquestion {index}\n\n**Agent:**\n\nanswer {index}\n"
        for index in range(count)
    )
    path = folder / f"{day}.md"
    path.write_text(f"# Chat Log -- {day}\n\n{body}", encoding="utf-8")
    return str(path)


@pytest.fixture
def vault(monkeypatch, tmp_path):
    """A vault named ``ck``. Only ``second_brain``'s binding is repointed.

    ``digest.default_vault_root`` reads that module attribute at call time, so this is
    the only binding there is to move -- and ``digest_tools`` inherits it, which is the
    same single-source property ``test_digest.py`` relies on.
    """
    return pathlib.Path(_point_vault_at(monkeypatch, tmp_path, "ck", modules=("second_brain",)))


# --- is it offered at all --------------------------------------------------------


def test_the_tool_is_offered_when_there_is_a_vault(vault):
    assert digest_tools.digest_enabled() is True
    assert [tool.name for tool in digest_tools.build_digest_tools()] == ["read_day_digest"]


def test_it_is_withheld_when_there_is_no_vault_to_read(monkeypatch, tmp_path):
    """Gated, unlike the clock.

    The distinction is the whole reason: a missing clock is a *silently wrong* answer,
    where a missing vault is a missing capability. With no directory on disk this tool's
    only possible reply is "no notes", and offering the model a tool that can only ever
    say that is worse than not offering it.
    """
    absent = tmp_path / "not-created"
    monkeypatch.setattr(second_brain, "VAULT_ROOT", str(absent))
    assert digest_tools.digest_enabled() is False
    assert digest_tools.build_digest_tools() == []


def test_the_agent_carries_the_tool_when_a_vault_exists(vault):
    """The wiring itself: written is not the same as offered.

    A tool that is gated correctly and then left out of ``root_agent.tools`` is
    invisible to the model, and no other assertion in this file would catch it.
    """
    from text_summarizer import agent

    names = [getattr(tool, "name", None) for tool in agent.root_agent.tools]
    assert "read_day_digest" in names


def test_the_agent_does_not_carry_it_without_a_vault(monkeypatch, tmp_path):
    """The gate is evaluated at import, so the tool list must not be built once.

    ``root_agent.tools`` is a list comprehension evaluated when ``agent.py`` is imported.
    Repointing ``second_brain.VAULT_ROOT`` afterwards cannot retract a tool that is
    already in it -- which is fine and is not what this asserts. What this asserts is the
    gate itself: same module, same function, no vault, no tool.
    """
    monkeypatch.setattr(second_brain, "VAULT_ROOT", str(tmp_path / "absent"))
    assert digest_tools.build_digest_tools() == []


# --- reading a day ---------------------------------------------------------------


def test_a_day_comes_back_with_what_was_written_and_where_it_came_from(vault):
    _plant(vault, f"{DAY} - dogs")
    payload = digest_tools.read_day_digest(DAY)
    assert payload["date"] == DAY
    assert payload["note_count"] == 1
    note = payload["notes"][0]
    assert note["title"] == f"{DAY} - dogs"
    assert note["provenance"]["generated_by_model"] == "gemini-3.5-flash-lite"
    assert note["provenance"]["generated_in_vault"] == "ck"


def test_an_empty_day_is_an_answer_and_not_an_error(vault):
    payload = digest_tools.read_day_digest(DAY)
    assert payload["note_count"] == 0
    assert payload["notes"] == []
    assert "error" not in payload


def test_omitting_the_day_reports_today(vault):
    assert digest_tools.read_day_digest()["date"] == date.today().isoformat()


def test_provenance_that_was_never_recorded_is_not_invented(vault):
    """A pre-provenance note carries no model, and the tool says nothing about one.

    The tool's own docstring tells the model not to attribute such a note to whichever
    model is serving now. That instruction is only credible if the payload gives it
    nothing to work with -- a back-filled model name would make the instruction
    unauditable, and this is the assertion that keeps it honest.
    """
    _plant(vault, f"{DAY} - old", model=None)
    note = digest_tools.read_day_digest(DAY)["notes"][0]
    assert "generated_by_model" not in note["provenance"]


def test_the_sources_block_never_reaches_the_payload(vault):
    """A ``[obsidian][ck][…]`` marker means nothing outside a rendered answer."""
    _plant(vault, f"{DAY} - dogs")
    brain = pathlib.Path(vault) / second_brain.BRAIN_DIR
    with (brain / f"{DAY} - dogs.md").open("a", encoding="utf-8") as handle:
        handle.write("\n[obsidian][ck][gemini-3.5-flash-lite][[Other]]: same topic\n")
    rendered = json.dumps(digest_tools.read_day_digest(DAY))
    assert "[obsidian]" not in rendered


# --- the answer carries provenance ----------------------------------------------
#
# The payload deliberately carries no ``[obsidian]`` marker (above), so the Sources
# block on a digest turn can only come from the model emitting a sentinel line and
# ``sources.render_sources`` substituting it -- which means the model has to be
# *asked*. Rule 7 asks, and it is scoped to "Before summarizing", i.e. the
# ``search_text``/``note_read`` flow. Rule 14 covers the digest flow and, until
# this test was written, did not ask.


def test_the_digest_rule_asks_for_sources_like_every_other_vault_answer():
    """A report *about* the vault is the most provenance-worthy answer there is.

    Found live, not reasoned about: session ``2d756ad4-ef8b-4dec-a6e0-f07342a9957a``,
    "What did you learn on Tuesday?". The model resolved the date with the clock,
    called ``read_day_digest``, got 7 notes and reported them accurately -- and the
    answer carried no Sources block at all, while the name stamp *was* applied. That
    asymmetry is the signature: the stamp is unconditional in the same callback, so a
    half-rendered answer means the renderer had nothing to substitute, which means no
    sentinel line, which means nothing asked for one.

    Asserted on the rule text because that is the only place the ask can be made.
    A failing test here is the fix, not an obstacle to it.
    """
    from text_summarizer.agent import root_agent

    rule_14 = _instruction_rule(root_agent.instruction, 14)

    assert "rule 7" in rule_14, "the digest rule must defer to rule 7's template, not restate it"
    assert "at the END" in rule_14, "the ask has to say where the lines go, as rule 7 does"
    # Deferring to rule 7 only works if rule 7 actually carries the sentinel.
    assert "@@ADK_VAULT@@" in _instruction_rule(root_agent.instruction, 7)


def _instruction_rule(instruction: str, number: int) -> str:
    """One numbered rule out of the ``instruction`` string, up to the next number.

    Splitting on the ``N. `` prefix rather than asserting on a substring of the
    whole block, so a mention in some *other* rule cannot keep this passing. The
    numbering is what the model reads too, so it is the right granularity: the
    failure this guards is "the ask lives in the wrong rule".
    """
    import re

    # Anchored to the start of a line, not to any whitespace: rule 13 ends a
    # sentence with "…as rule 7." and the next sentence begins "If the vault…",
    # which a looser pattern reads as a second rule numbered 7. That was this
    # helper's own bug, and it silently pointed the assertion at the wrong text.
    starts = [
        (int(m.group(1)), m.start())
        for m in re.finditer(r"^\s*(\d+)\. [A-Z]", instruction, re.MULTILINE)
    ]
    body = {}
    for index, (n, begin) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(instruction)
        body[n] = instruction[begin:end]

    assert number in body, f"rule {number} not found; found {sorted(body)}"
    return body[number]


# --- dates -----------------------------------------------------------------------


def test_a_date_the_tool_cannot_parse_is_an_error_the_model_can_read(vault):
    """Never an empty digest.

    ``build_digest`` matches a day against filenames and returns zero notes for a string
    that matches nothing, so passing ``last tuesday`` through unchecked produces a
    perfectly clean "note_count: 0" that reads as *you learned nothing that day*. That is
    the one failure this whole module exists to make impossible.
    """
    payload = digest_tools.read_day_digest("last tuesday")
    assert "error" in payload
    assert "note_count" not in payload
    # The remedy has to be in the payload: the model cannot read the traceback.
    assert "current_datetime" in payload["hint"]


@pytest.mark.parametrize(
    "value",
    ["2026-09-30", "2026-1-1", "20260101", "2026-W40-4", "Sept 30", "  ", "yesterday"],
)
def test_the_tool_and_the_cli_agree_about_what_a_date_is(value):
    """One definition of a valid date, asserted structurally.

    ``digest_tools`` calls ``digest.iso_date`` rather than parsing again, and this is
    what keeps that true: the same spellings go through ``--date``'s argparse type and
    through the tool, and both must accept or both must reject. A second date parser
    would pass every other test here.

    ``""`` is excluded and has its own test below, because the two *must* disagree on it.
    """
    cli_ok = True
    try:
        digest.iso_date(value)
    except argparse.ArgumentTypeError:
        cli_ok = False

    tool_payload = digest_tools.read_day_digest(value)
    tool_ok = "error" not in tool_payload
    assert tool_ok is cli_ok, f"{value!r}: cli={cli_ok} tool={tool_ok}"


def test_an_omitted_day_means_today_and_an_explicit_empty_one_is_an_error(vault):
    """The one spelling where the two disagree on purpose, pinned so it stays deliberate.

    ``day=""`` is the tool's *default*, and a function has no way to say "no argument"
    other than not being given one -- so an empty string has to mean "today". The CLI has
    no such constraint: typing ``--date ""`` is a mistake, and argparse rejects it.

    That asymmetry is the reason ``""`` is not in the list above. It is also why the tool
    tests ``raw != ""`` rather than ``raw.strip() != ""``: treating blank as omitted
    would quietly answer a different question than the caller asked, which is the failure
    this whole validation exists to rule out.
    """
    assert "note_count" in digest_tools.read_day_digest("")

    with pytest.raises(argparse.ArgumentTypeError):
        digest.iso_date("")

    # And blank is a value, not an omission -- rejected by both, never silently today.
    assert "error" in digest_tools.read_day_digest("   ")


def test_the_tool_and_the_cli_agree_about_the_day_they_report(vault):
    """Not just accept/reject -- the same string has to mean the same day."""
    _plant(vault, f"{DAY} - dogs")
    tool_day = digest_tools.read_day_digest(DAY)["date"]
    cli_day = digest.build_digest(digest.iso_date(DAY)).date
    assert tool_day == cli_day


# --- bounds ----------------------------------------------------------------------


def test_a_long_day_is_capped_and_says_how_many_it_withheld(vault):
    """``note_count`` is the truth; ``notes`` is what fit.

    Reporting a capped list with the true total is the difference between "25 notes" and
    "25 of 40 notes". A cap applied silently makes a busy day look like a quiet one --
    the exact failure ``digest``'s docstring says it must never have.
    """
    total = digest_tools.MAX_NOTES + 5
    for index in range(total):
        _plant(vault, f"{DAY} - note {index:02d}")

    payload = digest_tools.read_day_digest(DAY)
    assert payload["note_count"] == total
    assert len(payload["notes"]) == digest_tools.MAX_NOTES
    assert payload["truncated"] == {"notes": 5}


def test_a_day_that_fits_reports_no_truncation(vault):
    """The other branch, deliberately: an absent ``truncated`` means nothing was cut."""
    _plant(vault, f"{DAY} - dogs")
    payload = digest_tools.read_day_digest(DAY)
    assert "truncated" not in payload


def test_chat_is_capped_as_well(vault):
    _plant_chat(vault, DAY, digest_tools.MAX_CHAT + 3)
    payload = digest_tools.read_day_digest(DAY, include_chat=True)
    assert len(payload["chat"]) == digest_tools.MAX_CHAT
    assert payload["truncated"] == {"chat": 3}


def test_chat_is_off_unless_asked_for(vault):
    _plant_chat(vault, DAY, 2)
    assert digest_tools.read_day_digest(DAY)["chat"] is None
    assert len(digest_tools.read_day_digest(DAY, include_chat=True)["chat"]) == 2


def test_the_fingerprint_is_not_paid_for_in_context(vault):
    """A script grouping a day by what was asked wants it; the model does not.

    ``to_dict`` carries 64 hex characters per note, which are pure context cost in a
    tool result. The CLI keeps them because ``--json`` is a script's input; the tool drops
    them. Both assertions are here because the divergence is deliberate and would
    otherwise look like one of them being a bug.
    """
    _plant(vault, f"{DAY} - dogs")
    result = digest.build_digest(DAY)
    assert digest.to_dict(result)["notes"][0]["fingerprint"] == FP
    assert "fingerprint" not in digest_tools.read_day_digest(DAY)["notes"][0]


# --- failures are data -----------------------------------------------------------


def test_a_vault_that_cannot_be_read_is_data_not_an_exception(vault, monkeypatch):
    """Same reasoning as ``web_search._error``.

    The model has to be able to read the failure and fall through to another tool. An
    exception here would kill the turn and say nothing about why.
    """

    def boom(*args, **kwargs):
        raise OSError("Input/output error")

    monkeypatch.setattr(digest, "build_digest", boom)
    payload = digest_tools.read_day_digest(DAY)
    assert "Input/output error" in payload["error"]


# --- the command's stdout stays a data stream ------------------------------------


def test_the_json_flag_lands_on_a_clean_stdout(tmp_path):
    """``--json`` is only usable if stdout carries nothing but the JSON.

    Importing the package runs ``setup_observability`` and resolves the vault, and both
    used to print -- *to stdout* -- before this module's body ever ran. So a caller doing
    ``digest --json | jq`` got diagnostics prepended to the document, and the only
    documented workaround was to skip the first lines by hand. This asserts the
    property rather than the fix: the subprocess is given Langfuse env vars bad enough
    to make it print, and stdout must still parse.

    ``LANGFUSE_PUBLIC_KEY`` is set rather than unset precisely because the honest case --
    Langfuse configured and working -- is one this suite cannot depend on, and an unset
    key prints nothing at all.
    """
    vault = tmp_path / "ck"
    _plant(vault, f"{DAY} - dogs")

    env = dict(
        os.environ,
        LANGFUSE_PUBLIC_KEY="not-a-real-key",
        LANGFUSE_SECRET_KEY="not-a-real-key",
        LANGFUSE_BASE_URL="http://127.0.0.1:1",
        LANGFUSE_AUTH_CHECK_TIMEOUT="1",
        SECOND_BRAIN_VAULT=str(vault),
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "text_summarizer.digest",
            "--vault", str(vault), "--date", DAY, "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(digest.__file__))),
    )
    assert result.returncode == 0, result.stderr
    # The point of the env above: it has to have produced *something* on stderr, or
    # this test would pass for a reason that has nothing to do with the streams.
    assert result.stderr.strip(), "expected Langfuse to complain on stderr"
    assert json.loads(result.stdout)["note_count"] == 1


def test_a_diagnostic_never_lands_on_stdout_in_the_package():
    """Every ``print`` in the library modules names its stream.

    Scoped to the modules that are *libraries* -- an entry point like ``serve.py`` or
    ``gmail_oauth.py`` legitimately writes its own output to stdout. A module added to
    this list needs no thought, which is the point; a module *left off* it is the gap,
    and the subprocess test above is what catches the one that actually matters.
    """
    import ast

    package = pathlib.Path(digest.__file__).parent
    libraries = (
        "agent.py",
        "clock.py",
        "digest_tools.py",
        "eval_scoring.py",
        "gmail_tools.py",
        "obsidian_tools.py",
        "observability.py",
        "second_brain.py",
        "sources.py",
        "vaults.py",
    )
    offenders = []
    for name in libraries:
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
                and not any(keyword.arg == "file" for keyword in node.keywords)
            ):
                offenders.append(f"{name}:{node.lineno}")
    assert not offenders, f"print() with no explicit stream: {offenders}"


def test_a_stray_import_does_not_break_the_package(tmp_path):
    """Cheap guard that the new module is importable the way the agent imports it.

    ``digest_tools`` is the first thing in the package that imports both ``digest`` and
    ``google.adk.tools``; a typo in either import is a hard failure at agent import
    time, and this asserts it in a subprocess rather than relying on collection order.
    """
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from text_summarizer.digest_tools import build_digest_tools, read_day_digest;"
            "print(callable(read_day_digest))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, SECOND_BRAIN_VAULT=tempfile.mkdtemp(prefix="digest-tools-")),
    )
    assert result.returncode == 0, result.stderr
    assert "True" in result.stdout