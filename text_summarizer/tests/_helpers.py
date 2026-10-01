"""Fixtures shared by more than one test module, in one place.

Every helper here used to be copy-pasted. That is not a style complaint: these are
the objects the tests assert *on*, so two hand-rolled copies is two chances for one
module's version to drift from the other's and for a test to pass against a shape
production never produces. Two real defects came out of exactly that kind of drift
and are described in ``AGENTS.md`` (gotcha 13, and the duplicated-answer bug).

Not collected as a test module: the name does not match ``test_*.py``. Imported as
``from _helpers import ...`` because ``text_summarizer/tests/`` has no
``__init__.py``, so pytest prepends this directory to ``sys.path``.

What is deliberately *not* here: ``test_run_script.py`` keeps its own
``_make_vault`` (its signature takes a single path and derives the name from it,
and that module is owned by another workstream), and ``conftest.py`` stays as it is.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from text_summarizer import vaults
from text_summarizer.sources import MODEL_TOKEN, VAULT_TOKEN

#: The model name the stubs report. It appears in rendered source lines, so it is a
#: constant rather than a literal repeated in every assertion.
SERVED_MODEL = "gemini-3.5-flash-lite"

#: A canned answer carrying both sentinels, so a test can tell a *substituted*
#: response from the raw one the stub produced. The trailing line matters too: it
#: proves the substitution happens in place and does not truncate the answer.
MODEL_ANSWER = (
    "- Dogs are domesticated mammals valued for loyalty and companionship.\n"
    "- Dogs come in many breeds with varying size, color, and temperament.\n"
    "- Dogs are social animals that thrive on human and canine interaction.\n"
    "\n"
    f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: same topic\n"
    "\n"
    'Saved to the second brain as "Dogs Summary".'
)


# --- ADK stand-ins -------------------------------------------------------------


def _event(text: str = "", *, author: str = "text_summarizer", model_version=None, parts=None):
    """A minimal stand-in for an ADK ``Event`` (which subclasses ``LlmResponse``).

    One shape, three former callers. Empty ``text`` yields a part with ``text=None``
    rather than ``""`` because that is what a real tool-response part looks like, and
    ``parts=`` exists so a test can hand over a ``function_response`` part directly
    (the ``vault_info`` shape).
    """
    if parts is None:
        parts = [SimpleNamespace(text=text or None, function_response=None)]
    return SimpleNamespace(
        author=author,
        content=SimpleNamespace(parts=parts),
        model_version=model_version,
        partial=False,
    )


class _StubLlm(BaseLlm):
    """A ``BaseLlm`` that returns one canned answer and reports a ``model_version``.

    ``model_version`` is the load-bearing part: it is where the real backend's
    identity comes from, and the whole ``**Sources**`` provenance feature reads it
    from here.
    """

    answer: str = MODEL_ANSWER
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.answer)]),
            finish_reason=types.FinishReason.STOP,
            model_version=self.model,
        )


class _Client:
    """A stand-in for the Langfuse client that records what was scored.

    ``create_score`` keeps the real keyword-only signature on purpose: if the
    scoring code ever calls it positionally or with a new keyword, the stub raises
    instead of silently swallowing it, which is the whole point of the tests that
    assert on the recorded values.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.scores: dict[str, Any] = {}

    def create_score(self, *, trace_id, name, value, data_type) -> None:
        self.calls.append(
            {"trace_id": trace_id, "name": name, "value": value, "data_type": data_type}
        )
        self.scores[name] = value

    @property
    def names(self) -> list[str]:
        return list(self.scores)


# --- event text extraction -----------------------------------------------------


def _agent_texts(events) -> list[str]:
    """Every assistant-authored text event, in order."""
    return [
        "".join(p.text for p in (e.content.parts or []) if getattr(p, "text", None))
        for e in events
        if e.author == "text_summarizer" and e.content and e.content.parts
    ]


def _final_model_text(events) -> str:
    """The **last** assistant-authored text on the session, or ``""``.

    This is a known blind spot, not a convenience, and it is why
    ``test_the_answer_is_one_event_with_the_sources_substituted`` asserts on
    ``len(_agent_texts(events)) == 1`` rather than on the text this returns.

    "The last one" is what a Langfuse trace reports as the agent's output and what
    the old implementation's tests asserted on -- and when the answer was being
    emitted *twice* (``AGENTS.md`` gotcha 16), the last event carried the correct,
    substituted text. So a duplication bug produced a green test, a correct-looking
    trace, and a dev UI drawing the answer twice. A helper that hands you the final
    text keeps inviting that mistake; assert on the count.
    """
    texts = _agent_texts(events)
    return texts[-1] if texts else ""


def _all_text(events) -> str:
    """Every piece of text on every event, tool payloads included.

    Used to prove a sentinel string from a stub never reached the output, which only
    works if the search spans tool results too.
    """
    return "".join(
        p.text
        for e in events
        for p in ((e.content.parts if e.content else None) or [])
        if getattr(p, "text", None)
    )


# --- the vault -----------------------------------------------------------------


def _make_vault(
    parent, name: str = "ck", *, second_brain: bool = True, obsidian: bool = False
) -> str:
    """Create a vault directory under ``parent`` and return its path as a string.

    The two flags are the two things that make a directory a vault to different
    readers: ``Second Brain`` is this project's marker (``vaults.BRAIN_DIR``) and
    ``.obsidian`` is Obsidian's own (``vaults.OBSIDIAN_DIR``). A test that is
    exercising *discovery* wants the marker absent (``second_brain=False``); one that
    wants a real-looking vault wants both.
    """
    path = pathlib.Path(parent) / name
    path.mkdir(parents=True, exist_ok=True)
    if second_brain:
        (path / vaults.BRAIN_DIR).mkdir(exist_ok=True)
    if obsidian:
        (path / vaults.OBSIDIAN_DIR).mkdir(exist_ok=True)
    return str(path)


def _point_vault_at(
    monkeypatch,
    root,
    name: str = "ck",
    *,
    env_root=None,
    modules: tuple[str, ...] = ("second_brain", "agent"),
) -> str:
    """Repoint the vault at a temp dir and return the resolved path as a string.

    Three things have to move together, and forgetting one of them is how a test
    ends up asserting against a vault that is not the one it created:

    * ``second_brain.VAULT_ROOT`` and ``agent.VAULT_ROOT`` are module globals
      resolved at import time, so setting the environment variable alone does
      nothing to them -- and they are the two *separate* call sites that write
      ``generated_in_vault``, which is exactly how a note said ``ck`` while the
      answer said ``vaults``;
    * ``SECOND_BRAIN_VAULT`` is what a later re-resolution would read;
    * ``VAULT_NAME`` is deleted, because it outranks the path in
      ``sources.resolve_vault_name``. Tests that want it set call
      ``monkeypatch.setenv`` afterwards, which wins.

    ``env_root`` is for the shape Docker actually runs: ``SECOND_BRAIN_VAULT`` is
    the bind-mounted *parent* and the active vault is its single child, so the two
    arguments differ there and agree everywhere else. ``modules`` narrows which
    globals are repointed for tests that only exercise one call site, and ``name=""
    `` means "``root`` already *is* the vault".

    The directory is created, because ``resolve_vault_name`` requires the path to
    exist and a test that only repointed the globals would then resolve to
    ``unknown`` -- passing for the wrong reason.
    """
    resolved = pathlib.Path(root)
    if name:
        resolved = resolved / name
    resolved.mkdir(parents=True, exist_ok=True)

    import text_summarizer.agent as agent_module
    import text_summarizer.second_brain as second_brain_module

    for module in modules:
        target = second_brain_module if module == "second_brain" else agent_module
        monkeypatch.setattr(target, "VAULT_ROOT", str(resolved))

    monkeypatch.setenv("SECOND_BRAIN_VAULT", str(env_root if env_root is not None else resolved))
    monkeypatch.delenv("VAULT_NAME", raising=False)
    return str(resolved)


# --- resolve-vault.sh ----------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESOLVER = str(REPO_ROOT / "resolve-vault.sh")

#: The port ``resolve-vault.sh`` uses when nothing overrides it. Asserted as a literal
#: in the tests rather than imported from the shell script, because a test that reads
#: its expectation out of the thing under test asserts nothing.
DEFAULT_MCP_PORT = "37842"


def _run_resolver(parent: str, tmp_path, *, vault_name: str | None = None, port: str | None = None):
    """Run ``resolve-vault.sh`` against a stub ``obsidian-mcp``.

    Returns ``(returncode, stdout + stderr)``. The real binary is never needed, so
    this covers the shell logic rather than assuming it matches ``vaults.py`` --
    the two must never disagree about which vault is in use.
    """
    stub_dir = pathlib.Path(tmp_path) / "stubbin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "obsidian-mcp"
    stub.write_text('#!/bin/sh\necho "SERVE: $*"\n')
    stub.chmod(0o755)

    env = dict(os.environ, PATH=f"{stub_dir}:{os.environ['PATH']}", VAULT_PARENT=str(parent))
    # Both names the script consults, so the suite's pinned-but-empty values cannot
    # leak in and select a vault the test did not ask for.
    env.pop("VAULT_NAME", None)
    env.pop("OBSIDIAN_VAULT_NAME", None)
    if vault_name is not None:
        env["VAULT_NAME"] = vault_name
    if port is not None:
        env["OBSIDIAN_MCP_PORT"] = str(port)

    result = subprocess.run(["sh", RESOLVER], env=env, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout + result.stderr
