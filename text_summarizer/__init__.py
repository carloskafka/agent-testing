"""The agent package: load the environment, instrument, and export the entry points.

Two entry points are exported, and the difference between them is load-bearing.

``root_agent``
    The ``LlmAgent``. This is what ``adk eval``, ``adk run`` and every test in
    this package use, and it stays exactly what it was.

``app``
    The same agent wrapped in an ADK ``App`` with ``ResumabilityConfig(is_resumable=True)``.
    ``adk web`` prefers it -- ``cli/utils/agent_loader.py`` looks for ``app``
    before ``root_agent`` -- and nothing else does.

Why the wrapper exists
----------------------
``ask_user`` asks the user a question the agent cannot answer for itself, and it
does that by requesting a tool confirmation. **Emitting that confirmation is not
the same as pausing.** ADK builds the ``adk_request_confirmation`` event, marks
it long-running, and then... keeps going: in
``flows/llm_flows/base_llm_flow.py`` the line above the confirmation event sets
``invocation_context.end_invocation = True`` for an *auth* event and there is no
such line for a tool confirmation. So with no wrapper here the agent asks the
question and answers it itself on the very next step, and the user is handed a
form for a decision that has already been made.

There is a pause, but it is behind a flag. ``decide_step_resume``
(``flows/llm_flows/_resume_utils.py``) opens with::

    if not invocation_context.is_resumable:
      return ResumeDecision(ResumeAction.CONTINUE)

and ``is_resumable`` is true only when the app carries a resumability config --
which ``adk web`` does not build for you. Measured on a bare ``InMemoryRunner``,
no vault and no MCP, with one stub model that asks and then carries on:

    resumability_config=None      -> 4 events, answered, 2 model calls
    is_resumable=True             -> 3 events, silent,   1 model call

Both measured against google-adk 2.9.2. The session that prompted this is
``f0842db4-9d42-4911-8b34-255a3731021f``: the ``adk_request_confirmation`` event
was emitted at ``t=536.986`` and the next model call was made at ``t=536.997``,
eleven milliseconds later, and the answer was on screen 42 seconds before the
user could have answered anything.

**What this fixes and what it does not.** It makes the turn genuinely stop, which
is the precondition for everything else about ``ask_user`` being observable. It
does not give the user buttons: the bundled dev UI renders a confirmation as a
checkbox and a raw JSON payload editor, with no code path that reads
``payload.options``. That is patched separately
(``patch-adk-devui-confirm.py``), because it is a different failure in a
different layer, and conflating the two would make the first look unfixed when
only the second was.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from .observability import setup_observability

setup_observability()

from .agent import root_agent

#: The resumable App, or ``None`` on an ADK too old to have one.
app = None

try:
    from google.adk.apps.app import App, ResumabilityConfig

    app = App(
        name="text_summarizer",
        root_agent=root_agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
except ImportError:  # pragma: no cover - App is new in ADK 2.9
    # An older google-adk has no App. That degrades to the pre-App behaviour --
    # confirmation emitted, turn not paused -- rather than to an import error,
    # so the package still loads and the agent still answers. ask_user keeps
    # working through its default; it just cannot wait for an answer.
    pass

__all__ = ["app", "root_agent"]