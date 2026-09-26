"""Test bootstrap: make the suite independent of the developer's own ``.env``.

Two separate problems, both of which made test results lie.

**1. A network call on import.** Importing anything under ``text_summarizer`` runs
the package ``__init__``, which calls ``setup_observability()``. With a real
``LANGFUSE_PUBLIC_KEY`` that performs a network auth check (tens of seconds
against an unreachable host), so the key is blanked first -- ``load_dotenv`` uses
``setdefault`` and will not put it back.

**2. Secrets leaking in from ``.env`` and changing behaviour.** ``load_dotenv()``
also populates everything else in ``.env``, and ``agent.py`` builds its tool
lists at *import* time::

    tools = [*build_obsidian_tools(), *build_gmail_tools()]

So on a machine where the developer has configured Gmail, ``build_gmail_tools()``
takes its happy path during collection; on CI, with no ``.env``, it takes the
``return []`` path instead. Same suite, different code exercised, and a coverage
number that moves with whatever secrets happen to be on disk. ``SECOND_BRAIN_VAULT``
leaks the same way and made every run print a spurious
``[vault] could not create /vault/agent-vault`` permission warning.

Everything is therefore pinned here, *before* the first package import. Tests that
genuinely need a variable still set it with ``monkeypatch.setenv``, which happens
later and wins for that test.

Deliberately **not** blanked: ``CACHE_ENABLED``. ``agent.py`` reads it at import
time, so it must be pinned rather than defaulted -- a developer's exported value
would otherwise change whether the cache callback is even wired up.
"""

import os
import tempfile

#: Credentials for the optional Gmail MCP server. Blanked so ``build_gmail_tools()``
#: takes its disabled path during collection on every machine.
os.environ["GOOGLE_CLIENT_ID"] = ""
os.environ["GOOGLE_CLIENT_SECRET"] = ""
os.environ["GOOGLE_REFRESH_TOKEN"] = ""

#: Model selection, so a developer's provider choice cannot change which code runs.
os.environ["MODEL_PROVIDER"] = "gemini"
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENROUTER_API_BASE"] = ""
os.environ["MODEL_ALIAS"] = ""

#: Both Obsidian transports, so the MCP tool list is identical everywhere.
#: Gotcha 11: with neither set, ``build_obsidian_tools()`` returns ``[]``.
os.environ["OBSIDIAN_VAULT_PATH"] = ""
os.environ["OBSIDIAN_MCP_URL"] = ""

#: Observability: no key means no instrumenting, and no network.
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_BASE_URL"] = ""

#: A throwaway vault parent, so importing ``second_brain`` resolves somewhere
#: writable and quiet instead of warning about a path it cannot create. Tests that
#: care about the vault point it at their own ``tmp_path`` via
#: ``_point_vault_at``; this only has to be harmless.
#:
#: The child is named ``ck`` and given a ``Second Brain`` directory on purpose.
#: ``sources.resolve_vault_name`` prefers ``basename(vault_root)`` over
#: ``VAULT_NAME``, so a random temp name would leak straight into the provenance
#: of every test that asserts on a vault name. ``ck`` keeps the rendered output
#: deterministic regardless of the machine or the tmpdir suffix.
_VAULT_PARENT = tempfile.mkdtemp(prefix="adk-tests-")
os.makedirs(os.path.join(_VAULT_PARENT, "ck", "Second Brain"), exist_ok=True)
os.environ["SECOND_BRAIN_VAULT"] = os.path.join(_VAULT_PARENT, "ck")
os.environ["OBSIDIAN_VAULT_NAME"] = ""
os.environ["VAULT_NAME"] = ""

#: Pinned, not defaulted: read at import time by ``agent.py``.
os.environ["CACHE_ENABLED"] = "true"
