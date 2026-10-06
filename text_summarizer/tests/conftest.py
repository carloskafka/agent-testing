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
#: The OpenCode Zen tier. Pinned for the same reason as OPENROUTER_API_KEY above,
#: and it is not optional bookkeeping: with a developer's real key exported, the
#: chain gains a `hosted_vllm/space-bunny-free` entry and every test that asserts on
#: the chain's shape fails -- which is how this was found, by a test that had
#: nothing to do with model selection.
os.environ["OPENCODE_API_KEY"] = ""
os.environ["OPENCODE_API_BASE"] = ""

#: Both Obsidian transports, so the MCP tool list is identical everywhere.
#: Gotcha 11: with neither set, ``build_obsidian_tools()`` returns ``[]``.
os.environ["OBSIDIAN_VAULT_PATH"] = ""
os.environ["OBSIDIAN_MCP_URL"] = ""

#: The headless-browser renderer. Pinned like CACHE_ENABLED rather than defaulted,
#: because a developer's exported value would otherwise decide whether the web tier
#: falls back to a browser, and the tests that cover that path would pass or fail
#: depending on the machine. Empty here; the render tests set their own.
RENDERER_URL = os.environ.pop("RENDERER_URL", None)
os.environ["RENDERER_URL"] = ""


#: The weather tier, pinned for the same reason as ``RENDERER_URL`` above: it is a
#: tier whose *tool list* is built at import time, so a developer's ``.env`` would
#: decide whether ``weather_forecast`` is even in ``root_agent.tools``. ``WEATHER_ENABLED``
#: is pinned **on** rather than blanked, because on is the shipped default and blanking
#: it would silently stop the tests from covering the wired path.
os.environ["WEATHER_ENABLED"] = "true"
os.environ["WEATHER_GEOCODING_URL"] = "https://geocoding-api.open-meteo.com/v1/search"
os.environ["WEATHER_FORECAST_URL"] = "https://api.open-meteo.com/v1/forecast"
os.environ["WEATHER_PROVIDER"] = "open-meteo"
os.environ["WEATHER_LANGUAGE"] = "pt"

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

#: **3. The chart store.** ``chart.chart_dir()`` defaults to
#: ``$CWD/.adk/charts``, and the suite renders real charts -- ``test_sources.py`` and
#: ``test_weather.py`` both drive ``render_chart`` end to end and then assert on the
#: bytes that were written. Left alone, every test run dropped SVG files into the
#: repository's own ``.adk`` directory, which is git-ignored but is also where a real
#: deployment's charts live: a developer's next ``docker compose up`` would then be
#: serving images written by a test run, from a payload that was never requested.
#:
#: Set per-session rather than per-test because ``chart_dir`` resolves the environment
#: on every call, and a ``tmp_path`` here would not be importable from a test module.
#: The directory is created eagerly so the mount path and the write path agree.
_CHART_DIR = tempfile.mkdtemp(prefix="adk-tests-charts-")
os.makedirs(_CHART_DIR, exist_ok=True)
os.environ["CHART_DIR"] = _CHART_DIR
