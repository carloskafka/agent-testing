"""``docker-compose.yml`` must be a file Compose can actually read.

**Why this test exists.** PR #32 committed six lines of ``docker-compose.yml``
with a literal ``+`` at the start of each -- the diff markers of the patch that
was meant to remove them. The result does not parse:

    yaml.scanner.ScannerError: while scanning a simple key
      in "docker-compose.yml", line 69, column 1
    could not find expected ':'

Nothing caught it, and the reason is worth recording: **no gate in this repo
reads the compose file.** ``make lint`` is ruff, ``make types`` is mypy, ``make
test`` is pytest, and none of them parse YAML. The file is only ever read by
``docker compose`` -- so the corruption was invisible to CI and invisible to a
developer whose stack was already running. Our own containers kept working
throughout, because the working tree was never re-checked-out from ``main``; a
**fresh clone could not start at all**, which is the only situation that matters
to anyone but us.

So this is a gate on a file the gate previously did not cover, and the defect it
covers was one line of accidental formatting in an otherwise careful commit.

The assertion is deliberately about **structure, not formatting**: a stray ``+``
is a symptom, and what actually matters is that the file is loadable and that the
environment keys a container depends on are still declared. A test asserting
"there are no lines starting with +" would pass on a file that is valid YAML
with the renderer block deleted -- which is exactly the regression worth
preventing, because an undeclared ``RENDERER_URL`` fails *silently* (the tier
simply never renders) rather than loudly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

#: The repository root, from this file: ``text_summarizer/tests/<this>``.
REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "docker-compose.yml"

#: ``pyyaml`` is present transitively (many packages want it) but is NOT declared
#: in ``pyproject.toml``. Skipping keeps a future dependency prune from failing
#: the suite for a reason that has nothing to do with this file.
yaml = pytest.importorskip("yaml", reason="pyyaml is not a declared dependency")


def test_the_compose_file_is_at_the_expected_path():
    """Guards the test itself: a moved file would make every test here vacuous."""
    assert COMPOSE.is_file(), f"{COMPOSE} is missing; this test would pass on nothing"


def test_the_compose_file_parses():
    """The defect itself. A ``+`` at column 1 is a YAML scanner error."""
    with COMPOSE.open(encoding="utf-8") as handle:
        compose = yaml.safe_load(handle)
    assert isinstance(compose, dict), "docker-compose.yml did not load as a mapping"


def test_no_line_carries_a_stray_diff_marker():
    """The specific corruption, named so a recurrence is obvious in review.

    Asserted separately from parsing because the *mechanism* is worth recording:
    these lines are the ``+`` side of a patch left in the file, so the check is
    about provenance rather than YAML in general.
    """
    offenders = [
        (number, line)
        for number, line in enumerate(
            COMPOSE.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.startswith("+")
    ]
    assert not offenders, (
        "docker-compose.yml has lines beginning with '+' -- a patch's diff "
        f"markers committed as content: {offenders[:3]}"
    )


def test_every_service_declares_the_keys_the_stack_needs():
    """The silent half: a missing key does not fail a parse, it disables a tier.

    ``SECOND_BRAIN_VAULT`` points at the vault parent and ``RENDERER_URL`` names
    the renderer. Dropping either yields a compose file that loads perfectly and
    an agent that quietly stops rendering JS pages -- the same shape as the SSRF
    host that did not resolve, and the reason this asserts presence by name
    rather than only asserting the file parses.
    """
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = compose.get("services") or {}
    assert "agent-testing" in services, "the agent service is missing"
    environment = (services["agent-testing"].get("environment") or {})

    assert "SECOND_BRAIN_VAULT" in environment, (
        "SECOND_BRAIN_VAULT is unset: the vault parent is what keeps the real "
        "vault name in the path, so every **Sources** line would read 'vaults'"
    )
    assert "RENDERER_URL" in environment, (
        "RENDERER_URL is unset: web_fetch would never fall back to a browser, "
        "and a JS-rendered page would come back as a title and a nav bar"
    )


@pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker is not installed on this machine"
)
def test_docker_compose_can_read_the_file(tmp_path):
    """The real reader, when it is available.

    ``yaml.safe_load`` is a parser; Compose is the *consumer*, and it has its own
    opinion (variable interpolation, the ``name:`` on an external network, the
    ``network_mode: service:`` a sidecar depends on). This runs the actual
    command against a copy in ``tmp_path`` so it cannot touch the running stack
    -- ``docker compose config`` only renders, never applies.
    """
    workdir = tmp_path / "compose-check"
    workdir.mkdir()
    shutil.copy(COMPOSE, workdir / COMPOSE.name)
    # Compose requires `env_file: .env` to *exist*, even though every value in it
    # is optional and has a `${VAR:-default}` fallback. Without this the check
    # fails for a reason that has nothing to do with the YAML, which is exactly
    # the kind of test that gets deleted instead of fixed.
    (workdir / ".env").write_text("", encoding="utf-8")

    result = subprocess.run(
        ["docker", "compose", "config"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "docker compose config rejected docker-compose.yml:\n"
        f"{result.stderr.strip()[:600]}"
    )