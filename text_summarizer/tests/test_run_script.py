"""run.sh vault selection: the interactive half of the multi-vault fix.

``text_summarizer/vaults.py`` decides *which* vault; ``run.sh`` is what a person
actually runs, so it is the only place the choice can be made. These tests drive
the real script with a stubbed ``docker`` (so nothing is built or started) and a
temporary ``.env``/``HOME``, and assert both the message the user sees and the
values persisted for the next run.

The menu needs a TTY, so it is exercised through ``script(1)`` and skipped where
that is unavailable. Everything the menu leads to is also reachable non-interactively
via ``--vault``, which is covered unconditionally.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_SH = REPO_ROOT / "run.sh"

pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="run.sh needs bash")


def _make_vault(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "Second Brain").mkdir(exist_ok=True)
    (path / ".obsidian").mkdir(exist_ok=True)
    return path


def _sandbox(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A throwaway copy of the repo skeleton plus a stub docker on PATH.

    The script is copied rather than run in place because it writes to ``.env`` in
    its own directory, and the real repo's ``.env`` holds live API keys.
    """
    workdir = tmp_path / "repo"
    workdir.mkdir()
    shutil.copy(RUN_SH, workdir / "run.sh")
    shutil.copy(REPO_ROOT / "resolve-vault.sh", workdir / "resolve-vault.sh")
    shutil.copy(REPO_ROOT / ".env.example", workdir / ".env.example")
    (workdir / "docs").mkdir()
    (workdir / "text_summarizer").mkdir()
    shutil.copy(REPO_ROOT / "text_summarizer" / "vaults.py", workdir / "text_summarizer")

    stubbin = tmp_path / "bin"
    stubbin.mkdir()
    docker = stubbin / "docker"
    docker.write_text('#!/bin/sh\necho "STUB DOCKER: $*"\n')
    docker.chmod(0o755)

    return workdir, stubbin


def _sandbox_env(stubbin: pathlib.Path, home: pathlib.Path) -> dict[str, str]:
    """The environment every run.sh invocation gets: stub docker, a clean HOME.

    ``OBSIDIAN_VAULT*``/``VAULT_*`` are stripped because run.sh reads them from the
    process environment *before* .env, and they decide what the run does at all.
    ``conftest.py`` already blanks ``OBSIDIAN_VAULT_NAME`` and ``VAULT_NAME``, but
    not ``OBSIDIAN_VAULT_PARENT_HOST``: a developer who exports that gets a
    different parent, a different candidate list, and often no menu -- verified, the
    four interactive tests below all fail under
    ``env OBSIDIAN_VAULT_PARENT_HOST=/elsewhere``. Stripping the whole prefix here
    also covers the next variable someone adds.

    ``obsidian.json`` is a second leak of the same kind -- ``vaults.obsidian_vaults``
    reads the developer's real vaults off disk -- so it is pinned to ``{}`` unless a
    test wrote its own.
    """
    env = dict(os.environ)
    env["PATH"] = f"{stubbin}:{env['PATH']}"
    env["HOME"] = str(home)
    # Keep the real keys out of the sandbox .env; the script only needs these two.
    for key in list(env):
        if key.startswith(("OBSIDIAN_VAULT", "VAULT_")):
            env.pop(key)
    (home / ".config" / "obsidian").mkdir(parents=True, exist_ok=True)
    config = home / ".config" / "obsidian" / "obsidian.json"
    if not config.exists():
        # A test may have written a real one; never clobber it.
        config.write_text("{}")
    return env


def _run(
    workdir: pathlib.Path,
    stubbin: pathlib.Path,
    home: pathlib.Path,
    *args: str,
    stdin: str = "",
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(workdir / "run.sh"), *args],
        cwd=workdir,
        env=_sandbox_env(stubbin, home),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_in_pty(
    workdir: pathlib.Path,
    stubbin: pathlib.Path,
    home: pathlib.Path,
    stdin: str,
) -> subprocess.CompletedProcess:
    """Drive the menu. The answer has to reach run.sh's *own* stdin.

    The pty has to be run.sh's stdin, so the input is fed to script(1) rather than
    piped into run.sh: a pipe would make `[ -t 0 ]` false and skip the menu.
    script forwards its stdin to the pty. A pty echoes CRLF, so callers assert on
    ``(stdout + stderr).replace("\\r", "")``.
    """
    return subprocess.run(
        ["script", "-qec", "bash run.sh", "bash"],
        cwd=workdir,
        env=_sandbox_env(stubbin, home),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _env_value(workdir: pathlib.Path, key: str) -> str:
    for line in (workdir / ".env").read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{key} not written to .env")


# --- no configuration needed --------------------------------------------------


def test_no_vault_anywhere_creates_the_default(tmp_path):
    workdir, stubbin = _sandbox(tmp_path)
    result = _run(workdir, stubbin, tmp_path / "home")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "auto-create 'agent-vault'" in result.stdout
    assert _env_value(workdir, "OBSIDIAN_VAULT_PARENT_HOST") == "./vaults"
    assert "STUB DOCKER: compose up -d --build" in result.stdout


def test_a_single_vault_is_used_without_asking(tmp_path):
    """The common case stays zero-config: no prompt, no name in .env."""
    workdir, stubbin = _sandbox(tmp_path)
    _make_vault(workdir / "vaults" / "ck")

    result = _run(workdir, stubbin, tmp_path / "home")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Which one should the agent use?" not in result.stdout
    assert _env_value(workdir, "OBSIDIAN_VAULT_PARENT_HOST") == "./vaults"


# --- several vaults -----------------------------------------------------------


def test_several_vaults_without_a_tty_fail_with_the_candidate_names(tmp_path):
    """A script in CI must get a usable error, not a hang and not a guess."""
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work", "archive"):
        _make_vault(workdir / "vaults" / name)

    result = _run(workdir, stubbin, tmp_path / "home", "--no-vault-prompt")

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "personal" in combined and "work" in combined and "archive" in combined
    assert "OBSIDIAN_VAULT_NAME" in combined
    assert "STUB DOCKER" not in result.stdout, "must not start anything when ambiguous"


def test_vault_flag_selects_without_a_prompt_and_is_persisted(tmp_path):
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work", "archive"):
        _make_vault(workdir / "vaults" / name)

    result = _run(workdir, stubbin, tmp_path / "home", "--vault", "work")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Vault: work" in result.stdout
    assert _env_value(workdir, "OBSIDIAN_VAULT_NAME") == "work"
    assert _env_value(workdir, "OBSIDIAN_VAULT_PARENT_HOST") == "./vaults"


def test_vault_flag_persists_only_what_the_user_just_asked_for(tmp_path):
    """An .env that already names a vault must not be rewritten by a bare run."""
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work"):
        _make_vault(workdir / "vaults" / name)
    (workdir / ".env").write_text("OBSIDIAN_VAULT_NAME=personal\n")

    result = _run(workdir, stubbin, tmp_path / "home")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _env_value(workdir, "OBSIDIAN_VAULT_NAME") == "personal"


def test_a_name_that_matches_nothing_fails_with_the_candidates(tmp_path):
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work"):
        _make_vault(workdir / "vaults" / name)

    result = _run(workdir, stubbin, tmp_path / "home", "--vault", "wrok")

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "wrok" in combined
    assert "personal" in combined and "work" in combined
    assert "STUB DOCKER" not in result.stdout


def test_a_vault_obsidian_already_knows_is_offered_and_repoints_the_mount(tmp_path):
    """'I already have a lot of vaults' must not require rearranging them.

    The vault lives outside the managed parent, so the mount is repointed at the
    vault's own parent -- no copying, no symlink, and the real name stays in the
    path so provenance resolves it with nothing configured.
    """
    workdir, stubbin = _sandbox(tmp_path)
    real = _make_vault(tmp_path / "notes" / "journal")
    home = tmp_path / "home"
    (home / ".config" / "obsidian").mkdir(parents=True)
    (home / ".config" / "obsidian" / "obsidian.json").write_text(
        '{"vaults": {"abc": {"path": "%s"}}}' % real
    )

    result = _run(workdir, stubbin, home, "--vault", "journal")

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Vault parent: {real.parent}" in result.stdout
    assert _env_value(workdir, "OBSIDIAN_VAULT_NAME") == "journal"
    assert _env_value(workdir, "OBSIDIAN_VAULT_PARENT_HOST") == str(real.parent)
    # Nothing was copied or linked: the user's tree is untouched.
    assert not (workdir / "vaults" / "journal").exists()


# --- the menu -----------------------------------------------------------------


@pytest.mark.skipif(not shutil.which("script"), reason="script(1) is needed to allocate a pty")
@pytest.mark.parametrize("choice,expected", [("2", "work"), ("", "personal")])
def test_the_menu_lists_the_vaults_and_honours_the_choice(tmp_path, choice, expected):
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work"):
        _make_vault(workdir / "vaults" / name)

    result = _run_in_pty(workdir, stubbin, tmp_path / "home", f"{choice}\n")
    combined = result.stdout + result.stderr
    assert "Which one should the agent use?" in combined, combined
    assert "1) personal" in combined and "2) work" in combined
    # A pty echoes CRLF; normalise before asserting on the choice line.
    assert f"Vault: {expected}" in combined.replace("\r", ""), combined
    assert _env_value(workdir, "OBSIDIAN_VAULT_NAME") == expected


@pytest.mark.skipif(not shutil.which("script"), reason="script(1) is needed to allocate a pty")
def test_the_menu_can_import_an_existing_vault(tmp_path):
    """The import option: keep the mount where it is and bring the vault into it.

    Two listed candidates are needed, because a lone candidate is used without
    asking -- the import path is only reachable when there is a genuine choice to
    make. The path is typed rather than picked from the list, so this also covers
    a vault Obsidian does not know about. It is also the highest advertised
    choice, so it pins the top of the range the prompt prints.
    """
    workdir, stubbin = _sandbox(tmp_path)
    # Two listed candidates so the menu appears; the imported vault is deliberately
    # not among them, since the point of "import by path" is a vault this flow
    # could not have discovered.
    _make_vault(workdir / "vaults" / "local-notes")
    _make_vault(workdir / "vaults" / "other-notes")
    real = _make_vault(tmp_path / "notes" / "journal")

    result = _run_in_pty(workdir, stubbin, tmp_path / "home", f"4\n{real}\n")
    combined = (result.stdout + result.stderr).replace("\r", "")
    assert result.returncode == 0, combined
    assert "Import a vault by path" in combined
    assert "Choice [1-4]" in combined, "import is the last advertised choice"
    assert "Absolute path of the vault to import" in combined

    imported = workdir / "vaults" / "journal"
    # A copy, not a symlink: a link out of the bind mount dangles in the container.
    assert imported.is_dir() and not imported.is_symlink()
    assert (imported / "Second Brain").is_dir()
    # The user's vault is untouched.
    assert real.is_dir() and not (real / "journal").exists()
    assert _env_value(workdir, "OBSIDIAN_VAULT_NAME") == "journal"
    # The parent is left where it was: importing exists precisely to avoid
    # repointing the mount.
    assert _env_value(workdir, "OBSIDIAN_VAULT_PARENT_HOST") == "./vaults"


@pytest.mark.skipif(not shutil.which("script"), reason="script(1) is needed to allocate a pty")
def test_the_menu_can_create_a_new_empty_vault(tmp_path):
    """The choice just below the last one: the sibling of the import option.

    Its number used to come from a variable (`last`) that the loop had already
    advanced past the end of the listing, so "the last listed vault" and "create
    new" were the same number by coincidence. The advertised range and the
    handlers disagreed by one at the top; this pins the handler that moved.
    """
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work"):
        _make_vault(workdir / "vaults" / name)

    result = _run_in_pty(workdir, stubbin, tmp_path / "home", "3\n")
    combined = (result.stdout + result.stderr).replace("\r", "")
    assert result.returncode == 0, combined
    assert "Create a new empty vault (agent-vault)" in combined
    assert "out of range" not in combined
    assert "Vault: agent-vault" in combined
    # Nothing was copied: the vault is created inside the container on first boot.
    assert not (workdir / "vaults" / "agent-vault").exists()
    assert _env_value(workdir, "OBSIDIAN_VAULT_NAME") == "agent-vault"


@pytest.mark.skipif(not shutil.which("script"), reason="script(1) is needed to allocate a pty")
def test_the_menu_reports_the_valid_range_and_rejects_a_bad_choice(tmp_path):
    """A wildly out-of-range answer is rejected and nothing is started.

    ``99`` is rejected by any off-by-one, which is why the boundary -- the choice
    one past the advertised range -- has its own test below.
    """
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work"):
        _make_vault(workdir / "vaults" / name)

    result = _run_in_pty(workdir, stubbin, tmp_path / "home", "99\n")
    combined = (result.stdout + result.stderr).replace("\r", "")
    # 2 vaults + "create new" + "import by path" = 4 choices.
    assert "Choice [1-4]" in combined, combined
    assert result.returncode != 0
    assert "out of range" in combined
    assert "STUB DOCKER" not in result.stdout


@pytest.mark.skipif(not shutil.which("script"), reason="script(1) is needed to allocate a pty")
def test_the_menu_rejects_the_choice_past_the_advertised_range(tmp_path):
    """The boundary, which is the case that survived: one past the last choice.

    ``99`` (the test above) is rejected by any off-by-one, so it proved nothing.
    ``5`` is the exact bug: with N listed vaults the accepted range was computed
    as ``1..N+3`` while the prompt advertised ``1..N+2``, so N+3 passed
    validation, matched neither the create nor the import handler, and fell
    through to a `sed -n "5p"` that printed nothing. The caller then died on
    `[ -n "$CHOSEN" ] || exit 1` having printed no diagnostic at all.

    The advertised range and the accepted range must be the same number, so
    this asserts the message names the range the user was actually shown.
    """
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work"):
        _make_vault(workdir / "vaults" / name)

    result = _run_in_pty(workdir, stubbin, tmp_path / "home", "5\n")
    combined = (result.stdout + result.stderr).replace("\r", "")
    assert "Choice [1-4]" in combined, combined
    assert result.returncode != 0
    assert "out of range: '5'" in combined, combined
    assert "expected 1-4" in combined, combined
    # Loud, not a bare non-zero exit: this is what the bug cost the user.
    assert "run.sh:" in combined, combined
    assert "STUB DOCKER" not in result.stdout


# --- the shell itself ---------------------------------------------------------


def test_help_prints_usage_and_no_source(tmp_path):
    """--help must not print the script.

    It used to be `sed -n '2,25p' "$0"`, and the header comment is 20 lines, so the
    last five printed lines were source: `set -euo pipefail`,
    `cd "$(dirname "$0")"`, `PROMPT_FOR_VAULT=1`, `VAULT_OVERRIDE=""`, the comment
    above DEFAULT_VAULT_NAME. A line-number range cannot survive an edit to the
    file it describes, so the text now lives in a `usage()` function; this asserts
    the property structurally rather than re-pinning a range.
    """
    workdir, stubbin = _sandbox(tmp_path)

    result = _run(workdir, stubbin, tmp_path / "home", "--help")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "One-command bootstrap" in result.stdout
    assert "--vault NAME" in result.stdout
    assert "OBSIDIAN_VAULT_NAME" in result.stdout
    # Not one line of the script's own code may appear, which covers every line
    # named in the docstring plus whatever the next edit happens to move. Lines are
    # read whole and stripped, so indentation cannot hide a leak. Bare keywords
    # (`fi`, `esac`, `done`) are excluded: they are substrings of ordinary prose
    # and "fi" appears in "configured", so matching them would prove nothing.
    code_lines = [
        stripped
        for line in RUN_SH.read_text().splitlines()
        if (stripped := line.strip())
        and not stripped.startswith("#")
        and (" " in stripped or "=" in stripped)
    ]
    leaked = [line for line in code_lines if line in result.stdout]
    assert not leaked, f"--help printed the script's own source: {leaked}"
    # Help is answered before anything is configured or started.
    assert "STUB DOCKER" not in result.stdout
    assert not (workdir / ".env").exists()


def test_an_absolute_vault_parent_is_created(tmp_path):
    """An absolute OBSIDIAN_VAULT_PARENT_HOST is scaffolded like a relative one.

    The `mkdir -p` was guarded by `case "$VAULT_PARENT_HOST" in /*) ;;`, which
    only made sense while the default was the relative ./vaults: a user who named
    an absolute parent got a path that was never created, so compose bind-mounted
    a Docker-created directory of its own instead of the one that was configured.
    """
    workdir, stubbin = _sandbox(tmp_path)
    parent = tmp_path / "somewhere-else"
    assert not parent.exists()
    (workdir / ".env").write_text(f"OBSIDIAN_VAULT_PARENT_HOST={parent}\n")

    result = _run(workdir, stubbin, tmp_path / "home")

    assert result.returncode == 0, result.stdout + result.stderr
    assert parent.is_dir()
    assert f"will auto-create 'agent-vault' in {parent}" in result.stdout
    assert _env_value(workdir, "OBSIDIAN_VAULT_PARENT_HOST") == str(parent)
    # The configured parent is the only one that is created: nothing is written
    # into the repo checkout just because ./vaults is the default.
    assert not (workdir / "vaults").exists()
