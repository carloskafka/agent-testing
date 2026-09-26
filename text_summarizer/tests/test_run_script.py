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

pytestmark = pytest.mark.skipif(
    not shutil.which("bash"), reason="run.sh needs bash"
)


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


def _run(
    workdir: pathlib.Path,
    stubbin: pathlib.Path,
    home: pathlib.Path,
    *args: str,
    stdin: str = "",
) -> subprocess.CompletedProcess:
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
    return subprocess.run(
        ["bash", str(workdir / "run.sh"), *args],
        cwd=workdir,
        env=env,
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


@pytest.mark.skipif(
    not shutil.which("script"), reason="script(1) is needed to allocate a pty"
)
@pytest.mark.parametrize("choice,expected", [("2", "work"), ("", "personal")])
def test_the_menu_lists_the_vaults_and_honours_the_choice(tmp_path, choice, expected):
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work"):
        _make_vault(workdir / "vaults" / name)

    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        # The pty has to be run.sh's *own* stdin, so the answer is fed to script(1)
        # rather than piped into run.sh: a pipe would make `[ -t 0 ]` false and
        # skip the menu. script forwards its stdin to the pty.
        ["script", "-qec", "bash run.sh", "bash"],
        cwd=workdir,
        env={
            **os.environ,
            "PATH": f"{stubbin}:{os.environ['PATH']}",
            "HOME": str(home),
        },
        input=f"{choice}\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    assert "Which one should the agent use?" in combined, combined
    assert "1) personal" in combined and "2) work" in combined
    # A pty echoes CRLF; normalise before asserting on the choice line.
    assert f"Vault: {expected}" in combined.replace("\r", ""), combined
    assert f"Vault: {expected}" in combined, combined
    assert _env_value(workdir, "OBSIDIAN_VAULT_NAME") == expected


@pytest.mark.skipif(
    not shutil.which("script"), reason="script(1) is needed to allocate a pty"
)
def test_the_menu_can_import_an_existing_vault(tmp_path):
    """The import option: keep the mount where it is and bring the vault into it.

    Two listed candidates are needed, because a lone candidate is used without
    asking -- the import path is only reachable when there is a genuine choice to
    make. The path is typed rather than picked from the list, so this also covers
    a vault Obsidian does not know about.
    """
    workdir, stubbin = _sandbox(tmp_path)
    # Two listed candidates so the menu appears; the imported vault is deliberately
    # not among them, since the point of "import by path" is a vault this flow
    # could not have discovered.
    _make_vault(workdir / "vaults" / "local-notes")
    _make_vault(workdir / "vaults" / "other-notes")
    real = _make_vault(tmp_path / "notes" / "journal")

    result = subprocess.run(
        # The pty has to be run.sh's own stdin, so the input goes to script(1):
        # a pipe into run.sh would make `[ -t 0 ]` false and skip the menu.
        ["script", "-qec", "bash run.sh", "bash"],
        cwd=workdir,
        env={
            **os.environ,
            "PATH": f"{stubbin}:{os.environ['PATH']}",
            "HOME": str(tmp_path / "home"),
        },
        input=f"4\n{real}\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (result.stdout + result.stderr).replace("\r", "")
    assert result.returncode == 0, combined
    assert "Import a vault by path" in combined
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


def test_the_menu_reports_the_valid_range_and_rejects_a_bad_choice(tmp_path):
    """The prompt used to offer a range one wider than the list; both are pinned."""
    workdir, stubbin = _sandbox(tmp_path)
    for name in ("personal", "work"):
        _make_vault(workdir / "vaults" / name)

    result = subprocess.run(
        ["script", "-qec", "bash run.sh", "bash"],
        cwd=workdir,
        env={
            **os.environ,
            "PATH": f"{stubbin}:{os.environ['PATH']}",
            "HOME": str(tmp_path / "home"),
        },
        input="99\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (result.stdout + result.stderr).replace("\r", "")
    # 2 vaults + "create new" + "import by path" = 4 choices.
    assert "Choice [1-4]" in combined, combined
    assert result.returncode != 0
    assert "out of range" in combined
    assert "STUB DOCKER" not in result.stdout
