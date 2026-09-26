"""Vault discovery and selection: the rules that decide which vault a run uses.

``obsidian-mcp`` serves one vault, so the interesting cases are all about refusing
to guess. These tests are the guard on that: the cost of a wrong choice is a user's
notes landing in the wrong vault, silently.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from text_summarizer import vaults
from text_summarizer.vaults import (
    DEFAULT_VAULT_NAME,
    VAULT_NAME_ENV,
    VaultSelectionError,
    discover_vaults,
    import_vault,
    obsidian_vaults,
    select_vault,
)


def _make_vault(parent, name: str, *, second_brain: bool = True) -> str:
    path = os.path.join(parent, name)
    os.makedirs(path, exist_ok=True)
    if second_brain:
        os.makedirs(os.path.join(path, vaults.BRAIN_DIR), exist_ok=True)
    return path


# --- discovery ----------------------------------------------------------------


def test_discover_lists_non_hidden_child_dirs_only(tmp_path):
    _make_vault(str(tmp_path), "personal")
    _make_vault(str(tmp_path), "work")
    os.makedirs(tmp_path / ".obsidian-cache")     # hidden: Obsidian's own state
    (tmp_path / "notes.md").write_text("not a vault")

    found = discover_vaults(str(tmp_path))
    assert [v.name for v in found] == ["personal", "work"]


def test_discover_reports_obsidian_markers(tmp_path):
    _make_vault(str(tmp_path), "with-obsidian", second_brain=False)
    os.makedirs(tmp_path / "with-obsidian" / vaults.OBSIDIAN_DIR)
    _make_vault(str(tmp_path), "with-brain")

    by_name = {v.name: v for v in discover_vaults(str(tmp_path))}
    assert by_name["with-obsidian"].looks_like_vault
    assert by_name["with-brain"].looks_like_vault
    assert by_name["with-brain"].markers == [vaults.BRAIN_DIR]


def test_discover_of_a_missing_parent_is_empty_not_an_error(tmp_path):
    """Discovery runs at import time; it must never be why the agent fails."""
    assert discover_vaults(str(tmp_path / "nope")) == []


# --- selection ----------------------------------------------------------------


def test_single_vault_needs_no_configuration(tmp_path):
    _make_vault(str(tmp_path), "ck")
    selection = select_vault(str(tmp_path))
    assert selection.vault.name == "ck"
    assert selection.vault.path == os.path.join(str(tmp_path), "ck")
    assert not selection.needs_create


def test_empty_parent_resolves_to_the_auto_created_default(tmp_path):
    selection = select_vault(str(tmp_path))
    assert selection.vault.name == DEFAULT_VAULT_NAME
    assert selection.needs_create
    assert not os.path.exists(selection.vault.path), "selection must not create anything"


def test_parent_that_is_itself_a_vault_is_used_as_is(tmp_path):
    """SECOND_BRAIN_VAULT pointed straight at a vault must not gain a child."""
    _make_vault(str(tmp_path), "solo")
    selection = select_vault(os.path.join(str(tmp_path), "solo"))
    assert selection.vault.name == "solo"
    assert selection.vault.path == os.path.join(str(tmp_path), "solo")


def test_several_vaults_without_a_name_is_an_error_listing_them(tmp_path):
    """The load-bearing case: this is the bug, not a detail.

    Before selection existed, the rule was "the single child", so anybody with two
    vaults got a refusal they could not act on. Now they get the names.
    """
    _make_vault(str(tmp_path), "personal")
    _make_vault(str(tmp_path), "work")

    with pytest.raises(VaultSelectionError) as excinfo:
        select_vault(str(tmp_path))
    assert excinfo.value.candidates == ["personal", "work"]
    message = str(excinfo.value)
    assert VAULT_NAME_ENV in message
    assert "personal" in message and "work" in message


def test_a_configured_name_wins_over_other_vaults(tmp_path):
    _make_vault(str(tmp_path), "personal")
    _make_vault(str(tmp_path), "work")
    _make_vault(str(tmp_path), "archive")

    selection = select_vault(str(tmp_path), name="work")
    assert selection.vault.name == "work"
    assert VAULT_NAME_ENV in selection.reason


def test_a_configured_name_that_matches_nothing_is_an_error(tmp_path):
    """A typo must be loud, not a silent fallback to some other vault."""
    _make_vault(str(tmp_path), "personal")
    with pytest.raises(VaultSelectionError) as excinfo:
        select_vault(str(tmp_path), name="wrok")
    assert "wrok" in str(excinfo.value)
    assert excinfo.value.candidates == ["personal"]


def test_a_configured_name_may_name_the_mounted_directory_itself(tmp_path):
    _make_vault(str(tmp_path), "solo")
    selection = select_vault(os.path.join(str(tmp_path), "solo"), name="solo")
    assert selection.vault.path == os.path.join(str(tmp_path), "solo")
    assert not selection.needs_create


def test_selecting_inside_an_empty_parent_reports_no_candidates(tmp_path):
    with pytest.raises(VaultSelectionError) as excinfo:
        select_vault(str(tmp_path), name="whatever")
    assert excinfo.value.candidates == []


def test_create_default_false_refuses_an_empty_parent(tmp_path):
    with pytest.raises(VaultSelectionError):
        select_vault(str(tmp_path), create_default=False)


# --- Obsidian's own config ----------------------------------------------------


def test_obsidian_vaults_are_read_from_the_app_config(tmp_path):
    """The point of this source: the user's real vaults are already on disk."""
    personal = _make_vault(str(tmp_path / "vaults" / "personal"), "personal")
    work = _make_vault(str(tmp_path / "elsewhere" / "work"), "work")
    config = tmp_path / "obsidian.json"
    config.write_text(
        json.dumps(
            {
                "vaults": {
                    "abc123": {"path": personal, "ts": 1, "open": True},
                    "def456": {"path": work},
                }
            }
        )
    )

    found = {v.name: v.path for v in obsidian_vaults(str(config))}
    assert found == {"personal": personal, "work": work}


def test_obsidian_vaults_drops_entries_whose_path_is_gone(tmp_path):
    """Obsidian's config is a cache; moved or deleted vaults are still listed."""
    alive = _make_vault(str(tmp_path / "personal"), "personal")
    config = tmp_path / "obsidian.json"
    config.write_text(
        json.dumps(
            {
                "vaults": {
                    "a": {"path": alive},
                    "b": {"path": str(tmp_path / "deleted")},
                }
            }
        )
    )
    assert [v.name for v in obsidian_vaults(str(config))] == ["personal"]


@pytest.mark.parametrize(
    "content", ["", "not json at all", "{}", '{"vaults": "wrong type"}', '{"vaults": {}}']
)
def test_a_malformed_obsidian_config_yields_nothing(tmp_path, content):
    config = tmp_path / "obsidian.json"
    config.write_text(content)
    assert obsidian_vaults(str(config)) == []


def test_a_missing_obsidian_config_yields_nothing(tmp_path):
    assert obsidian_vaults(str(tmp_path / "absent.json")) == []


def test_two_config_ids_for_one_directory_collapse_to_one(tmp_path):
    real = _make_vault(str(tmp_path / "personal"), "personal")
    config = tmp_path / "obsidian.json"
    config.write_text(json.dumps({"vaults": {"a": {"path": real}, "b": {"path": real}}}))
    assert len(obsidian_vaults(str(config))) == 1


def test_obsidian_config_path_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OBSIDIAN_CONFIG_DIR", str(tmp_path / "custom"))
    assert vaults.obsidian_config_path() == str(tmp_path / "custom" / "obsidian.json")


# --- importing ----------------------------------------------------------------


def test_import_copies_a_vault_that_lives_outside_the_mount(tmp_path):
    """A symlink would be correct on the host and broken in the container.

    Docker bind-mounts the parent, so a symlink pointing outside it dangles
    inside the container -- the path simply does not exist there. Verified live:
    such a link takes obsidian-mcp down at startup. So an out-of-tree vault is
    copied, and the copy is what both sides then use.
    """
    source = _make_vault(str(tmp_path / "real" / "personal"), "personal")
    pathlib.Path(source, "note.md").write_text("real note")

    parent = str(tmp_path / "managed")
    os.makedirs(parent)
    imported = import_vault(source, parent)

    assert imported.path == os.path.join(parent, "personal")
    assert not os.path.islink(imported.path), "an out-of-tree symlink dangles in the mount"
    assert pathlib.Path(imported.path, "note.md").read_text() == "real note"
    # The original is left exactly as it was.
    assert os.path.isfile(os.path.join(source, "note.md"))
    assert sorted(os.listdir(source)) == sorted(os.listdir(imported.path))


def test_import_link_mode_is_available_for_a_run_outside_docker(tmp_path):
    """An explicit request is honoured: without a mount there is nothing to dangle."""
    source = _make_vault(str(tmp_path / "real" / "personal"), "personal")
    pathlib.Path(source, "note.md").write_text("real note")
    parent = str(tmp_path / "managed")
    os.makedirs(parent)

    imported = import_vault(source, parent, mode="link")
    assert os.path.islink(imported.path)
    assert pathlib.Path(imported.path, "note.md").read_text() == "real note"


def test_import_refuses_a_source_already_inside_the_parent(tmp_path):
    """There is no in-tree case to prefer: the target would be the source."""
    parent = str(tmp_path / "managed")
    source = _make_vault(os.path.join(parent, "personal"), "personal")

    with pytest.raises(VaultSelectionError, match="already exists"):
        import_vault(source, parent)


def test_import_copy_mode_duplicates_the_tree(tmp_path):
    source = _make_vault(str(tmp_path / "real" / "personal"), "personal")
    pathlib.Path(source, "note.md").write_text("real note")

    parent = str(tmp_path / "managed")
    os.makedirs(parent)
    imported = import_vault(source, parent, mode="copy")

    assert not os.path.islink(imported.path)
    assert pathlib.Path(imported.path, "note.md").read_text() == "real note"


def test_import_refuses_to_overwrite_an_existing_entry(tmp_path):
    """The one failure mode that must never happen silently."""
    source = _make_vault(str(tmp_path / "real" / "personal"), "personal")
    parent = str(tmp_path / "managed")
    _make_vault(parent, "personal")

    with pytest.raises(VaultSelectionError, match="already exists"):
        import_vault(source, parent)


def test_import_rejects_a_source_that_is_not_a_vault(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(VaultSelectionError, match="does not look like"):
        import_vault(str(plain), str(tmp_path / "managed"))


def test_import_rejects_a_missing_source(tmp_path):
    with pytest.raises(VaultSelectionError, match="not a directory"):
        import_vault(str(tmp_path / "absent"), str(tmp_path / "managed"))


def test_import_rejects_an_unknown_mode(tmp_path):
    source = _make_vault(str(tmp_path / "real" / "personal"), "personal")
    with pytest.raises(VaultSelectionError, match="unknown import mode"):
        import_vault(source, str(tmp_path / "managed"), mode="hardlink")


def test_an_imported_vault_is_then_selectable(tmp_path):
    """The two halves have to compose: import is only useful if selection finds it."""
    source = _make_vault(str(tmp_path / "real" / "personal"), "personal")
    parent = str(tmp_path / "managed")
    os.makedirs(parent)
    import_vault(source, parent)

    selection = select_vault(parent, name="personal")
    assert selection.vault.name == "personal"
    assert os.path.isdir(selection.vault.path)
    assert not selection.needs_create


# --- second_brain integration -------------------------------------------------


def test_resolve_vault_root_honours_the_configured_name(monkeypatch, tmp_path):
    from text_summarizer import second_brain

    _make_vault(str(tmp_path), "personal")
    _make_vault(str(tmp_path), "work")

    monkeypatch.setenv(VAULT_NAME_ENV, "work")
    assert second_brain.resolve_vault_root(str(tmp_path)) == os.path.join(
        str(tmp_path), "work"
    )


def test_resolve_vault_root_creates_the_default_lazily(monkeypatch, tmp_path):
    from text_summarizer import second_brain

    root = second_brain.resolve_vault_root(str(tmp_path))
    assert root == os.path.join(str(tmp_path), DEFAULT_VAULT_NAME)
    assert os.path.isdir(os.path.join(root, vaults.BRAIN_DIR))


def test_resolve_vault_root_degrades_instead_of_guessing(monkeypatch, tmp_path, capsys):
    """An ambiguous mount must not write into a vault nobody chose.

    Returns the unresolved parent so a read-only turn still returns notes; writes
    then land outside any vault, which is visibly wrong rather than silently wrong.
    ``resolve-vault.sh`` exits non-zero on the same input.
    """
    from text_summarizer import second_brain

    _make_vault(str(tmp_path), "personal")
    _make_vault(str(tmp_path), "work")
    monkeypatch.delenv(VAULT_NAME_ENV, raising=False)

    assert second_brain.resolve_vault_root(str(tmp_path)) == str(tmp_path)
    printed = capsys.readouterr().out
    assert VAULT_NAME_ENV in printed
    assert "personal" in printed and "work" in printed


# --- resolve-vault.sh ---------------------------------------------------------
#
# The script and vaults.py implement the same rules and MUST agree: an agent
# pointed at a different vault than its MCP server is worse than an agent that
# does not start. These run the real script with a stub binary on PATH, so the
# shell logic is covered rather than assumed. No daemon needed.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOLVER = os.path.join(REPO_ROOT, "resolve-vault.sh")


def _run_resolver(parent: str, *, vault_name: str | None = None, tmp_path=None):
    """Run resolve-vault.sh against a stub obsidian-mcp; return (rc, stdout+stderr)."""
    import subprocess

    stub_dir = tmp_path / "stubbin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "obsidian-mcp"
    stub.write_text('#!/bin/sh\necho "SERVE: $*"\n')
    stub.chmod(0o755)

    env = dict(os.environ, PATH=f"{stub_dir}:{os.environ['PATH']}", VAULT_PARENT=parent)
    env.pop("VAULT_NAME", None)
    env.pop("OBSIDIAN_VAULT_NAME", None)
    if vault_name is not None:
        env["VAULT_NAME"] = vault_name
    result = subprocess.run(
        ["sh", RESOLVER], env=env, capture_output=True, text=True, timeout=30
    )
    return result.returncode, result.stdout + result.stderr


def test_resolver_serves_the_only_vault(tmp_path):
    parent = str(tmp_path / "mount")
    os.makedirs(parent)
    _make_vault(parent, "ck")
    rc, output = _run_resolver(parent, tmp_path=tmp_path)
    assert rc == 0, output
    assert f"SERVE: --http --host 0.0.0.0 --port 37842 {parent}/ck" in output


def test_resolver_auto_creates_the_default_vault(tmp_path):
    parent = str(tmp_path / "mount")
    os.makedirs(parent)
    rc, output = _run_resolver(parent, tmp_path=tmp_path)
    assert rc == 0, output
    assert f"SERVE: --http --host 0.0.0.0 --port 37842 {parent}/{DEFAULT_VAULT_NAME}" in output
    assert os.path.isdir(os.path.join(parent, DEFAULT_VAULT_NAME, vaults.BRAIN_DIR))


def test_resolver_serves_a_vault_named_by_VAULT_NAME(tmp_path):
    """The multi-vault fix: several vaults, one named, no guessing."""
    parent = str(tmp_path / "mount")
    os.makedirs(parent)
    for name in ("personal", "work", "archive"):
        _make_vault(parent, name)

    rc, output = _run_resolver(parent, vault_name="work", tmp_path=tmp_path)
    assert rc == 0, output
    assert f"{parent}/work" in output
    assert "personal" not in output.split("SERVE:")[-1]


def test_resolver_refuses_an_ambiguous_mount_and_lists_the_candidates(tmp_path):
    parent = str(tmp_path / "mount")
    os.makedirs(parent)
    for name in ("personal", "work"):
        _make_vault(parent, name)

    rc, output = _run_resolver(parent, tmp_path=tmp_path)
    assert rc != 0
    assert "refusing to guess" in output
    assert "OBSIDIAN_VAULT_NAME" in output
    # The point of listing them: the user can act on the error immediately.
    assert "personal" in output and "work" in output


def test_resolver_rejects_a_name_that_matches_nothing(tmp_path):
    parent = str(tmp_path / "mount")
    os.makedirs(parent)
    _make_vault(parent, "personal")

    rc, output = _run_resolver(parent, vault_name="wrok", tmp_path=tmp_path)
    assert rc != 0
    assert "wrok" in output
    assert "personal" in output, "the error should list what is available"


def test_resolver_uses_the_mounted_directory_when_it_is_a_vault(tmp_path):
    parent = str(tmp_path / "mount" / "solo")
    _make_vault(str(tmp_path / "mount"), "solo")
    rc, output = _run_resolver(parent, tmp_path=tmp_path)
    assert rc == 0, output
    assert f"SERVE: --http --host 0.0.0.0 --port 37842 {parent}" in output


def test_resolver_honours_the_port_override(tmp_path):
    parent = str(tmp_path / "mount")
    os.makedirs(parent)
    _make_vault(parent, "ck")
    import subprocess

    stub_dir = tmp_path / "stubbin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "obsidian-mcp"
    stub.write_text('#!/bin/sh\necho "SERVE: $*"\n')
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{stub_dir}:{os.environ['PATH']}",
        VAULT_PARENT=parent,
        OBSIDIAN_MCP_PORT="39999",
    )
    env.pop("VAULT_NAME", None)
    result = subprocess.run(
        ["sh", RESOLVER], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--port 39999" in result.stdout

