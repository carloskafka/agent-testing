"""Vault discovery and selection: which Obsidian vault does this run use?

Why this module exists
----------------------
``obsidian-mcp`` accepts exactly one vault path as a CLI argument, so a deployment
has exactly one *active* vault. The question this module answers is therefore not
"which of many" but "which one, and how does the user say so".

The awkward part is that the obvious answer -- mount the parent and take the only
child -- stops working the moment the user has more than one vault, which is the
normal case for anyone who actually uses Obsidian. Guessing there would mean
silently writing a user's notes into the wrong vault, so the rules are:

* one candidate            -> use it, no configuration;
* zero candidates          -> create ``agent-vault`` so a fresh clone runs at all;
* a name is configured     -> use that vault, whatever else is there;
* more than one, no name   -> refuse, and list the candidates by name.

The refusal is deliberate and load-bearing: the alternative is writing to a vault
the user did not choose. ``run.sh`` turns the same discovery into a menu, so the
interactive path never reaches the error.

Two deployment shapes, one rule
------------------------------
The vault is discovered at **runtime from the directory that gets bind-mounted**,
and that directory is the vault's *parent*, never the vault itself. Mounting the
vault directly (``.../ck:/vault``) hands the container a path whose last segment is
the mount point, so the real host name is unrecoverable -- ``basename`` says
``vault``, obsidian-mcp's ``vault_info`` says ``vault``, and every ``**Sources**``
line in the agent's output names a vault that does not exist. Mounting the parent
keeps the real name in the path, so renaming the folder on the host is enough and
no name has to be configured anywhere.

``resolve-vault.sh`` mirrors :func:`select_vault` in POSIX shell. The two must never
disagree about which vault is in use: an agent pointed at a different vault than its
MCP server is worse than an agent that does not start, so the shell script exits
non-zero on ambiguity where the Python path degrades to a clear error message.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

__all__ = [
    "BRAIN_DIR",
    "DEFAULT_VAULT_NAME",
    "VAULT_NAME_ENV",
    "Vault",
    "VaultSelection",
    "VaultSelectionError",
    "discover_vaults",
    "format_candidates",
    "import_vault",
    "obsidian_config_path",
    "obsidian_vaults",
    "select_vault",
]

#: Directory that marks a directory as *our* vault layout. Used for detection only;
#: a vault that has never been written to by this project will not have it.
BRAIN_DIR = "Second Brain"

#: Directory Obsidian itself creates inside every vault. The reliable marker when
#: importing a vault the user already has.
OBSIDIAN_DIR = ".obsidian"

#: Name of the vault auto-created when the mounted parent is empty, so a fresh
#: clone-and-run works with no vault at all.
DEFAULT_VAULT_NAME = "agent-vault"

#: Selects the active vault by directory name when the parent holds several.
#: Deliberately distinct from ``VAULT_NAME`` (see ``sources.py``), which only
#: *labels* the resolved vault for provenance and never chooses it.
VAULT_NAME_ENV = "OBSIDIAN_VAULT_NAME"


class VaultSelectionError(RuntimeError):
    """No single vault could be chosen, and guessing is not an option.

    Carries the candidate names so the caller can print an actionable message
    instead of a bare failure.
    """

    def __init__(self, message: str, candidates: list[str] | None = None) -> None:
        super().__init__(message)
        self.candidates = list(candidates or [])


@dataclass(frozen=True)
class Vault:
    """One vault directory: where it is and what it is called."""

    name: str
    path: str
    #: True when the directory carries an Obsidian marker, i.e. it is a real vault
    #: rather than some unrelated directory that happens to sit in the parent.
    looks_like_vault: bool = False

    @property
    def markers(self) -> list[str]:
        found = []
        if os.path.isdir(os.path.join(self.path, OBSIDIAN_DIR)):
            found.append(OBSIDIAN_DIR)
        if os.path.isdir(os.path.join(self.path, BRAIN_DIR)):
            found.append(BRAIN_DIR)
        return found


@dataclass(frozen=True)
class VaultSelection:
    """The chosen vault plus why it was chosen.

    ``needs_create`` means the directory does not exist yet and the caller must
    create it before writing. Kept separate from the path so a read-only turn
    against a missing vault can report the intended path without creating
    anything.
    """

    vault: Vault
    reason: str
    candidates: tuple[str, ...] = ()
    needs_create: bool = False


def looks_like_vault(path: str) -> bool:
    """True when the directory carries an Obsidian or second-brain marker."""
    return os.path.isdir(os.path.join(path, OBSIDIAN_DIR)) or os.path.isdir(
        os.path.join(path, BRAIN_DIR)
    )


def discover_vaults(parent: str) -> list[Vault]:
    """Every non-hidden child directory of ``parent``, as a vault candidate.

    Directories are not filtered by marker: a vault the user has just created in
    Obsidian but never written to has no ``Second Brain`` yet, and refusing to
    offer it would make a brand new vault unselectable. The marker is reported
    per candidate so a UI can prefer the real ones.

    Returns ``[]`` for a missing or unreadable ``parent`` rather than raising --
    discovery runs at import time and must not be able to break the agent.
    """
    try:
        entries = sorted(os.scandir(parent), key=lambda e: e.name)
    except OSError:
        return []
    return [
        Vault(name=e.name, path=e.path, looks_like_vault=looks_like_vault(e.path))
        for e in entries
        if e.is_dir() and not e.name.startswith(".")
    ]


def select_vault(
    parent: str,
    *,
    name: str | None = None,
    create_default: bool = True,
) -> VaultSelection:
    """Choose the one vault to use out of ``parent``.

    Resolution order:

    1. ``name`` -- the configured choice always wins, including when the parent
       holds other vaults. A name that matches no child is an error rather than a
       silent fallback, so a typo is visible immediately.
    2. ``parent`` is itself a vault (it holds ``Second Brain``) -- used as-is. This
       covers pointing ``SECOND_BRAIN_VAULT`` straight at a vault.
    3. Exactly one child -- that child, which is the common single-vault case and
       needs no configuration.
    4. No children -- ``DEFAULT_VAULT_NAME`` under the parent, flagged
       ``needs_create``.
    5. Several children and no name -- :class:`VaultSelectionError`, carrying the
       candidate names.
    """
    parent = (parent or "").strip()
    wanted = (name or "").strip()

    if wanted:
        # Pointed straight at the vault it names: nothing to descend into.
        if os.path.isdir(os.path.join(parent, BRAIN_DIR)) and os.path.basename(
            os.path.normpath(parent)
        ) == wanted:
            return VaultSelection(
                Vault(wanted, parent, looks_like_vault=True),
                reason=f"{VAULT_NAME_ENV}={wanted} names the mounted directory itself",
            )
        target = os.path.join(parent, wanted)
        if os.path.isdir(target):
            return VaultSelection(
                Vault(wanted, target, looks_like_vault=looks_like_vault(target)),
                reason=f"{VAULT_NAME_ENV}={wanted}",
            )
        candidates = tuple(v.name for v in discover_vaults(parent))
        raise VaultSelectionError(
            f"{VAULT_NAME_ENV}={wanted!r} is not a directory under {parent!r}"
            + (f" (available: {', '.join(candidates)})" if candidates else ""),
            candidates=list(candidates),
        )

    if os.path.isdir(os.path.join(parent, BRAIN_DIR)):
        return VaultSelection(
            Vault(os.path.basename(os.path.normpath(parent)), parent, True),
            reason="the mounted directory is itself a vault",
        )

    candidates = discover_vaults(parent)
    if not candidates:
        if not create_default:
            raise VaultSelectionError(f"no vault directory found in {parent!r}")
        target = os.path.join(parent, DEFAULT_VAULT_NAME)
        return VaultSelection(
            Vault(DEFAULT_VAULT_NAME, target, False),
            reason=f"{parent!r} is empty; using the auto-created {DEFAULT_VAULT_NAME!r}",
            needs_create=True,
        )

    if len(candidates) == 1:
        only = candidates[0]
        return VaultSelection(only, reason="the only vault in the mounted parent")

    raise VaultSelectionError(
        f"{parent!r} holds {len(candidates)} vaults "
        f"({', '.join(v.name for v in candidates)}); set {VAULT_NAME_ENV} to the one to use",
        candidates=[v.name for v in candidates],
    )


def format_candidates(candidates) -> str:
    """Render candidate vaults as an indented, aligned list for a menu or an error."""
    entries = list(candidates)
    if not entries:
        return "  (none found)"
    width = max(len(v.name) for v in entries)
    lines = []
    for vault in entries:
        marker = f"  [{', '.join(vault.markers)}]" if vault.markers else ""
        lines.append(f"  {vault.name.ljust(width)}  {vault.path}{marker}")
    return "\n".join(lines)


# --- existing Obsidian vaults -------------------------------------------------


def obsidian_config_path(config_path: str | None = None) -> str:
    """Path to Obsidian's own config, which lists every vault the user has open.

    Honours ``$OBSIDIAN_CONFIG_DIR`` because Obsidian itself does, then falls back
    to the documented per-platform locations.
    """
    if config_path:
        return config_path
    explicit = os.environ.get("OBSIDIAN_CONFIG_DIR", "").strip()
    if explicit:
        return os.path.join(explicit, "obsidian.json")
    return os.path.join(
        os.path.expanduser("~"), ".config", "obsidian", "obsidian.json"
    )


def obsidian_vaults(config_path: str | None = None) -> list[Vault]:
    """Every vault Obsidian knows about, read from its config file.

    This is what makes "I already have a lot of vaults" work without the user
    restructuring anything: the answer is already on disk, written by Obsidian.

    The file is a cache, so entries can point at a vault that has since been moved
    or deleted. Non-existent paths are dropped, and a malformed or missing file
    yields ``[]`` -- discovery must never be the reason the agent fails to start.
    """
    path = obsidian_config_path(config_path)
    try:
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return []

    entries = config.get("vaults") if isinstance(config, dict) else None
    if not isinstance(entries, dict):
        return []

    found: dict[str, Vault] = {}
    for vault_id, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        raw = entry.get("path")
        if not isinstance(raw, str) or not raw.strip():
            continue
        resolved = os.path.abspath(os.path.expanduser(raw.strip()))
        if not os.path.isdir(resolved):
            continue
        name = os.path.basename(os.path.normpath(resolved)) or str(vault_id)
        # Several config ids can name one directory; keep the first.
        found.setdefault(
            resolved, Vault(name=name, path=resolved, looks_like_vault=True)
        )
    return sorted(found.values(), key=lambda v: v.name.lower())


# --- importing a vault the user already has -----------------------------------


def import_vault(source: str, parent: str, *, mode: str = "copy") -> Vault:
    """Make an existing vault reachable from the managed parent directory.

    The primary way to use a real vault needs no import at all: point
    ``OBSIDIAN_VAULT_PARENT_HOST`` at the directory the vault already lives in and
    set ``OBSIDIAN_VAULT_NAME``. Import is for when the managed parent has to stay
    where it is, and it **copies** by default.

    Copying is the default because a symlink would be correct on the host and
    *broken inside the container*. Docker bind-mounts the parent, and a symlink
    whose target lies outside the mounted tree dangles, because that path does not
    exist in the container's filesystem. Verified, not assumed: a
    ``ck -> /home/.../ck-vault/ck`` link inside the mount lists fine on the host
    and gives "No such file or directory" at ``/vaults/ck/``, which stops
    ``obsidian-mcp`` at startup. There is also no in-tree case to prefer: a source
    already inside the parent resolves to the very path the import would create,
    so it is refused as an overwrite rather than linked to itself.

    Pass ``mode="link"`` to symlink anyway -- correct for a run outside Docker,
    where there is no mount to dangle in. The trade-off is real: with a link the
    user's vault stays the single source of truth, and with a copy the agent's
    notes are kept entirely apart and will not drift back.

    Refuses to overwrite an existing entry. An import that silently replaced a
    vault would be the worst possible failure mode here.
    """
    source = os.path.abspath(os.path.expanduser((source or "").strip()))
    if not os.path.isdir(source):
        raise VaultSelectionError(f"import source is not a directory: {source!r}")
    if not looks_like_vault(source):
        raise VaultSelectionError(
            f"{source!r} has no {OBSIDIAN_DIR}/ or {BRAIN_DIR}/, "
            "so it does not look like an Obsidian vault"
        )

    parent = os.path.abspath(os.path.expanduser((parent or "").strip()))
    name = os.path.basename(os.path.normpath(source))
    target = os.path.join(parent, name)
    if os.path.lexists(target):
        raise VaultSelectionError(
            f"{target!r} already exists; remove it or pick a different vault"
        )
    if mode not in ("copy", "link"):
        raise VaultSelectionError(
            f"unknown import mode {mode!r}; use 'copy' or 'link'"
        )

    os.makedirs(parent, exist_ok=True)
    if mode == "link":
        os.symlink(source, target)
    else:
        shutil.copytree(source, target, symlinks=True)

    return Vault(name=name, path=target, looks_like_vault=True)
