#!/bin/sh
# Resolve the vault directory and exec the obsidian-mcp HTTP server.
#
# obsidian-mcp takes exactly one vault path, so this script's whole job is picking
# that one and refusing to guess. It mirrors text_summarizer/vaults.py
# (select_vault) so the agent and its MCP server can never disagree about which
# vault is in use -- an agent pointed at a different vault than its MCP server is
# worse than an agent that does not start.
#
# The vault's *parent* is bind-mounted so the real vault name survives in the path:
# mounting the vault itself (host .../ck-vault/ck -> /vault) flattens the name to
# "vault", which is unrecoverable from inside the container.
#
#   VAULT_PARENT  in-container directory holding the vault(s)   (default /vaults)
#   VAULT_NAME    which child of VAULT_PARENT to serve; the same value as
#                 OBSIDIAN_VAULT_NAME on the Python side. Optional when there is
#                 only one vault, required when there are several.
set -eu

VAULT_PARENT="${VAULT_PARENT:-/vaults}"
VAULT_SELECT="${VAULT_NAME:-${OBSIDIAN_VAULT_NAME:-}}"
BRAIN_DIR="Second Brain"
# Name of the vault auto-created when VAULT_PARENT is empty, so a fresh
# clone-and-run works with no vault at all. Keep in sync with
# vaults.DEFAULT_VAULT_NAME.
DEFAULT_VAULT_NAME="agent-vault"

die() {
    echo "obsidian-mcp: $1" >&2
    shift
    for line in "$@"; do
        echo "obsidian-mcp: $line" >&2
    done
    exit 1
}

list_candidates() {
    for entry in "$VAULT_PARENT"/*; do
        [ -d "$entry" ] || continue
        case "${entry##*/}" in
            .*) continue ;;
        esac
        echo "obsidian-mcp:   ${entry##*/}  ($entry)" >&2
    done
}

[ -d "$VAULT_PARENT" ] || die "'$VAULT_PARENT' is not a directory (is the volume mounted?)"

# An explicit name always wins, including when other vaults are present. A name
# that matches nothing is an error rather than a silent fallback, so a typo in
# .env is visible on the first boot instead of quietly serving the wrong vault.
if [ -n "$VAULT_SELECT" ]; then
    if [ -d "$VAULT_PARENT/$VAULT_SELECT" ]; then
        vault="$VAULT_PARENT/$VAULT_SELECT"
    elif [ -d "$VAULT_PARENT/$BRAIN_DIR" ] && [ "${VAULT_PARENT##*/}" = "$VAULT_SELECT" ]; then
        # Pointed straight at the vault it names; nothing to descend into.
        vault="$VAULT_PARENT"
    else
        list_candidates
        die "'$VAULT_SELECT' is not a directory under '$VAULT_PARENT'." \
            "Set OBSIDIAN_VAULT_NAME in .env to one of the vaults listed above."
    fi
# Empty parent: auto-create the default vault (and its Second Brain dir) so the
# server has a concrete vault to operate on. Mirrors vaults.select_vault.
elif [ -z "$(ls -A "$VAULT_PARENT")" ]; then
    vault="$VAULT_PARENT/$DEFAULT_VAULT_NAME"
    mkdir -p "$vault/$BRAIN_DIR"
# Already the vault itself, e.g. VAULT_PARENT points straight at it.
elif [ -d "$VAULT_PARENT/$BRAIN_DIR" ]; then
    vault="$VAULT_PARENT"
else
    # No glob expansion / subshell surprises: collect non-hidden child directories.
    count=0
    names=""
    for entry in "$VAULT_PARENT"/*; do
        [ -d "$entry" ] || continue
        case "${entry##*/}" in
            .*) continue ;;
        esac
        count=$((count + 1))
        if [ "$count" -eq 1 ]; then
            names="${entry##*/}"
        else
            names="$names, ${entry##*/}"
        fi
    done

    if [ "$count" -eq 0 ]; then
        die "no vault directory found in '$VAULT_PARENT'"
    fi
    if [ "$count" -gt 1 ]; then
        list_candidates
        die "'$VAULT_PARENT' holds $count vaults ($names); refusing to guess." \
            "Set OBSIDIAN_VAULT_NAME in .env to the one to use, then re-run."
    fi
    vault="$VAULT_PARENT/$names"
fi

echo "obsidian-mcp: using vault $vault"
exec obsidian-mcp --http --host 0.0.0.0 --port "${OBSIDIAN_MCP_PORT:-37842}" "$vault"
