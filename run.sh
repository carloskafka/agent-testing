#!/usr/bin/env bash
# One-command bootstrap: run the whole solution (agent web UI + obsidian-mcp).
#
#   ./run.sh
#   ./run.sh --vault work            # pick a vault without the menu
#   ./run.sh --no-vault-prompt       # fail instead of asking (CI, scripts)
#
# Creates a local .env from .env.example (if missing), picks which Obsidian vault
# to use, builds both containers, starts them, and prints the URLs.
#
# Vault selection is the interesting part. obsidian-mcp serves exactly one vault,
# so when the mounted parent holds several this prompts instead of guessing:
#
#   * no vault anywhere  -> auto-create "agent-vault" (fresh clone-and-run)
#   * one vault          -> use it, no configuration
#   * several            -> numbered menu, including vaults Obsidian already knows
#   * OBSIDIAN_VAULT_NAME / --vault -> use it, no prompt
#
# The choice is persisted to .env, so later runs (and `docker compose up` by hand)
# are non-interactive.
set -euo pipefail
cd "$(dirname "$0")"

PROMPT_FOR_VAULT=1
VAULT_OVERRIDE=""
# Keep in sync with vaults.DEFAULT_VAULT_NAME / resolve-vault.sh.
DEFAULT_VAULT_NAME="agent-vault"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-vault-prompt) PROMPT_FOR_VAULT=0 ;;
        --vault)
            [ $# -ge 2 ] || { echo "run.sh: --vault needs a name" >&2; exit 2; }
            VAULT_OVERRIDE="$2"
            shift
            ;;
        --vault=*) VAULT_OVERRIDE="${1#--vault=}" ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "run.sh: unknown option '$1' (try --help)" >&2
            exit 2
            ;;
    esac
    shift
done

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it and set your GEMINI_API_KEY"
    echo "(and optionally OpenRouter / Langfuse / Gmail keys) before continuing."
    echo
fi

# Read a key from the process environment first, then .env. Never `source .env`:
# it is user input and this script only needs two scalar values out of it.
env_get() {
    local key="$1" fallback="${2:-}"
    local from_env="${!key:-}"
    if [ -n "$from_env" ]; then
        printf '%s' "$from_env"
        return
    fi
    if [ -f .env ]; then
        local line
        line="$(grep -E "^[[:space:]]*${key}=" .env | tail -1 || true)"
        if [ -n "$line" ]; then
            line="${line#*=}"
            # Strip surrounding quotes and trailing whitespace/CR.
            line="${line%$'\r'}"
            line="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
            line="${line%\"}"; line="${line#\"}"
            line="${line%\'}"; line="${line#\'}"
            printf '%s' "$line"
            return
        fi
    fi
    printf '%s' "$fallback"
}

# Replace KEY's value in .env, or append the key when it is not there yet.
env_set() {
    local key="$1" value="$2"
    if grep -qE "^[[:space:]]*${key}=" .env; then
        # Use a temp file: sed -i on a bind-mounted or CRLF file is not portable.
        local tmp
        tmp="$(mktemp)"
        KEY="$key" VALUE="$value" awk '
            BEGIN { key = ENVIRON["KEY"]; value = ENVIRON["VALUE"] }
            $0 ~ "^[[:space:]]*" key "=" { print key "=" value; next }
            { print }
        ' .env > "$tmp"
        mv "$tmp" .env
    else
        printf '%s=%s\n' "$key" "$value" >> .env
    fi
}

# A plain interpreter for vault discovery. vaults.py imports only the stdlib, so
# any python3 will do; the project venv is preferred when it exists.
PYTHON=""
for candidate in text_summarizer/.venv/bin/python python3 python; do
    if command -v "$candidate" > /dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "run.sh: no python3 found; skipping vault discovery." >&2
    echo "run.sh: set OBSIDIAN_VAULT_NAME and OBSIDIAN_VAULT_PARENT_HOST in .env by hand." >&2
fi

VAULT_PARENT_HOST="$(env_get OBSIDIAN_VAULT_PARENT_HOST './vaults')"
VAULT_NAME="$(env_get OBSIDIAN_VAULT_NAME '')"
# An explicit --vault wins over the environment and over .env, and is persisted so
# the next bare `./run.sh` agrees with what the user just asked for.
if [ -n "$VAULT_OVERRIDE" ]; then
    VAULT_NAME="$VAULT_OVERRIDE"
fi

# --- vault discovery ----------------------------------------------------------
#
# Prints one candidate per line: "<name>\t<absolute path>\t<markers>". Sources:
# the children of the configured parent, plus every vault Obsidian already knows
# about (read from its own config), de-duplicated by path. The second source is
# what makes "I already have a lot of vaults" work without rearranging anything.

discover_vaults() {
    [ -n "$PYTHON" ] || return 0
    "$PYTHON" - "$VAULT_PARENT_HOST" <<'PY'
import os, sys

sys.path.insert(0, os.path.join(os.getcwd(), "text_summarizer"))
from vaults import discover_vaults, format_candidates, obsidian_vaults

parent = sys.argv[1]
parent = os.path.abspath(os.path.expanduser(parent))

seen, out = {}, []
for vault in discover_vaults(parent) + obsidian_vaults():
    real = os.path.realpath(vault.path)
    if real in seen:
        continue
    seen[real] = True
    out.append(vault)

if not out:
    sys.exit(0)
for vault in out:
    print(f"{vault.name}\t{vault.path}\t{','.join(vault.markers)}")
PY
}

# --- choosing -----------------------------------------------------------------

choose_vault() {
    local listing="$1" count="$2"

    if [ "$count" -eq 1 ]; then
        printf '%s\t\n' "$(printf '%s' "$listing" | head -1 | cut -f1)"
        return
    fi

    # A menu needs a terminal. Without one the caller gets a non-zero exit and an
    # error naming the candidates, which is the right outcome for CI: never a
    # guess, never a hang.
    if [ ! -t 0 ] || [ "$PROMPT_FOR_VAULT" -ne 1 ]; then
        echo "run.sh: $VAULT_PARENT_HOST holds $count vaults and there is no TTY to ask." >&2
        printf '%s\n' "$listing" | cut -f1,2 | sed 's/^/  /' >&2
        echo >&2
        echo "run.sh: set OBSIDIAN_VAULT_NAME in .env to pick one, e.g." >&2
        echo "         OBSIDIAN_VAULT_NAME=$(printf '%s' "$listing" | head -1 | cut -f1) ./run.sh" >&2
        echo >&2
        echo "run.sh: or pass --vault NAME, or --vault NAME to import it by symlink." >&2
        return 1
    fi

    # Everything the user reads goes to stderr. stdout carries only the return
    # value ("<name>\t<path-to-import>"), because the caller reads it through $( ),
    # which would otherwise swallow the menu into the vault name.
    local i=1 last name path markers mark answer source
    {
        echo
        echo "Several Obsidian vaults are available. Which one should the agent use?"
        echo
        while IFS="$(printf '\t')" read -r name path markers; do
            mark=""
            [ -n "$markers" ] && mark="  [$markers]"
            echo "  $i) $name$mark"
            echo "       $path"
            i=$((i + 1))
        done <<< "$listing"
        last=$((i))
        echo "  $i) Create a new empty vault ($DEFAULT_VAULT_NAME) in $VAULT_PARENT_HOST"
        i=$((i + 1))
        echo "  $i) Import a vault by path into $VAULT_PARENT_HOST (copied, original untouched)"
        i=$((i + 1))
        echo
        printf 'Choice [1-%d] (default 1): ' "$((i - 1))"
    } >&2

    read -r answer || true
    answer="${answer:-1}"
    if ! printf '%s' "$answer" | grep -qE '^[0-9]+$'; then
        echo "run.sh: not a number: '$answer'" >&2
        return 1
    fi
    # Valid choices are the listed vaults (1..last), "create new" (last+1) and
    # "import by path" (last+2) -- the same range the prompt advertises.
    if [ "$answer" -lt 1 ] || [ "$answer" -gt $((last + 2)) ]; then
        echo "run.sh: out of range: '$answer' (expected 1-$((last + 2)))" >&2
        return 1
    fi

    # "Create a new empty vault": name it, nothing to import.
    if [ "$answer" -eq "$last" ]; then
        printf '%s\t\n' "$DEFAULT_VAULT_NAME"
        return
    fi
    # "Import a vault by path": the path is typed, not chosen from the list, so this
    # works for a vault Obsidian does not know about. Both the name and the source
    # are returned, because a variable set inside $( ) does not survive the subshell.
    if [ "$answer" -eq $((last + 1)) ]; then
        {
            echo
            printf 'Absolute path of the vault to import: '
        } >&2
        read -r source || true
        source="${source#"${source%%[![:space:]]*}"}"
        source="${source%"${source##*[![:space:]]}"}"
        if [ -z "$source" ]; then
            echo "run.sh: no path given; keeping $VAULT_PARENT_HOST as the vault parent." >&2
            return 1
        fi
        if [ ! -d "$source" ]; then
            echo "run.sh: '$source' is not a directory." >&2
            return 1
        fi
        printf '%s\t%s\n' "$(basename "$source")" "$source"
        return
    fi
    printf '%s\t\n' "$(printf '%s' "$listing" | sed -n "${answer}p" | cut -f1)"
}

IMPORT_THIS=""

LISTING="$(discover_vaults || true)"
COUNT="$(printf '%s\n' "$LISTING" | grep -c . || true)"
COUNT="${COUNT:-0}"

# Absolute path of a candidate, by vault name.
path_of() {
    printf '%s' "$LISTING" | awk -F'\t' -v n="$1" '$1 == n { print $2; exit }'
}

# Point the mount at the chosen vault's own parent. Preferred over an import: no
# indirection, and the real name stays in the path, so the **Sources** lines resolve
# it with nothing configured. A no-op when the vault already sits in the mount.
repoint_parent_to() {
    local selected="$1"
    [ -n "$selected" ] || return 0
    local new_parent current
    new_parent="$(cd "$(dirname "$selected")" && pwd)"
    if [ -d "$VAULT_PARENT_HOST" ]; then
        current="$(cd "$VAULT_PARENT_HOST" && pwd)"
    else
        current=""
    fi
    if [ "$new_parent" != "$current" ]; then
        VAULT_PARENT_HOST="$new_parent"
        echo "Vault parent: $VAULT_PARENT_HOST"
    fi
}

if [ -n "$VAULT_NAME" ]; then
    SELECTED_PATH="$(path_of "$VAULT_NAME")"
    if [ -z "$SELECTED_PATH" ] && [ ! -d "$VAULT_PARENT_HOST/$VAULT_NAME" ]; then
        echo "run.sh: no vault named '$VAULT_NAME'." >&2
        if [ "$COUNT" -gt 0 ]; then
            echo "run.sh: available vaults:" >&2
            printf '%s' "$LISTING" | cut -f1,2 | sed 's/^/  /' >&2
        fi
        echo >&2
        echo "run.sh: run ./run.sh without --vault to pick from a menu." >&2
        exit 1
    fi
    [ -n "$SELECTED_PATH" ] && repoint_parent_to "$SELECTED_PATH"
    [ -n "$VAULT_OVERRIDE" ] && env_set OBSIDIAN_VAULT_NAME "$VAULT_NAME"
    echo "Vault: $VAULT_NAME"
elif [ "$COUNT" -eq 0 ]; then
    # Nothing to choose from: scaffold the parent and let the auto-create
    # default take over on first boot.
    case "$VAULT_PARENT_HOST" in
        /*) ;;
        *) mkdir -p "$VAULT_PARENT_HOST" ;;
    esac
    echo "No Obsidian vault found — will auto-create '$DEFAULT_VAULT_NAME' in $VAULT_PARENT_HOST"
else
    CHOSEN=""
    IMPORT_THIS=""
    IFS="$(printf '\t')" read -r CHOSEN IMPORT_THIS <<< "$(choose_vault "$LISTING" "$COUNT")" || exit 1
    [ -n "$CHOSEN" ] || exit 1
    SELECTED_PATH="$(path_of "$CHOSEN")"

    if [ -n "$IMPORT_THIS" ]; then
        # Keep the parent where it is and bring the vault into it. Copy, not
        # symlink: a link whose target is outside the bind mount dangles inside
        # the container, which stops obsidian-mcp at startup. See
        # vaults.import_vault, which copies for the same reason.
        if [ ! -e "$VAULT_PARENT_HOST/$CHOSEN" ]; then
            mkdir -p "$VAULT_PARENT_HOST"
            cp -r "$IMPORT_THIS" "$VAULT_PARENT_HOST/$CHOSEN"
            echo "Imported $CHOSEN from $IMPORT_THIS (copied; the original is untouched)"
        else
            echo "$VAULT_PARENT_HOST/$CHOSEN already exists; using it as is"
        fi
    else
        repoint_parent_to "$SELECTED_PATH"
    fi

    env_set OBSIDIAN_VAULT_NAME "$CHOSEN"
    VAULT_NAME="$CHOSEN"
    echo "Vault: $VAULT_NAME"
fi

env_set OBSIDIAN_VAULT_PARENT_HOST "$VAULT_PARENT_HOST"

docker compose up -d --build

echo
echo "Agent web UI  : http://localhost:8001"
echo "Obsidian vault: ${VAULT_NAME:-<the only vault in>} ${VAULT_PARENT_HOST}"
echo "Onboarding guide: docs/getting-started.md"
