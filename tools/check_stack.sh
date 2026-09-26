#!/bin/sh
# Assert the stack is in a state where a recording would be truthful.
#
# The two traps this exists to catch, both of which produce a *plausible* capture
# of a broken run rather than an error:
#
#   1. obsidian-mcp runs with `network_mode: service:agent-testing`, so it shares
#      the agent container's network namespace. Rebuilding or recreating only the
#      agent (`docker compose up -d --build agent-testing`) leaves the sidecar
#      attached to the old namespace: it stays "Up", its log still says
#      "listening", and every obsidian-mcp call is refused. The agent then answers
#      "no notes found" -- which looks like a retrieval bug and is not one.
#      Always recreate both services together.
#   2. A vault that resolves to a different directory than the MCP server's, so
#      writes and reads land in different places.
#
# Usage: sh tools/check_stack.sh   (exit 0 = good to record)

set -eu
cd "$(dirname "$0")/.."

fail() { echo "FAIL: $1" >&2; exit 1; }

echo "== containers"
docker compose ps --format '{{.Name}}\t{{.Status}}' | sed 's/^/  /'
up=$(docker compose ps --status running -q | wc -l)
[ "$up" -ge 2 ] || fail "expected both services running, found $up"

echo "== dev UI"
code=$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 10 http://localhost:8001/dev-ui/)
[ "$code" = "200" ] || fail "dev UI returned $code on :8001"
echo "  :8001/dev-ui/ -> $code"

echo "== obsidian-mcp reachable from the agent's network namespace"
docker compose exec -T agent-testing python - <<'PY' || fail "obsidian-mcp not reachable in the agent's netns"
import socket, sys
s = socket.socket()
sys.exit(0 if s.connect_ex(("127.0.0.1", 37842)) == 0 else 1)
PY
echo "  127.0.0.1:37842 -> open"

echo "== vault both sides agree on"
mcp=$(docker compose logs obsidian-mcp 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -o 'watcher started.*' | tail -1 | sed 's/.*=//')
# Run it as the package, from the workspace root: second_brain uses relative
# imports, so importing it as a top-level module fails.
agent=$(docker compose exec -T agent-testing sh -lc 'cd /workspace && text_summarizer/.venv/bin/python -c "
from text_summarizer.second_brain import VAULT_ROOT
print(VAULT_ROOT)
"' 2>/dev/null | tail -1)
mcp=$(printf '%s' "$mcp" | tr -d '\000' | sed 's/[[:space:]]*$//')
echo "  mcp:   ${mcp:-unknown}"
echo "  agent: ${agent:-unknown}"
[ -n "$mcp" ] && [ -n "$agent" ] || fail "could not resolve a vault path on one side"
[ "$mcp" = "$agent" ] || fail "the two sides disagree: mcp=$mcp agent=$agent"

echo "OK: safe to record"
