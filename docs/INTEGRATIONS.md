# Integrations

The agent can read your Gmail inbox and use an Obsidian vault as a second brain.
Both are optional and controlled purely by environment variables — with none of
them set the agent still works as a plain text summarizer.

## Gmail (Read-Only)

The agent gets four Gmail tools ([see integration code](../text_summarizer/gmail_tools.py)):

| Tool | Purpose |
|---|---|
| `gmail_get_latest_messages` | The N most recent inbox messages |
| `gmail_search` | Search with a Gmail query (e.g. `from:jobs subject:offer is:unread`) |
| `gmail_read` | Full email (headers + body) by message id |
| `gmail_get_thread` | Every message in a thread, oldest first |

It runs through a local MCP server (`gmail_mcp_server.py`) that the agent spawns
over stdio. Only read access (`gmail.readonly` scope) is used — nothing is ever
sent or deleted.

### Setup (one time)

1. **Create OAuth credentials** in the [Google Cloud Console](https://console.cloud.google.com/apis/credentials):
   - Enable the **Gmail API** (APIs & Services → Library → Gmail API → Enable)
   - **Create Credentials → OAuth client ID → type "Desktop app"** → download `client_secret.json`
   - If the app is in Testing mode on the consent screen, add your Gmail address as a **Test user**

2. **Mint a refresh token** from inside the repo:

   ```bash
   cd text_summarizer
   uv run python -m text_summarizer.gmail_oauth --console
   ```

   `--console` prints a URL — open it in any browser (handy when the browser can't
   reach this machine's localhost, e.g. a VM), approve read-only access, and paste
   the authorization code back into the terminal.

3. **Put the three printed values into your `.env`:**

   ```env
   GOOGLE_CLIENT_ID=...
   GOOGLE_CLIENT_SECRET=...
   GOOGLE_REFRESH_TOKEN=...
   ```

The tools load automatically as soon as all three are set. If the token is ever
revoked or expired, just re-run step 2 and replace `GOOGLE_REFRESH_TOKEN`.

### Troubleshooting

- **`ModuleNotFoundError: dotenv`** — run the command from inside `text_summarizer/`
  (or `uv run --project text_summarizer ...`), not the repo root.
- **Browser says connection refused after approving** — that's Google's localhost
  redirect racing a VM's browser; ignore it and copy the **authorization code**
  instead, then paste it at the terminal prompt. Don't paste the whole URL.
- **`Scope has changed from ... to ...`** — Google echoes previously-granted
  scopes for the client (Drive/Calendar); the helper now requests only
  `gmail.readonly` and ignores extras, but re-run if you saw an old version.

## Obsidian Vault ("Second Brain")

The agent persists summaries, chat logs, and an index graph into an Obsidian
vault, and retrieves related notes before summarizing.

### Local (stdio)

```env
OBSIDIAN_VAULT_PATH=/absolute/path/to/your/vault
```

### Docker (streamable HTTP)

```env
OBSIDIAN_MCP_URL=http://127.0.0.1:37842/mcp
```

Set **exactly one** of the two. Under Docker the vault's **parent** is mounted
into both containers at `/vaults` so the real vault directory name survives the
mount; `SECOND_BRAIN_VAULT=/vaults` points the write path at the same place.
The host path comes from `OBSIDIAN_VAULT_PARENT_HOST` in `.env` (default
`./vaults`), and if the mount is empty the vault **`agent-vault`** is
auto-created on first boot — so a fresh clone-and-run works with no vault setup
at all. See [Docker deployment](ARCHITECTURE.md#docker-topology).

### Using a vault you already have

`obsidian-mcp` serves exactly one vault, so with several on disk the choice is
explicit and never guessed. `./run.sh` handles it: it discovers the children of
`OBSIDIAN_VAULT_PARENT_HOST` **and** every vault listed in your Obsidian config
(`~/.config/obsidian/obsidian.json`), shows them as a numbered menu, and writes
your answer back to `.env`.

```
$ ./run.sh

Several Obsidian vaults are available. Which one should the agent use?

  1) ck  [.obsidian,Second Brain]
       /home/you/Documents/Obsidian/ck
  2) journal  [.obsidian]
       /home/you/Documents/Journal
  3) Create a new empty vault (agent-vault) in ./vaults
  4) Import a vault by path into ./vaults (copied, original untouched)

Choice [1-4] (default 1): 1
Vault: ck
Vault parent: /home/you/Documents/Obsidian
```

Picking a vault that already lives elsewhere **repoints the mount** at that
vault's own parent and sets `OBSIDIAN_VAULT_NAME` — no copying, no symlink, and
the real name stays in the path so provenance renders it with nothing hardcoded.
That is the recommended route for an existing vault.

Prefer to keep the mount where it is? Choose **import**: it copies the vault
into `OBSIDIAN_VAULT_PARENT_HOST`, leaving your original untouched. It copies
rather than symlinks on purpose — a symlink pointing outside the bind mount
dangles inside the container, where that path does not exist, and it stops
`obsidian-mcp` at startup. Nothing that already exists is ever overwritten.

Non-interactive equivalents: `OBSIDIAN_VAULT_NAME` in `.env`, or `./run.sh
--vault <name>`. If the parent holds several vaults and no name is set, both the
agent and the MCP server refuse and print the candidate names rather than pick
one:

```
obsidian-mcp:   personal  (/vaults/personal)
obsidian-mcp:   work  (/vaults/work)
obsidian-mcp: '/vaults' holds 2 vaults (personal, work); refusing to guess.
obsidian-mcp: Set OBSIDIAN_VAULT_NAME in .env to the one to use, then re-run.
```

A single vault needs no configuration at all, which is why this only shows up
once a second vault exists.
