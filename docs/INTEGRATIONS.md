# Integrations

The agent can read your Gmail inbox, use an Obsidian vault as a second brain, and —
when the vault has nothing — search the web through a self-hosted SearXNG. All three
are optional and controlled purely by environment variables: with none of them set the
agent still works as a plain text summarizer.

The three tiers are consulted in order — **vault → web → the model's own knowledge** —
and the agent is told to reach for a later one only after the earlier ones come up
empty.

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

## Web Search (self-hosted SearXNG)

The third tier. Two first-party tools over a plain HTTP API — no MCP server, no
subprocess:

| Tool | Purpose |
|---|---|
| `web_search` | Query SearXNG; returns titles, URLs and short snippets |
| `web_fetch` | Fetch and sanitise one page's text |

```env
# The host must be the SERVICE NAME of your SearXNG container on agent-net --
# check `docker network inspect agent-net`. This repo's own SearXNG is
# `searxng-core`; a wrong name does not fail loudly, it just stops being a tier.
SEARXNG_URL=http://searxng-core:8080
```

Unset means **no web tools at all** and the agent is unchanged — a summarizer that
cannot search does not merely get worse at searching, it stops offering to.
`WEB_SEARCH_ENABLED=false` removes the tools while leaving `SEARXNG_URL` alone, which
is the control to use when measuring.

### When the vault is not allowed to end the question

The tier is a *last resort* by design: rule 7 searches the vault first, and a note
on the topic normally settles the matter. Two things override that, because a note
is a record of an earlier fetch and it does not carry what the reader came for:

- **Exhaustive requests** — "all movies", "each one", "todos os filmes". A note
  that summarises a listing is itself an index, not the answer; each item usually
  has its own page.
- **Live facts** — today's session times, prices, opening hours, stock. These change
  within the day, so a note written this morning is a wrong answer rather than a
  cached one.

Found live: asked *"todos os filmes disponíveis pra hoje à noite"*, the agent found a
vault note titled almost exactly that, stopped, and answered from it with **zero** web
calls — and therefore not one clickable link in the answer. See
[the web tier](ARCHITECTURE.md#the-web-tier).

When a fact does come from a page, the URL travels with it: on the fact itself as a
markdown link, and into the saved note. A note that records times but no URLs
cannot answer the same question from the vault tomorrow, which is how a follow-up
turn ends up citing a page nobody can click.

Under Docker, SearXNG runs as a **separate compose project** reached over a shared
external docker network (`agent-net`) by service name rather than a published port, so
nothing is exposed on the LAN for the agent's sake. `run.sh` creates that network
idempotently, and only when `SEARXNG_URL` is set.

### Booking links come with the time they belong to

When a page publishes screenings as schema.org `ScreeningEvent` records — cinemas, airlines and event listings all do — `web_fetch` lists each one with **its time, its venue and its booking URL** on one line:

```
[sessions on this page]
- 21:15 Kinoplex Osasco  https://checkout.ingresso.com/?sessionId=87210684&partnership=home
- 15:20 Cinemark Osasco  https://checkout.ingresso.com/?sessionId=87205315&partnership=home
```

This is not a nicety. Without it the page's own anchors reach the model as bare session IDs — `?sessionId=87210684&partnership=home` — with no way to tell which is which, and the model cites the page it fetched instead of the session you asked about. Found live: asked for one screening, given the film's page URL.

Two details worth knowing:

- **A record missing any of the three fields is dropped.** Half a record is the defect rather than the fix, so an offer with no venue, or a venue with no `startDate`, does not appear at all.
- **The date is stripped from the line.** Printing `2026-10-03T21:15` beside every session invites reading a timestamp as a time of day.

Not every venue publishes it. A listing page may carry no sessions at all, in which case the block is absent and per-film pages are the route to a booking link.

### Pages behind a gate, and where a purchase stops

Some pages hide their content behind a button: an 18+ rating, a region lock, an
"are you over 18?" modal. `web_fetch` recognises that and returns a refusal naming
the gate rather than a page of nothing. The agent then calls `ask_user`, and only
if **you actually answered** may it re-fetch with `attested=true`.

That flag is **checked, not trusted**. It is refused unless `ask_user` recorded a
real answer in the session, because a boolean the model supplies is a boolean the
model supplies — a model that reads the refusal learns exactly what the flag does.
Only the branch where a human picked an option records it; declined, unrecognised
and "no turn context" all leave the default, and all three are the agent choosing.

Clearing the gate is a real click, so it runs in the **renderer**, and the renderer
treats it as the one write it has:

- **A dismiss must not navigate.** The URL is compared before and after; a click
  that moves the page is reported as an error rather than as success.
- **A dismiss is not a form submit.** An element inside a `<form>` is refused before
  anything is clicked.

Those two properties are what make it general-purpose rather than Ingresso-specific,
and they are what a purchase cannot get past: *Accept all cookies* does not
navigate, *Buy ticket* does. No allow-listed phrase reaches a checkout.

Which pages it may act on, and which labels count as a dismissal, are operator
lists in the **renderer's** environment rather than the agent's — the renderer is
the process that executes a stranger's JavaScript, so that choice is not the
model's to make. Both blank means the primitive is **off**.

```env
RENDERER_DISMISS_HOSTS=ingresso.com
RENDERER_DISMISS_TEXT=I'm aware,I agree,I accept,Accept all,Got it,Continue,Allow all
```

**Where the flow stops is a purchase, and it stops at the checkout.** The agent
reads the page — session, room, seat rows, price — and hands you the exact URL to
act on. It does not buy, pay, reserve or complete anything, and there is no tool
that would let it. A payment link you are given is a payment you make yourself.

This is not a missing feature. Ingresso.com requires an account to buy, and the
renderer holds no credentials by design: it is the container with no vault, no API
key and no access to anything else in the deployment. It is also the only thing
here that would spend your money on a page written by a stranger.

### Retrieved pages are treated as untrusted

Anyone can publish a page that ranks for a query, and its text reaches the model
verbatim. Three defences, all required — see
[the web tier](ARCHITECTURE.md#the-web-tier) for why each one exists and what it
costs to remove:

- the HTTP client dials the **address that was vetted**, not the hostname, so a host
  cannot be re-pointed at an internal address between check and connect (DNS
  rebinding);
- retrieved text is wrapped in an `<untrusted_content>` marker that a page **cannot
  close from inside**, including via an HTML-entity-encoded closing tag;
- snippets are framed too — they are the page author's own meta description, and the
  most poisonable input the agent sees.

A failed search returns an error **as data**, so the model can fall through to the
vault or its own knowledge rather than the turn dying.

