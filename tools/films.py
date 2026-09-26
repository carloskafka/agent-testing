#!/usr/bin/env python3
"""The three walkthrough films, composed from real recordings.

Kept apart from build_gifs.py so the frame plumbing (crops, palette, encoding)
and the story (which real capture goes in which scene) can be read separately.

Every scene is a 1:1 crop of a screenshot of the real dev UI mid-turn, or a panel
built from files read out of the live vault. Nothing is invented: a panel that
quotes a note is the note, and the latencies in a tag are the capture's own
timestamps. The only drawn parts are the labels and the panels that hold real
values -- there is no file manager or Obsidian on this host to photograph, so
those are rendered and said to be rendered.

Source geometry, measured from the DOM of a 1400x814 capture (not guessed):

    .chat-messages            x 480..1388   y  97..737
    .message-row-container    x 480..1388   y 117..289
    composer textarea         x 600..1176   y 747..804
    event rows                y 550..737
"""

from __future__ import annotations

import hashlib
import re

from build_gifs import (AMBER, BAR, BLUE, BOTTOM, CORAL, FAINT, F, GREEN, MUTED, OUT_H,
                        OUT_W, PANEL, PANEL2, STROKE, TEXT, VIOLET, Frame, hold, play)

CHAT = (480, 0, 1388, 814)          # the chat column of a capture
CERN = "2026-09-26 - cern-discovery-of-xi-cc-double-plus-baryon"
MODEL = "gemini-3.5-flash-lite"


def fingerprint(text: str) -> str:
    """The real second_brain.source_fingerprint of a prompt."""
    norm = " ".join((text or "").strip().lower().split())
    for prefix in ("summarize: ", "summarize the following: ", "please summarize: ",
                   "can you summarize: ", "summarize "):
        if norm.startswith(prefix):
            norm = norm[len(prefix):]
            break
    return hashlib.sha256(norm.encode()).hexdigest()


def clip(s: str, n: int) -> str:
    """Truncate on a word boundary, so a quoted line never ends mid-word."""
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut + "…"


def strip_lines(body: str) -> tuple[list[str], str]:
    """Frontmatter + bullets, stopping before the wiki-link section."""
    out, tail = [], ""
    for ln in body.split("\n"):
        if ln.startswith("## Related"):
            tail = ln
            break
        out.append(ln)
    return out, tail


# =============================================================================
# Film 1 -- the main flow
# =============================================================================

def main_film(caps, vault):
    live, ask, repeat = caps.get("live"), caps.get("ask"), caps.get("repeat")
    out = []

    # 1 -- the prompt, as the app received it
    if live:
        f = Frame("1  the prompt", BLUE, "the real composer, then sent")
        f.region(live.at(1), (480, 150, 1388, 300))
        y = BAR + 172
        f.rrect((14, y, OUT_W - 14, y + 150), r=10, fill=PANEL, outline=STROKE, w=1)
        f.text((30, y + 16), "the turn that follows it", F(11, bold=True), TEXT, anchor="lt")
        fp = fingerprint(live.prompt)
        rows = [
            ("agent", "text_summarizer", TEXT),
            ("vault", "ck", GREEN),
            ("model", MODEL, VIOLET),
            ("fingerprint of the prompt", fp[:12] + "…" + fp[-8:], AMBER),
        ]
        for i, (k, v, col) in enumerate(rows):
            xx = 30 + (i % 2) * 450
            yy = y + 46 + (i // 2) * 48
            f.text((xx, yy), k, F(9.5, mono=True), FAINT, anchor="lt")
            f.text((xx, yy + 15), v, F(11, mono=True), col, anchor="lt")
        f.caption(14, y + 168, OUT_W - 28,
                  "The fingerprint is computed in code from the user's own message, before "
                  "any model call. Nothing is asked of the model about the input it was "
                  "given, which is what makes the last scene free.")
        out += play([f.done()], 2200)

    # 2 -- the turn running: the recorded event stream
    if live:
        seq = []
        for i in live.distinct(1.6)[1:]:
            g = Frame("2  the turn runs", BLUE, f"t+{live.times[i]:.1f}s  ·  recorded")
            g.shot(live.at(i), y0=BOTTOM)
            seq.append(g.done())
        out += play(seq, 190) + hold(seq, 500)

    # 3 -- the answer
    if live:
        f = Frame("3  the answer", GREEN, f"{len(live)} frames  ·  "
                                          f"{live.times[-1] - live.times[2]:.1f}s of work")
        f.shot(live.last(), y0=BOTTOM)
        f.chip(24, OUT_H - 32, "rule 1-5: bullets, one sentence each, nothing added",
               GREEN, bg=(18, 58, 42), f=F(9.5, mono=True))
        f.chip(24 + 330, OUT_H - 32, "rule 8: saved  ·  rule 9: logged", MUTED,
               bg=PANEL2, border=STROKE, f=F(9.5, mono=True))
        out += play([f.done()], 2400)

    # 4 -- provenance: what the model wrote, and what the reader gets
    if ask:
        f = Frame("4  the Sources block is written in code", VIOLET,
                  "both lines are from the recorded response")
        f.region(ask.last(), (480, 292, 1388, 366), dst=(14, BAR + 14))
        f.text((14, BAR + 90), "the model's own event #13 — sentinels it was told to copy, "
               "verbatim", F(9.5), FAINT, anchor="lt")
        f.region(ask.last(), (480, 604, 1388, 692), dst=(14, BAR + 130))
        f.text((14, BAR + 236), "the response the reader gets — the same text with the real "
               "identifiers", F(9.5), FAINT, anchor="lt")
        yy = BAR + 268
        f.rrect((14, yy, OUT_W - 14, OUT_H - 14), r=10, fill=PANEL, outline=STROKE, w=1)
        f.text((30, yy + 14), "after_agent_callback  →  sources.render_sources", F(10.5,
                 bold=True, mono=True), BLUE, anchor="lt")
        f.caption(30, yy + 36, 420,
                  "Both @@ADK_VAULT@@ and @@ADK_MODEL@@ are replaced in code. The vault comes "
                  "from the resolved vault path, the model from Event.model_version of the "
                  "call that actually answered, and an unresolvable one renders as the "
                  "literal unknown rather than a guess.")
        f.caption(486, yy + 36, 400,
                  "The model is never asked for either identifier: it has no way to know "
                  "them, and a paraphrase would be a different string on every run. One "
                  "canonical block is emitted whatever heading the model wrote, so a "
                  "response cannot end up with two.")
        out += play([f.done()], 3000)

    # 5 -- the note that landed on disk, read back out of the vault
    if vault:
        title = next((n for n in vault.notes if "cern" in n.lower()), vault.notes[-1])
        body = vault.read(title)
        lines, _ = strip_lines(body)
        f = Frame("5  on disk, in the vault", GREEN, f"{title}.md")
        y = f.card((14, BAR + 12, 566, OUT_H - 14), f"Second Brain/{title}.md", GREEN)
        step = (OUT_H - 26 - y) / max(1, len(lines))
        for i, ln in enumerate(lines):
            if ln.strip() == "---":
                col = BLUE
            elif ln.startswith(("generated_", "source_", "date:", "aliases:", "tags:")):
                col = VIOLET
            elif ln.startswith("  - "):
                col = MUTED
            elif ln.startswith("- "):
                col = TEXT
            else:
                col = FAINT
            f.text((30, y + i * step), ln, F(9.5, mono=True), col, anchor="lt")
        ry = f.card((582, BAR + 12, OUT_W - 14, OUT_H - 14), "written by the tool", GREEN)
        prov = {}
        for ln in lines:
            if ":" in ln and not ln.startswith((" ", "-")):
                k, _, v = ln.partition(":")
                prov[k.strip()] = v.strip()
        rows = [
            ("date", prov.get("date", ""), TEXT),
            ("source_fingerprint", prov.get("source_fingerprint", "")[:10] + "…"
             + prov.get("source_fingerprint", "")[-6:], AMBER),
            ("generated_by_model", prov.get("generated_by_model", "").strip('"'), VIOLET),
            ("generated_in_vault", prov.get("generated_in_vault", "").strip('"'), GREEN),
        ]
        yy = ry
        for k, v, col in rows:
            f.text((598, yy), k, F(9, mono=True), FAINT, anchor="lt")
            f.text((598, yy + 14), v, F(10, mono=True), col, anchor="lt")
            yy += 40
        f.rect((598, yy + 2, OUT_W - 30, yy + 3), fill=STROKE)
        f.caption(598, yy + 12, OUT_W - 30 - 598,
                  "Read from the live invocation, never asked of the model, and omitted "
                  "rather than guessed when unresolvable.")
        out += play([f.done()], 2600)

    # 6 -- the same question again
    if repeat and live:
        f = Frame("6  ask it again", AMBER, f"same prompt, new session  ·  "
                                           f"{repeat.times[2] - repeat.times[1]:.1f}s")
        f.shot(repeat.last(), y0=BOTTOM)
        y = BAR + 380
        f.rrect((14, y, OUT_W - 14, OUT_H - 14), r=10, fill=PANEL, outline=STROKE, w=1)
        x = 30
        for label, col, bg in (("2 events", GREEN, (18, 58, 42)),
                               ("0 model calls", GREEN, (18, 58, 42)),
                               ("cache.hit 1.0", AMBER, (63, 49, 15)),
                               ("no quality scores", MUTED, PANEL2)):
            x += f.chip(x, y + 16, label, col, bg=bg, f=F(9.5, mono=True)) + 8
        f.caption(30, y + 52, OUT_W - 60,
                  f"The fingerprint is already on disk, so before the first model call the "
                  f"stored note replays. The first ask above took "
                  f"{live.times[-1] - live.times[2]:.0f}s and ran real tools; this one cost "
                  f"nothing but the lookup.")
        out += play([f.done()], 2800)

    return dict(frames=out, still=-1)


# =============================================================================
# Film 2 -- the Obsidian graph and look-up
# =============================================================================

def obsidian_film(caps, vault):
    ask = caps.get("ask")
    out = []

    # 1 -- the real vault
    if vault:
        f = Frame("1  one vault, real files", GREEN,
                  f"{len(vault.notes)} notes  ·  {len(vault.topics)} topic stubs")
        y = f.card((14, BAR + 12, 452, OUT_H - 14), "the live vault, as it is on disk", GREEN)
        f.text((30, y), "Second Brain/", F(10, mono=True), TEXT, anchor="lt")
        recent = vault.notes[-8:]
        for i, n in enumerate(recent):
            f.text((44, y + 20 + i * 19), f"├─ {n[:34]}.md", F(9, mono=True), MUTED, anchor="lt")
        ty = y + 20 + len(recent) * 19 + 10
        f.rect((30, ty - 6, 436, ty - 5), fill=STROKE)
        f.text((30, ty), "Topics/", F(10, mono=True), TEXT, anchor="lt")
        for i, t in enumerate(vault.topics[-3:]):
            f.text((44, ty + 20 + i * 19), f"├─ {t[:34]}.md", F(9, mono=True), MUTED, anchor="lt")
        f.text((30, ty + 20 + 3 * 19 + 6), f"└─ … {len(vault.topics)} stubs, one index, "
                 f"one chat log", F(9, mono=True), FAINT, anchor="lt")
        f.caption(30, ty + 130, 410,
                  "obsidian-mcp serves exactly one vault, so the choice is made once and "
                  "never guessed. The parent directory is what gets mounted, which is why "
                  "the real name survives in the path and lands in the note's frontmatter.")

        ry = f.card((468, BAR + 12, OUT_W - 14, OUT_H - 14), "how the vault is chosen", GREEN)
        for i, (k, v, col) in enumerate([
            ("OBSIDIAN_VAULT_NAME set", "that vault wins", GREEN),
            ("parent is itself a vault", "used as is", MUTED),
            ("exactly one child", "that child", MUTED),
            ("no children", "agent-vault, created lazily", MUTED),
            ("several, no name", "refuses and lists them", CORAL),
        ]):
            yy = ry + i * 38
            f.text((484, yy), k, F(10, mono=True), TEXT, anchor="lt")
            f.text((484, yy + 15), v, F(9.5), col, anchor="lt")
        f.rect((484, ry + 194, OUT_W - 30, ry + 195), fill=STROKE)
        f.caption(484, ry + 206, OUT_W - 30 - 484,
                  "The same rule runs in the agent and in the MCP server's entrypoint, and "
                  "the two must never disagree — the server exits rather than serve a vault "
                  "nobody chose, while the agent degrades to unknown provenance instead of "
                  "not answering.")
        out += play([f.done()], 2600)

    # 2 -- retrieve before answering
    if ask:
        seq = []
        for i in ask.distinct(1.2)[1:]:
            g = Frame("2  rule 7: retrieve first", VIOLET,
                      f"t+{ask.times[i]:.1f}s  ·  recorded")
            g.shot(ask.at(i), y0=BOTTOM)
            seq.append(g.done())
        out += play(seq, 700)

    # 3 -- what the vault actually returned
    if vault:
        f = Frame("3  the notes it found", VIOLET, "read out of the live vault")
        found = [n for n in vault.notes if "photosynthesis" in n.lower()] or vault.notes[-2:]
        y = BAR + 14
        for n in found[:3]:
            body = vault.read(n)
            bullets = [ln for ln in body.split("\n") if ln.startswith("- ")]
            links = re.findall(r"\[\[([^\]]+)\]\]", body)
            f.rrect((14, y, OUT_W - 14, y + 128), r=10, fill=PANEL, outline=STROKE, w=1)
            f.text((30, y + 14), f"{n}.md", F(10.5, mono=True), TEXT, anchor="lt")
            for j, b in enumerate(bullets[:2]):
                f.caption(30, y + 36 + j * 18, 840, clip(b.strip("- "), 118), F(9.5, mono=True),
                          fill=MUTED, lh=1.4)
            x = 30
            for t in links[:5]:
                x += f.chip(x, y + 84, f"[[{t}]]", GREEN, bg=(18, 58, 42),
                            f=F(9, mono=True), h=18) + 6
            y += 140
        f.caption(14, y + 4, OUT_W - 28,
                  "Titles, bodies and wiki links are the files themselves, read from disk when "
                  "this film was built. The model read the promising ones rather than the "
                  "whole vault, and the hub note, before it wrote a word of summary.")
        out += play([f.done()], 2600)

    # 4 -- the citations, in the recorded response
    if ask:
        f = Frame("4  the citations", VIOLET, "real response, real vault name")
        f.region(ask.last(), (480, 604, 1388, 692), dst=(14, BAR + 14))
        y = BAR + 110
        f.rrect((14, y, OUT_W - 14, OUT_H - 14), r=10, fill=PANEL, outline=STROKE, w=1)
        f.text((30, y + 14), "[obsidian][ck][" + MODEL + "]", F(11, bold=True, mono=True),
               GREEN, anchor="lt")
        f.caption(30, y + 40, 430,
                  "One canonical block, at the very end, one line per note that was actually "
                  "relevant. If nothing relevant exists the block is omitted entirely rather "
                  "than padded.")
        f.caption(486, y + 40, 400,
                  "A vault that renders as unknown is a visible failure, not a silent one: it "
                  "means retrieval is not wired up. Note that the citations name notes that "
                  "exist — the fingerprint of this prompt is on the note it wrote.")
        out += play([f.done()], 2800)

    # 5 -- what one save writes
    if vault:
        title = next((n for n in vault.notes if "photosynthesis" in n.lower()), vault.notes[-1])
        f = Frame("5  one save, four writes", GREEN, f"the turn that made {title[:26]}")
        y = BAR + 16
        for num, what, where, col in [
            ("1", "one dated note", f"Second Brain/{title}.md", GREEN),
            ("2", "a stub per topic", f"Topics/ — {len(vault.links(title))} linked topics",
             GREEN),
            ("3", "one line in the hub", "Second Brain Index.md", AMBER),
            ("4", "one chat-log entry", "Chat Log/2026-09-26.md", VIOLET),
        ]:
            f.rrect((24, y, OUT_W - 24, y + 84), r=10, fill=PANEL, outline=STROKE, w=1)
            f.ellipse((44, y + 24, 76, y + 56), fill=PANEL2)
            f.text((60, y + 40), num, F(13, bold=True), col, anchor="mm")
            f.text((94, y + 18), what, F(12, bold=True), TEXT, anchor="lt")
            f.text((94, y + 40), where, F(9.5, mono=True), col, anchor="lt")
            if num != "4":
                f.line((60, y + 84, 60, y + 98), STROKE, 1)
            y += 98
        f.caption(24, y + 2, OUT_W - 48,
                  "So the vault stays a graph rather than a folder of loose files: every note is "
                  "linked to its topics and to the hub on its way past, and nothing else has to "
                  "be maintained.")
        out += play([f.done()], 2600)

    # 6 -- the graph, from the real links
    if vault:
        f = Frame("6  the graph it grows", BLUE,
                  f"{len(vault.notes)} notes  ·  {len(vault.topics)} topics  ·  real links")
        notes = vault.notes[-6:]
        topics = sorted({t for n in notes for t in vault.links(n)})[:8]
        hub = (OUT_W / 2, BAR + 40)
        npos, tpos = {}, {}
        for i, n in enumerate(notes):
            npos[n] = (110 + (i % 3) * 320, BAR + 150 + (i // 3) * 130)
        for i, t in enumerate(topics):
            tpos[t] = (60 + (i % 4) * 210, BAR + 380 + (i // 4) * 80)
        for n, (nx, ny) in npos.items():
            for t in vault.links(n):
                if t in tpos:
                    f.line((nx, ny, tpos[t][0], tpos[t][1]), STROKE, 1)
        for n, (nx, ny) in npos.items():
            f.line((hub[0], hub[1], nx, ny), AMBER, 1)
        f.ellipse((hub[0] - 10, hub[1] - 10, hub[0] + 10, hub[1] + 10), fill=AMBER)
        f.text((hub[0] + 18, hub[1]), "Second Brain Index", F(10, bold=True, mono=True),
               TEXT, anchor="lm")
        for n, (nx, ny) in npos.items():
            f.ellipse((nx - 7, ny - 7, nx + 7, ny + 7), fill=BLUE)
            f.text((nx + 12, ny), n[:30], F(8.5, mono=True), MUTED, anchor="lm")
        for t, (tx, ty) in tpos.items():
            f.ellipse((tx - 5, ty - 5, tx + 5, ty + 5), fill=GREEN)
            f.text((tx + 10, ty), t[:18], F(8.5, mono=True), MUTED, anchor="lm")
        f.caption(14, OUT_H - 34, OUT_W - 28,
                  "Laid out here for the page — the edges are the real [[wikilinks]] in the "
                  "real notes, read from the vault. The same files are what Obsidian opens; "
                  "there is no plugin, no export and no index to keep.")
        out += play([f.done()], 2800)

    return dict(frames=out, still=-1)


# =============================================================================
# Film 3 -- Gmail summarization
# =============================================================================

def gmail_film(caps, vault):
    gmail = caps.get("gmail")
    out = []

    # 1 -- what has to be true before the inbox is readable
    f = Frame("1  read-only, or nothing", CORAL, "scope gmail.readonly")
    y = f.card((14, BAR + 12, 452, OUT_H - 14), "one-time setup", CORAL)
    for i, ln in enumerate([
        "python -m text_summarizer.gmail_oauth",
        "",
        "GOOGLE_CLIENT_ID=...apps.googleusercontent.com",
        "GOOGLE_CLIENT_SECRET=...",
        "GOOGLE_REFRESH_TOKEN=...   (minted once)",
    ]):
        f.text((30, y + i * 22), ln, F(9.5, mono=True),
               GREEN if "REFRESH" in ln else MUTED, anchor="lt")
    yy = y + 5 * 22 + 10
    f.rect((30, yy, 436, yy + 1), fill=STROKE)
    f.text((30, yy + 14), "scope", F(10, bold=True), CORAL, anchor="lt")
    f.chip(78, yy + 10, "gmail.readonly", CORAL, bg=(66, 31, 29), f=F(10, mono=True))
    f.caption(30, yy + 46, 410,
              "No send, modify or delete tool is declared at all, so the grant cannot be used "
              "to change anything. All three variables have to be present, or "
              "build_gmail_tools() returns an empty list and the agent simply cannot read "
              "mail — the rest of it keeps working.")

    ry = f.card((468, BAR + 12, OUT_W - 14, OUT_H - 14), "the four tools it adds", GREEN)
    for i, (name, why) in enumerate([
        ("gmail_get_latest_messages", "headers and snippets, cheap"),
        ("gmail_search", "a query: label, sender, window"),
        ("gmail_read", "one message, full body"),
        ("gmail_get_thread", "a whole exchange, in order"),
    ]):
        f.text((484, ry + i * 50), name, F(10.5, mono=True), GREEN, anchor="lt")
        f.text((484, ry + i * 50 + 18), why, F(9.5), MUTED, anchor="lt")
    f.rect((484, ry + 204, OUT_W - 30, ry + 205), fill=STROKE)
    f.caption(484, ry + 216, OUT_W - 30 - 484,
              "The toolset is additive: Gmail is one optional integration among several, and a "
              "deployment without credentials behaves exactly like one with them minus the "
              "mail tools.")
    out += play([f.done()], 2400)

    if gmail:
        # 2 -- the question
        f = Frame("2  a question about the inbox", CORAL, "the real composer, then sent")
        f.region(gmail.at(1), (480, 150, 1388, 300))
        f.caption(14, BAR + 168, OUT_W - 28,
                  "Nothing about this turn is special. The phrasing is what selects the tool: a "
                  "count, a label, a sender or a word like thread each steer the search, and "
                  "the agent has no way to answer from its own knowledge — every fact in the "
                  "summary has to come back from a tool.")
        out += play([f.done()], 2200)

        # 3 -- the real searches
        seq = []
        for i in gmail.distinct(1.2)[1:]:
            g = Frame("3  the phrasing picks the query", CORAL, f"t+{gmail.times[i]:.1f}s")
            g.shot(gmail.at(i), y0=BOTTOM)
            seq.append(g.done())
        out += play(seq, 200) + hold(seq, 400)

        # 4 -- the honest answer
        f = Frame("4  nothing found is reported, not invented", CORAL,
                  f"{len(gmail)} frames  ·  {gmail.times[-1] - gmail.times[2]:.1f}s recorded")
        f.shot(gmail.last(), y0=BOTTOM)
        x = 24
        for label, col, bg in (("rule 11", CORAL, (66, 31, 29)),
                               ("only what the tools returned", MUTED, PANEL2),
                               ("no thread invented", MUTED, PANEL2)):
            x += f.chip(x, OUT_H - 32, label, col, bg=bg, f=F(9.5, mono=True),
                        border=STROKE if bg is PANEL2 else None) + 8
        out += play([f.done()], 2800)

    # 5 -- saved like any other turn
    if gmail:
        f = Frame("5  saved like any other turn", GREEN, "no special case")
        y = f.card((14, BAR + 12, OUT_W - 14, OUT_H - 14), "the same second-brain path", GREEN)
        for i, (tool, effect, col) in enumerate([
            ("gmail_search", "find the messages worth reading", CORAL),
            ("gmail_read / gmail_get_thread", "read their bodies", CORAL),
            ("save_summary_to_second_brain", "one dated note, a stub per topic", GREEN),
            ("log_conversation", "the exchange, in the chat log", VIOLET),
        ]):
            yy = y + i * 58
            f.rrect((30, yy, OUT_W - 30, yy + 48), r=8, fill=PANEL2, outline=STROKE, w=1)
            f.text((46, yy + 24), tool, F(10, mono=True), col, anchor="lm")
            f.text((OUT_W - 46, yy + 24), effect, F(10), MUTED, anchor="rm")
            if i < 3:
                f.line((60, yy + 48, 60, yy + 58), STROKE, 1)
        f.caption(30, y + 4 * 58 + 8, OUT_W - 60,
                  "An inbox turn reuses the same summarizer, the same rules and the same vault. "
                  "It is also cached like any other turn, which is how a repeat of the same "
                  "question came to cost nothing at all.")
        out += play([f.done()], 2600)

    return dict(frames=out, still=-1)


def build_films(caps, vault):
    return {
        "main": main_film(caps, vault),
        "obsidian": obsidian_film(caps, vault),
        "gmail": gmail_film(caps, vault),
    }
