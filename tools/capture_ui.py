#!/usr/bin/env python3
"""Drive the real ADK dev UI in Firefox and save a frame sequence of a turn.

The walkthrough GIFs on the docs page are recordings of this app, not mock-ups:
this script opens the dev UI at :8001, types a prompt, and screenshots the real
chat and the real event stream while the agent works. `tools/build_gifs.py` then
assembles the frames into the filmstrips.

    python3 tools/capture_ui.py --out /tmp/cap/main --prompt "Summarize: ..."
    python3 tools/capture_ui.py --out /tmp/cap/gmail --prompt "What are my latest 5 emails?"

Requires the compose stack to be up (`./run.sh`, or `docker compose up -d`) and a
Firefox already listening for Marionette on 127.0.0.1:2828:

    firefox --headless --marionette about:blank &

The API key in .env is spent on every turn here, so each scene is one real call.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from firefox_marionette import Firefox  # noqa: E402

UI = "http://localhost:8001/dev-ui/"

# What the composer shows while a turn runs. Recorded for the log only: the
# button keeps the text "Turn Complete" after the turn, so a screen that has
# stopped moving is the only trustworthy "the turn is over" signal.
PROBE = """
const send = [...document.querySelectorAll('button')].find(b => /send|stop|Turn Complete/.test(b.textContent||''));
return JSON.stringify({ composer: send ? (send.textContent||'').trim() : null });
"""


def attach(port: int = 2828, width: int = 1400, height: int = 900) -> Firefox:
    ff = Firefox(width, height)
    ff.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    ff.sock.settimeout(300)
    ff._read_packet()
    ff.call("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {}}})
    ff.call("WebDriver:SetWindowRect", {"width": width, "height": height, "x": 0, "y": 0})
    ff.call("WebDriver:SetTimeouts", {"implicit": 0, "pageLoad": 60000, "script": 30000})
    return ff


def dismiss_dialogs(ff: Firefox) -> None:
    """Close the telemetry consent dialog and anything else modal."""
    for label in ("No Thanks", "Dismiss", "Close"):
        try:
            ff.click(f"mat-dialog-container button:text-is('{label}')")
        except Exception:
            continue
        for _ in range(20):
            if not ff.script("return !!document.querySelector('mat-dialog-container')"):
                return
            time.sleep(0.25)
    # Fall back to removing it from the DOM: a stray overlay would sit on top of
    # every frame otherwise.
    ff.script("document.querySelectorAll('mat-dialog-container').forEach(d => d.remove()); return 1")


def new_session(ff: Firefox) -> None:
    """Start a fresh chat session so a capture never inherits the last turn."""
    try:
        ff.click("button:has-text('NEW SESSION')")
    except Exception:
        try:
            ff.click("mat-select[aria-label*='session' i], .session-select")
        except Exception:
            return
    time.sleep(0.6)
    for label in ("New session", "NEW SESSION"):
        try:
            ff.click(f"button:text-is('{label}'), .mat-mdc-option:text-is('{label}')")
            break
        except Exception:
            continue
    time.sleep(1.2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="directory for the frames")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--name", default="turn")
    ap.add_argument("--every", type=float, default=1.5, help="seconds between frames")
    ap.add_argument("--settle", type=float, default=6.0, help="seconds to wait after the turn ends")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--keep", action="store_true", help="keep the session open when done")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--no-send", action="store_true", help="capture the composer only, do not send")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ff = attach(width=args.width, height=args.height)
    frames: list[dict] = []

    def shoot(tag: str, force: bool = False) -> bool:
        """Save a frame, skipping it when the screen has not moved.

        The UI is a live app: while a turn runs the event list, the token
        counter and the typing indicator all move, so an unchanged screenshot is
        a genuinely idle moment. Dropping those is what keeps a recording from
        being hundreds of identical frames.
        """
        nonlocal last
        raw = ff.shot_raw()
        if not force and raw == last:
            return False
        last = raw
        name = f"{args.name}-{len(frames):03d}-{tag}.png"
        with open(os.path.join(args.out, name), "wb") as fh:
            fh.write(raw)
        frames.append({"file": name, "t": round(time.time() - t0, 2),
                       "state": state.get("composer")})
        print(f"  {name}  {frames[-1]['t']}s  {frames[-1]['state']}", flush=True)
        return True

    try:
        ff.open(UI)
        time.sleep(4.0)
        dismiss_dialogs(ff)
        new_session(ff)
        time.sleep(1.0)

        t0 = time.time()
        last: bytes | None = None
        state: dict = {}
        shoot("empty", force=True)
        ff.send_keys("textarea.chat-input-box", args.prompt)
        time.sleep(0.4)
        shoot("typed", force=True)
        if args.no_send:
            print(json.dumps(frames, indent=2))
            return 0

        ff.press_enter("textarea.chat-input-box")
        deadline = time.time() + args.timeout
        quiet = 0
        while time.time() < deadline:
            time.sleep(args.every)
            try:
                state = json.loads(ff.script(PROBE))
            except Exception:
                state = {}
            if shoot("live"):
                quiet = 0
            else:
                quiet += 1
                # The turn is over once the screen has stopped moving for three
                # polls: while a turn runs the event list grows on every step, so
                # a static screen is a finished turn, not a slow generation.
                if quiet >= 3:
                    break
        time.sleep(args.settle)
        shoot("final", force=True)
        print(f"captured {len(frames)} frames in {time.time() - t0:.1f}s")
        with open(os.path.join(args.out, "index.json"), "w") as fh:
            json.dump({"prompt": args.prompt, "frames": frames}, fh, indent=2)
        return 0
    finally:
        if not args.keep:
            ff.close()


if __name__ == "__main__":
    sys.exit(main())
