#!/usr/bin/env python3
"""Assemble the end-to-end walkthrough GIFs from real recordings of the agent.

The frames in docs/assets/ are **recordings**, not illustrations: they are
screenshots of the ADK dev UI at :8001 while it runs a real turn, captured by
``tools/capture_ui.py`` driving Firefox over Marionette, plus a few panels that
render *real* files read back out of the live vault (there is no file manager or
Obsidian on this host to photograph). Every string in a panel is read from disk
at build time, so the walkthrough cannot drift from the vault it describes.

    sh tools/check_stack.sh                      # both services healthy
    python3 tools/capture_ui.py --out /tmp/cap/live --prompt "Summarize: ..."
    python3 tools/build_gifs.py --captures /tmp/cap

Output per film: an animated GIF plus a still for ``prefers-reduced-motion``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

# --- geometry ----------------------------------------------------------------

OUT_W, OUT_H = 920, 640          # the page's content column is 920 CSS px
BAR = 30                         # drawn header strip
BODY_H = OUT_H - BAR             # 610 of real pixels, never scaled
CHAT_X = 480                     # the dev UI's chat panel starts here at 1400px wide
BOTTOM = 10_000                  # a y0 this large means "as low as the frame allows"

# --- palette (matches the docs page's own dark scheme) ----------------------

BG = (13, 16, 20)
PANEL = (21, 26, 33)
PANEL2 = (27, 33, 42)
STROKE = (43, 52, 63)
TEXT = (231, 236, 242)
MUTED = (150, 162, 178)
FAINT = (104, 115, 130)
BLUE = (124, 178, 255)
GREEN = (109, 220, 154)
AMBER = (242, 193, 78)
VIOLET = (183, 156, 255)
CORAL = (242, 139, 130)

FONTDIR = "/usr/share/fonts/truetype/dejavu"
FONTDIR_ENV = "AGENT_TESTING_FONTDIR"
_fonts: dict = {}


def font_dir() -> str:
    """The directory `F` loads faces from: the override if set, else `FONTDIR`.

    The constant above is one host's layout -- where `fonts-dejavu-core` installs
    on Debian and Ubuntu -- and this module is otherwise portable. Resolved per
    call rather than at import, because an override set after import (which is
    what a test's `monkeypatch.setenv` is) would otherwise be read too early.
    """
    return os.environ.get(FONTDIR_ENV) or FONTDIR


def F(size: float, bold: bool = False, mono: bool = False):
    key = (size, bold, mono)
    hit = _fonts.get(key)
    if hit is None:
        face = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "")
        path = os.path.join(font_dir(), face + ".ttf")
        if not os.path.exists(path):
            # ImageFont.truetype's own message names the file but not the knob
            # that fixes it, so a missing font package reads as a bug here.
            raise OSError(
                f"no font at {path}. Install the DejaVu faces "
                f"(fonts-dejavu-core), or point {FONTDIR_ENV} at a directory "
                "that has them.")
        hit = ImageFont.truetype(path, int(size))
        _fonts[key] = hit
    return hit


def wrap(d: ImageDraw.ImageDraw, s: str, f, max_w: float) -> list[str]:
    out: list[str] = []
    for para in s.split("\n"):
        cur = ""
        for word in para.split():
            trial = (cur + " " + word).strip()
            if not cur or d.textlength(trial, font=f) <= max_w:
                cur = trial
            else:
                out.append(cur)
                cur = word
        out.append(cur)
    return out


# --- the frame --------------------------------------------------------------

class Frame:
    def __init__(self, title: str, accent: tuple, tag: str = ""):
        self.img = Image.new("RGB", (OUT_W, OUT_H), BG)
        self.d = ImageDraw.Draw(self.img)
        self.d.rectangle((0, 0, OUT_W, BAR), fill=PANEL)
        self.d.line((0, BAR, OUT_W, BAR), fill=STROKE)
        self.d.text((12, BAR // 2), title, font=F(11.5, bold=True, mono=True), fill=accent,
                    anchor="lm")
        if tag:
            self.d.text((OUT_W - 12, BAR // 2), tag, font=F(10, mono=True), fill=FAINT,
                        anchor="rm")
        self.d.text((OUT_W // 2, BAR // 2), "", font=F(10), fill=FAINT)

    # -- primitives (coordinates are frame-absolute) --

    def rrect(self, box, r=8, fill=None, outline=None, w=1):
        self.d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=w)

    def rect(self, box, fill=None, outline=None, w=1):
        self.d.rectangle(box, fill=fill, outline=outline, width=w)

    def text(self, xy, s, f, fill, anchor="la"):
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor)

    def tw(self, s, f) -> float:
        return self.d.textlength(s, font=f)

    def para(self, x, y, lines, f, fill, lh=1.5):
        step = f.size * lh
        for i, ln in enumerate(lines):
            self.text((x, y + i * step), ln, f, fill, anchor="lt")
        return y + len(lines) * step

    def ellipse(self, box, fill=None, outline=None, w=1):
        self.d.ellipse(box, fill=fill, outline=outline, width=w)

    def line(self, pts, fill, w=1, dash=None):
        self.d.line(pts, fill=fill, width=w)

    def shot(self, png: str, y0: int = 0, x0: int = CHAT_X, margin: int = 6) -> "Frame":
        """Paste a full-height strip of a real screenshot, never scaled.

        `x0` is where the strip is read from; it is pasted at `margin`, not at
        `x0`, so the whole 908px chat column lands inside the frame instead of
        being pushed off the right edge.
        """
        im = Image.open(png).convert("RGB")
        y0 = max(0, min(y0, im.height - BAR - BODY_H))
        w = min(OUT_W - margin * 2, im.width - x0)
        h = min(BODY_H, im.height - (BAR + y0))
        self.img.paste(im.crop((x0, BAR + y0, x0 + w, BAR + y0 + h)), (margin, BAR))
        self.rect((margin - 1, BAR - 1, margin + w, BAR + h), outline=STROKE, w=1)
        return self

    def region(self, png: str, src, dst=None, border=True) -> "Frame":
        """Paste a 1:1 crop of a real screenshot at `dst` (default: top of body).

        Screenshots are cropped, never rescaled: at 1:1 the type in a recording
        is the type the reader would see in the app.
        """
        im = Image.open(png).convert("RGB")
        crop = im.crop(src)
        if dst is None:
            dst = (0, BAR)
        self.img.paste(crop, dst)
        if border:
            self.rect((dst[0], dst[1], dst[0] + crop.width, dst[1] + crop.height),
                      outline=STROKE, w=1)
        return self

    def caption(self, x, y, w, s, f=None, fill=MUTED, lh=1.5) -> float:
        f = f or F(10)
        return self.para(x, y, wrap(self.d, s, f, w), f, fill, lh=lh)

    def shot_fit(self, png: str, box, bg=BG) -> "Frame":
        """Paste a whole screenshot scaled to fit a box, preserving aspect."""
        x0, y0, x1, y1 = box
        im = Image.open(png).convert("RGB")
        im.thumbnail((x1 - x0, y1 - y0), Image.LANCZOS)
        self.rect(box, fill=bg)
        self.img.paste(im, (x0 + (x1 - x0 - im.width) // 2, y0 + (y1 - y0 - im.height) // 2))
        return self

    def card(self, box, title=None, accent=None, fill=PANEL, r=10):
        self.rrect(box, r=r, fill=fill, outline=STROKE, w=1)
        if title:
            self.text((box[0] + 14, box[1] + 14), title, F(10.5, bold=True, mono=True),
                      accent or MUTED, anchor="lt")
            return box[1] + 32
        return box[1] + 12

    def chip(self, x, y, label, color=TEXT, bg=None, border=None, f=None, h=20):
        f = f or F(10, bold=True)
        w = self.tw(label, f) + 16
        if bg or border:
            self.rrect((x, y, x + w, y + h), r=h / 2, fill=bg, outline=border, w=1)
        self.text((x + 8, y + h / 2), label, f, color, anchor="lm")
        return w

    def meter(self, x, y, w, p, color=BLUE, h=5):
        self.rrect((x, y, x + w, y + h), r=h / 2, fill=PANEL2)
        if p > 0:
            self.rrect((x, y, x + max(h, w * p), y + h), r=h / 2, fill=color)

    def done(self) -> Image.Image:
        return self.img


# --- capture helpers --------------------------------------------------------

class Captures:
    """The frames one ``capture_ui.py`` run produced, in order."""

    def __init__(self, root: str, name: str):
        self.dir = os.path.join(root, name)
        with open(os.path.join(self.dir, "index.json")) as fh:
            self.index = json.load(fh)
        self.prompt = self.index["prompt"]
        self.frames = [os.path.join(self.dir, f["file"]) for f in self.index["frames"]]
        self.times = [f["t"] for f in self.index["frames"]]

    def __len__(self) -> int:
        return len(self.frames)

    def at(self, i: int) -> str:
        return self.frames[max(0, min(len(self.frames) - 1, i))]

    def last(self) -> str:
        return self.frames[-1]

    def first_live(self) -> str:
        for p in self.frames:
            if "live" in os.path.basename(p) or "final" in os.path.basename(p):
                return p
        return self.frames[-1]

    def with_text(self, needle: str) -> str | None:
        """The first frame whose event list mentions `needle`."""
        for p in self.frames:
            if needle in os.path.basename(p):
                return p
        return None

    def elapsed(self) -> float:
        return self.times[-1] - self.times[0]

    def distinct(self, threshold: float = 2.0) -> list[int]:
        """Indices of the frames that actually changed, in order.

        A recording of a 40s turn has long stretches where the only thing moving
        is a progress bar, and playing those at 190ms each just pads the loop.
        Frames are compared as small greyscale thumbnails and one is kept only
        when enough of it moved.
        """
        kept: list[int] = []
        prev = None
        for i, path in enumerate(self.frames):
            im = Image.open(path).convert("L")
            im.thumbnail((160, 160), Image.BILINEAR)
            px = im.tobytes()
            if prev is not None:
                diff = sum(1 for a, b in zip(px, prev) if abs(a - b) > 12)
                if diff / max(1, len(px)) * 100 < threshold:
                    continue
            prev = px
            kept.append(i)
        return kept


# --- a scene is a list of (image, ms) ---------------------------------------


def hold(frames: list[Image.Image], ms: int) -> list[tuple[Image.Image, int]]:
    """Repeat the last frame for `ms`, so a pause costs one frame of bytes."""
    return [(frames[-1], ms)]


def play(frames: list[Image.Image], ms: int = 260) -> list[tuple[Image.Image, int]]:
    return [(f, ms) for f in frames]


def reveal(caps: Captures, a: int, b: int, ms: int = 220) -> list[tuple[Image.Image, int]]:
    return play(caps.frames[a:b + 1], ms)


# --- real vault data --------------------------------------------------------

class Vault:
    """The live vault, read straight off disk.

    The panels that show notes, topics and the graph are rendered from these
    files, so they are the real thing rather than a picture of it. Absolute host
    paths never reach a frame: only names, so nothing leaks the layout of the
    machine that recorded it.
    """

    def __init__(self, root: str):
        self.root = root
        self.notes_dir = os.path.join(root, "Second Brain")
        self.topics_dir = os.path.join(root, "Topics")
        self.notes = sorted(f[:-3] for f in os.listdir(self.notes_dir) if f.endswith(".md"))
        self.topics = sorted(f[:-3] for f in os.listdir(self.topics_dir) if f.endswith(".md"))

    def read(self, title: str) -> str:
        with open(os.path.join(self.notes_dir, f"{title}.md"), encoding="utf-8") as fh:
            return fh.read()

    def links(self, title: str) -> list[str]:
        return re.findall(r"\[\[([^\]]+)\]\]", self.read(title))

    def note_titles_in(self, title: str) -> list[str]:
        """Wiki links from a note that point at other notes (not topic stubs)."""
        return [l for l in self.links(title) if l in self.notes]


# --- encoding ---------------------------------------------------------------

def write_gif(frames: list[tuple[Image.Image, int]], path: str, still: int | None = None) -> dict:
    merged: list[tuple[Image.Image, int]] = []
    for img, ms in frames:
        if merged and merged[-1][0].tobytes() == img.tobytes():
            merged[-1] = (merged[-1][0], merged[-1][1] + ms)
        else:
            merged.append((img, ms))
    step = max(1, len(merged) // 12)
    montage = Image.new("RGB", (OUT_W * 3, OUT_H * 4))
    for i, (img, _) in enumerate(merged[::step][:12]):
        montage.paste(img, ((i % 3) * OUT_W, (i // 3) * OUT_H))
    palette = montage.quantize(colors=255, method=Image.MEDIANCUT)
    quants = [img.quantize(palette=palette, dither=Image.Dither.NONE) for img, _ in merged]
    quants[0].save(path, save_all=True, append_images=quants[1:],
                   duration=[ms for _, ms in merged], loop=0, optimize=True, disposal=1)
    if still is not None:
        # Full size, not half. The poster is displayed at the width of the stage,
        # which on a HiDPI screen is more device pixels than a half-size still
        # has, so a small one is visibly soft for the second or two before the
        # canvas appears -- and for good, for a reduced-motion reader.
        quants[still].convert("RGB").save(
            path.replace(".gif", "-still.png"), optimize=True)
    return {"frames": len(merged), "ms": sum(ms for _, ms in merged),
            "bytes": os.path.getsize(path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", default="/tmp/cap", help="root of the capture_ui.py output")
    ap.add_argument("--vault", help="the live vault directory (the one named in .env)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), os.pardir,
                                                  "docs", "assets"))
    args = ap.parse_args()
    from films import build_films  # local module next to this one

    caps = {name: Captures(args.captures, name)
            for name in ("live", "ask", "repeat", "gmail", "digest")
            if os.path.isdir(os.path.join(args.captures, name))}
    vault = Vault(args.vault) if args.vault else None
    os.makedirs(args.out, exist_ok=True)
    for name, spec in build_films(caps, vault).items():
        frames = spec["frames"]
        out = os.path.join(args.out, f"e2e-{name}.gif")
        info = write_gif(frames, out, still=spec.get("still"))
        print(f"{name:9s} {info['frames']:3d} frames  {info['ms'] / 1000:5.1f}s  "
              f"{info['bytes'] / 1024:7.1f} KiB  {out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
