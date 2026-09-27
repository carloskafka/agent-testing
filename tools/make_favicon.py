#!/usr/bin/env python3
"""Draw the site icon from the M3 colour roles and write it out in four forms.

The tab icon is a squircle in --primary with the Material Symbols "summarize"
glyph in --on-primary, which is the same relationship the page's filled buttons
use, so the icon and the UI are the same design language. Two variants, because
M3 defines separate values for primary and on-primary in dark mode and a deep
blue tab icon is invisible on a dark tab strip.

    python3 tools/make_favicon.py

Writes into docs/:
    favicon.svg          light, squircle          (Chromium, Firefox, Safari 10+)
    favicon-dark.svg     dark, squircle           (via <link media>)
    favicon-32.png       light raster fallback
    apple-touch-icon.png 180x180, full bleed     (iOS masks it itself)

Nothing is fetched at run time. The glyph below is Material Symbols "summarize"
(Apache 2.0, https://fonts.google.com/icons), inlined so the build does not
depend on a font CDN, and parsed here rather than handed to a rasteriser because
the usual rasterisers (rsvg, inkscape, cairosvg) are not installed.

Two details worth knowing if this is edited:

  * The path is in Material Symbols' own coordinate system -- 960 units square
    with y running negative upwards -- so it is centred on the ink's bounding box
    rather than on the viewBox, or the glyph sits visibly low and left.
  * The second subpath is a document outline with a hole in it, so the fill is
    even-odd. Pillow's polygon fill paints subpaths independently and would fill
    the hole solid, so the subpaths are combined with XOR instead.
"""

from __future__ import annotations

import os
import re

from PIL import Image, ImageChops, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, os.pardir, "docs")
CSS = os.path.join(DOCS, "index.html")

# Material Symbols "summarize" (Apache 2.0, https://fonts.google.com/icons),
# inlined so the build never depends on a font CDN, and parsed here rather than
# handed to a rasteriser because the usual rasterisers (rsvg, inkscape,
# cairosvg) are not installed. In Material Symbols' own coordinate system:
# 960 units square, y running negative upwards.
GLYPH = (
    "M330-630q9-9 9-21t-9-21q-9-9-21-9t-21 9q-9 9-9 21t9 21q9 9 21 9t21-9Z"
    "m0 171q9-9 9-21t-9-21q-9-9-21-9t-21 9q-9 9-9 21t9 21q9 9 21 9t21-9Z"
    "m0 171q9-9 9-21t-9-21q-9-9-21-9t-21 9q-9 9-9 21t9 21q9 9 21 9t21-9Z"
    "M180-120q-24 0-42-18t-18-42v-600q0-24 18-42t42-18h462l198 198v462"
    "q0 24-18 42t-42 18H180Zm0-60h600v-428.57H609V-780H180v600Z"
    "m0-600v171.43V-780v600-600Z"
)


def token_pair(text: str) -> tuple:
    """(primary, on-primary) from a slice of the stylesheet, last declaration.

    The icon is read out of index.html rather than repeated here, so a change to
    --primary cannot leave a stale hex sitting in docs/.
    """
    primary = re.findall(r"--primary:\s*(#[0-9a-fA-F]{3,8})", text)
    on_primary = re.findall(r"--on-primary:\s*(#[0-9a-fA-F]{3,8})", text)
    if not primary or not on_primary:
        raise SystemExit(
            "could not find --primary/--on-primary in the given slice of "
            f"{os.path.relpath(CSS, HERE)} -- the icon is drawn from those tokens, so"
            " fix the stylesheet or update this script")
    return primary[-1].lower(), on_primary[-1].lower()

# How much of the icon box the glyph's height fills. A favicon is seen at 16px,
# where a glyph filling the tile is mush; 64% keeps it readable at that size and
# still reads as the mark at 180.
GLYPH_FILL = 0.64
CURVE_SEGMENTS = 16     # flattening step for each quadratic


def tokenise(d: str) -> list:
    """Numbers and commands, with SVG's sign-as-separator ("9-9" is two numbers)."""
    return re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)


def parse(d: str) -> list:
    """The path as a list of subpaths, each a list of (x, y) points."""
    tokens = tokenise(d)
    i = 0
    subpaths: list = []
    cur: list = []
    x = y = 0.0            # current point
    start = (0.0, 0.0)    # start of the subpath, for Z
    cx = cy = None        # last quadratic control point, for T/t
    cmd = None

    def flush():
        nonlocal cur
        if len(cur) > 2:
            subpaths.append(cur)
        cur = []

    def quad(px, py, qx, qy, ex, ey):
        """Flatten one quadratic; SVG's T/t reflect the previous control point."""
        nonlocal x, y, cx, cy
        pts = []
        for step in range(1, CURVE_SEGMENTS + 1):
            t = step / CURVE_SEGMENTS
            u = 1 - t
            pts.append((u * u * px + 2 * u * t * qx + t * t * ex,
                        u * u * py + 2 * u * t * qy + t * t * ey))
        cur.extend(pts)
        x, y, cx, cy = ex, ey, qx, qy

    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            i += 1
            if cmd in "Zz":
                cur.append(start)
                flush()
                x, y = start
                cx = cy = None
                continue
        # An M or m is a moveto only for its first pair; the pairs after it are
        # implicit linetos, which is what cmd is switched to at the bottom of
        # those two branches.
        rel = cmd.islower()
        c = cmd.upper()

        if c == "M":
            nx, ny = float(tokens[i]), float(tokens[i + 1])
            i += 2
            flush()
            x, y = (x + nx, y + ny) if rel else (nx, ny)
            start = (x, y)
            cur = [(x, y)]
            cx = cy = None
            cmd = "L" if not rel else "l"
        elif c == "L":
            nx, ny = float(tokens[i]), float(tokens[i + 1])
            i += 2
            x, y = (x + nx, y + ny) if rel else (nx, ny)
            cur.append((x, y))
            cx = cy = None
        elif c == "H":
            nx = float(tokens[i])
            i += 1
            x = x + nx if rel else nx
            cur.append((x, y))
            cx = cy = None
        elif c == "V":
            ny = float(tokens[i])
            i += 1
            y = y + ny if rel else ny
            cur.append((x, y))
            cx = cy = None
        elif c == "m":
            nx, ny = float(tokens[i]), float(tokens[i + 1])
            i += 2
            flush()
            x, y = x + nx, y + ny
            start = (x, y)
            cur = [(x, y)]
            cx = cy = None
            cmd = "l"
        elif c == "Q":
            qx, qy = float(tokens[i]), float(tokens[i + 1])
            ex, ey = float(tokens[i + 2]), float(tokens[i + 3])
            i += 4
            if rel:
                qx, qy, ex, ey = x + qx, y + qy, x + ex, y + ey
            quad(x, y, qx, qy, ex, ey)
        elif c == "T":
            ex, ey = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if rel:
                ex, ey = x + ex, y + ey
            qx, qy = (2 * x - cx, 2 * y - cy) if cx is not None else (x, y)
            quad(x, y, qx, qy, ex, ey)
        else:
            raise ValueError(f"unsupported path command {cmd!r}")
    flush()
    return subpaths


def ink_box(subpaths: list) -> tuple:
    xs = [p[0] for sp in subpaths for p in sp]
    ys = [p[1] for sp in subpaths for p in sp]
    return min(xs), min(ys), max(xs), max(ys)


def hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def render(size: int, background: str, glyph: str, radius_ratio: float,
           supersample: int = 8) -> Image.Image:
    """The mark at `size` px. radius_ratio 0 gives a full-bleed square."""
    subpaths = parse(GLYPH)
    big = size * supersample
    box = ink_box(subpaths)
    ink_w, ink_h = box[2] - box[0], box[3] - box[1]
    scale = (big * GLYPH_FILL) / ink_h          # height is the tighter axis
    off_x = (big - ink_w * scale) / 2 - box[0] * scale
    off_y = (big - ink_h * scale) / 2 - box[1] * scale

    # Even-odd across subpaths, so the hole in the document outline stays open.
    mask = Image.new("L", (big, big), 0)
    for sp in subpaths:
        layer = Image.new("L", (big, big), 0)
        ImageDraw.Draw(layer).polygon(
            [(px * scale + off_x, py * scale + off_y) for px, py in sp], fill=255)
        mask = ImageChops.difference(mask, layer)

    canvas = Image.new("RGBA", (big, big), hex_to_rgb(background) + (255,))
    if radius_ratio:
        r = big * radius_ratio
        # Round the corners by punching them out of an otherwise full square.
        squircle = Image.new("L", (big, big), 0)
        ImageDraw.Draw(squircle).rounded_rectangle((0, 0, big - 1, big - 1),
                                                   radius=r, fill=255)
        canvas.putalpha(squircle)
    canvas.paste(hex_to_rgb(glyph) + (255,), (0, 0), mask)
    return canvas.resize((size, size), Image.LANCZOS)


def svg(background: str, glyph: str, radius: float) -> str:
    subpaths = parse(GLYPH)
    box = ink_box(subpaths)
    ink_w, ink_h = box[2] - box[0], box[3] - box[1]
    scale = (48 * GLYPH_FILL) / ink_h
    off_x = 24 - ink_w * scale / 2 - box[0] * scale
    off_y = 24 - ink_h * scale / 2 - box[1] * scale
    r = 48 * radius
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48"
     viewBox="0 0 48 48" role="img" aria-label="Text Summarizer Agent">
  <rect width="48" height="48" rx="{r:.2f}" fill="{background}"/>
  <path fill="{glyph}" transform="translate({off_x:.3f} {off_y:.3f}) scale({scale:.5f})"
        d="{GLYPH}"/>
</svg>
"""


def write(name: str, data) -> None:
    path = os.path.join(DOCS, name)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode) as fh:
        fh.write(data)
    size = os.path.getsize(path)
    print(f"  {name:24s} {size:6d} bytes")


def main() -> int:
    with open(CSS, encoding="utf-8") as fh:
        page = fh.read()
    for marker in ("<style>", "@media (prefers-color-scheme: dark)"):
        if marker not in page:
            raise SystemExit(f"no {marker!r} in {os.path.relpath(CSS, HERE)}")
    # Only the stylesheet: the same media query also appears in <meta> and
    # <link> attributes, and the first of those sits above :root.
    css = page[page.index("<style>"):]
    dark_at = css.index("@media (prefers-color-scheme: dark)")
    light = token_pair(css[:dark_at])      # the :root baseline
    dark = token_pair(css[dark_at:])       # the dark-scheme block
    print(f"writing the site icon   light {light[0]} on {light[1]}"
          f"   dark {dark[0]} on {dark[1]}")
    write("favicon.svg", svg(*light, radius=0.25))
    write("favicon-dark.svg", svg(*dark, radius=0.25))
    # iOS applies its own mask, so a pre-rounded tile would be rounded twice.
    write("apple-touch-icon.png", _png_bytes(render(180, *light, radius_ratio=0.0)))
    write("favicon-32.png", _png_bytes(render(32, *light, radius_ratio=0.25)))
    return 0


def _png_bytes(im: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
