#!/usr/bin/env python3
"""Make the bundled ADK dev UI usable on a phone, in place.

Why this exists
---------------
`adk web` serves a prebuilt Angular app from
``google/adk/cli/browser/index.html``. It is a **desktop developer tool** and
makes no attempt at mobile support. Measured across both of its style blocks and
all 100 JS bundles:

* ``dvh`` / ``svh``:  **0** occurrences
* ``safe-area-inset-*``: **0** occurrences
* ``viewport-fit=cover``: absent from the viewport meta

Meanwhile three layers pin their height to the legacy ``100vh`` unit:

    body                { height:100vh; overflow:hidden }
    [_nghost-*]         { height:100vh; overflow:hidden }   <- app shell
    .builder-mode-container { height:100vh }

On a phone ``100vh`` resolves to the *largest* viewport (the height you get with
the URL bar retracted), not the visible one, so the shell is 15-20% taller than
the screen. The chat composer is the last row of that fixed-height flex column,
so it lands below the fold -- and because ``body`` and the shell are both
``overflow:hidden``, the overflow is unreachable. The on-screen keyboard makes
it worse, halving the visible height.

The result is exactly the reported symptom: **messages render fine, the input
box does not**, because the bubbles are ``max-width:800px`` in a centred column
and shrink correctly. The breakage is vertical, not horizontal.

What it does
------------
Injects one stylesheet before ``</head>`` rather than rewriting vendor rules, so
the patch is easy to verify and to remove. Wrapped in ``@supports (height:
100dvh)`` so a browser without it keeps the vendor behaviour untouched.

  * ``100dvh`` heights, so the shell tracks the *dynamic* viewport -- this is
    what fixes both the URL bar and the on-screen keyboard
  * safe-area bottom padding on the composer, for the home indicator
  * the assistant side panel capped to the viewport instead of a hard 400px

It also adds two viewport meta parameters, which cannot be done in CSS:

  * ``viewport-fit=cover`` -- without it ``env(safe-area-inset-*)`` is always 0,
    so the safe-area padding above would be inert on a notched device
  * ``interactive-widget=resizes-content`` -- Chromium overlays the keyboard by
    default, which covers the composer. This resizes the layout viewport
    instead. Safari has no equivalent and relies on ``dvh``.

Deliberately not done: the patch cannot rewrite the app's own rules, whose
selectors carry a per-build ``[_ngcontent-<hash>]`` qualifier. Overrides are
therefore written as bare class names with ``!important``, which beats the
higher-specificity vendor rule. If a future ADK renames these classes the
overrides silently stop applying -- the safe-area padding degrades to zero and
nothing breaks loudly. That is the accepted trade for not forking a bundle.

Run with no arguments it locates the file inside ``.venv``; pass a path to patch
a specific one. Exits non-zero if the file does not look like the ADK dev UI, so
an ADK upgrade that changes it fails the Docker build instead of shipping a
patch that does nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Identifies our own output, so re-running is a no-op rather than a second copy.
MARKER = "adk-mobile-fix"

#: Strings that must be present for the patch to mean anything. Their absence
#: means ADK changed the file and a maintainer needs to look.
EXPECTED = (
    "body{height:100vh",
    "<app-root>",
)

#: The viewport meta, matched loosely so formatting changes do not break it.
VIEWPORT_META = re.compile(
    r'<meta\s+name="viewport"\s+content="(?P<content>[^"]*)"\s*/?>', re.IGNORECASE
)

OVERRIDE_CSS = f"""
/* {MARKER}: injected by patch-adk-devui-mobile.py -- see AGENTS.md.
   The bundled ADK dev UI is a desktop tool that sizes itself with 100vh and
   hides page overflow, so on a phone the chat composer sits below the fold with
   no way to scroll to it. 100dvh tracks the dynamic viewport, which is what
   makes the URL bar and the on-screen keyboard behave. */
@supports (height: 100dvh) {{
  html, body {{ height: 100dvh; }}

  /* The root component's host element. Its own rule is [_nghost-<hash>], which
     is rebuilt per ADK release, so this is matched by tag name instead. */
  app-root {{
    height: 100dvh !important;
    max-height: 100dvh !important;
  }}

  /* The composer is the last row of the shell's flex column, so the home
     indicator lands on top of it without this. Inert unless the viewport meta
     also carries viewport-fit=cover, which this script adds. */
  .chat-input-container {{
    padding-bottom: calc(20px + env(safe-area-inset-bottom, 0px)) !important;
  }}

  /* Hard 400px, which overflows a 390px phone. */
  .assistant-panel {{
    width: min(400px, 100vw) !important;
    height: calc(100dvh - 72px) !important;
  }}
}}
"""


def find_default_target() -> Path | None:
    """Locate the bundled index.html inside the project's virtualenv."""
    here = Path(__file__).resolve().parent
    candidates = sorted(
        here.glob(
            ".venv/lib/python*/site-packages/google/adk/cli/browser/index.html"
        )
    )
    if not candidates:
        candidates = sorted(
            here.glob(
                "**/site-packages/google/adk/cli/browser/index.html"
            )
        )
    return candidates[0] if candidates else None


def patch(text: str) -> tuple[str, list[str]]:
    """Return the patched HTML and a list of what changed."""
    if MARKER in text:
        return text, []

    missing = [needle for needle in EXPECTED if needle not in text]
    if missing:
        raise SystemExit(
            "patch-adk-devui-mobile: this does not look like the ADK dev UI.\n"
            f"  missing: {', '.join(repr(m) for m in missing)}\n"
            "  ADK has probably been upgraded. Re-check the patch against the new\n"
            "  bundle before relaxing this check -- a patch that silently does\n"
            "  nothing is worse than a failed build."
        )

    if text.count("</head>") != 1:
        raise SystemExit(
            "patch-adk-devui-mobile: expected exactly one </head>, found "
            f"{text.count('</head>')}. Refusing to guess where to inject."
        )

    changes: list[str] = []

    # viewport-fit=cover: without it env(safe-area-inset-*) is 0, so the CSS
    # override for the home indicator would be inert.
    match = VIEWPORT_META.search(text)
    if match is None:
        raise SystemExit(
            "patch-adk-devui-mobile: no viewport meta tag found. The safe-area "
            "fix depends on adding viewport-fit=cover to it."
        )
    content = match.group("content")
    additions = [
        param
        for param, present in (
            ("viewport-fit=cover", "viewport-fit" in content),
            # Chromium-only, but harmless elsewhere and decisive on Android.
            ("interactive-widget=resizes-content", "interactive-widget" in content),
        )
        if not present
    ]
    if additions:
        new_content = f"{content.rstrip().rstrip(',')}, {', '.join(additions)}"
        text = text[: match.start("content")] + new_content + text[match.end("content") :]
        changes.append(f"viewport meta: added {', '.join(additions)}")

    text = text.replace("</head>", f"{OVERRIDE_CSS}</head>", 1)
    changes.append("injected the 100dvh / safe-area override stylesheet")

    return text, changes


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else find_default_target()
    if target is None:
        print(
            "patch-adk-devui-mobile: could not find the ADK browser bundle. Run "
            "`uv sync` first, or pass the path to index.html as an argument.",
            file=sys.stderr,
        )
        return 1
    if not target.is_file():
        print(f"patch-adk-devui-mobile: no such file: {target}", file=sys.stderr)
        return 1

    original = target.read_text(encoding="utf-8")
    patched, changes = patch(original)

    if not changes:
        print(f"patch-adk-devui-mobile: {target} already patched, nothing to do")
        return 0

    target.write_text(patched, encoding="utf-8")
    print(f"patch-adk-devui-mobile: patched {target}")
    for change in changes:
        print(f"  - {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
