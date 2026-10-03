#!/usr/bin/env python3
"""Give ADK tool confirmations clickable options, in place.

Why this exists
---------------
``ask_user`` asks the user to pick between options by requesting a tool
confirmation. ADK builds the ``adk_request_confirmation`` event and the bundled
dev UI draws it -- as a **checkbox, a read-only JSON dump, and a Submit
button**. ``ToolConfirmation.payload`` is typed ``Optional[Any]``, arbitrary
JSON rather than a schema, and nothing in the bundle reads ``options`` out of
it. The whole of the UI's confirmation branch is::

    this.confirmationModel.confirmed = args.toolConfirmation.confirmed || false
    this.confirmationModel.payload    = JSON.stringify(args.originalFunctionCall.args)

Measured on session ``f0842db4-9d42-4911-8b34-255a3731021f``: asked *"Para qual
cidade voce deseja ver os filmes disponiveis?"* with options ``["Osasco",
"Campinas"]``, the user was shown that array as JSON text inside a code viewer
and had to infer that it was a menu.

There is no first-class way to fix that from outside the app. ``confirmationModel``
is an Angular component property with no DOM handle, and the confirmation view has
**no payload textarea** -- the textarea belongs to the form-field tab, which a
confirmation never uses -- so there is nothing to type a choice into. So the
response-building expression in the bundle is rewritten.

What it does
------------
Two halves, deliberately in two files.

``adk-confirm-options.js`` (this repo's, node-testable) renders ``options`` as
buttons. It learns what to render by observing the ``/run`` responses the app
already receives, so it uses the payload ADK actually sent rather than whatever
Angular happened to render -- the DOM-shaped version of this is guesswork.

This script makes the click mean something. It rewrites the one expression that
builds the confirmation response so it prefers an explicit choice::

    let a={confirmed:this.confirmationModel.confirmed,payload:o};
    ->
    let c=window.__adkConfirmChoice&&window.__adkConfirmChoice[this.functionCall.id];
    let a={confirmed:c?true:this.confirmationModel.confirmed,payload:c||o};

``confirmed`` is forced true for an explicit choice because a button click *is* a
confirmation, whereas a bare Submit on the stock UI means "I did not choose" --
which is exactly what ``ask_user`` has to be able to tell apart, since it reports
``answered_by: "default_declined"`` for the second and would otherwise have no way
to say the user actually picked something.

Then the script is inlined into ``index.html`` before ``</body>``.

Guards
------
Rewriting minified vendor JS is the thing this project does least, so it is made
to fail loudly rather than quietly:

* the anchor must occur **exactly once** in the bundle. Zero means ADK changed the
  method; more than one means a guess, and a guess about which of two identical
  expressions builds a user's confirmation answer is not a guess worth making.
* ``EXPECTED`` strings must be present in ``index.html``.
* the whole thing is idempotent (one marker comment) and never rewrites vendor
  rules -- it *adds* an alternative and leaves the original expression intact as
  the fallback, so removing this patch restores stock behaviour rather than
  corrupting the bundle.

**Not verified:** no browser was available to confirm the buttons render and the
click round-trips. What is verified is that the served page carries the patch, the
anchor is unique, the replacement is present, the shim is inlined exactly once, and
the shim's own logic passes its node tests against the real event shapes. The
round trip through a real click is the part to check on a screen.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Identifies our own output, so re-running is a no-op rather than a second copy.
MARKER = "adk-confirm-fix"

#: Strings that must be present in index.html for the patch to mean anything.
EXPECTED = (
    "<app-root>",
    "</body>",
)

#: The expression the bundled dev UI builds its confirmation response with.
#:
#: Verified to occur exactly once in google-adk 2.9.2's ``main-*.js`` -- it sits
#: inside ``onSend``'s ``if (this.isConfirmationRequest)`` branch. The uniqueness
#: is re-checked at patch time rather than trusted from here, because that is the
#: property that makes the replacement safe and it is the one that a vendor bump
#: would break.
ANCHOR = "let a={confirmed:this.confirmationModel.confirmed,payload:o};"

#: What it becomes. Reads an explicit choice the shim left behind, and treats its
#: presence as a confirmation. Everything else is untouched, including the
#: ``payload:o`` fallback, so a card with no buttons still behaves exactly as
#: stock.
REPLACEMENT = (
    "let c=window.{choice_global}&&window.{choice_global}[this.functionCall.id];"
    "let a={{confirmed:c?!0:this.confirmationModel.confirmed,payload:c||o}};"
)

#: Must match ``CHOICE_GLOBAL`` in adk-confirm-options.js. Asserted in the tests,
#: because a rename on either side produces a shim that draws buttons and a bundle
#: that ignores them -- which looks exactly like the original bug.
CHOICE_GLOBAL = "__adkConfirmChoice"

_SCRIPT_SRC = Path(__file__).resolve().parent / "adk-confirm-options.js"


def _load_shim() -> str:
    try:
        source = _SCRIPT_SRC.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - only if the file is missing
        raise SystemExit(
            f"patch-adk-devui-confirm: cannot read {_SCRIPT_SRC}: {exc}\n"
            "  The shim and the bundle patch ship together; neither works alone."
        ) from exc
    if "</script" in source.lower():
        raise SystemExit(
            "patch-adk-devui-confirm: adk-confirm-options.js contains a literal "
            "'</script>' and would terminate the inline <script> early."
        )
    return source


def _bundle_candidates(browser_dir: Path) -> list[Path]:
    """Every bundle that could be the app's, newest-looking first."""
    return sorted(browser_dir.glob("main-*.js"), key=lambda p: p.stat().st_mtime, reverse=True)


def patch_bundle(text: str) -> tuple[str, list[str]]:
    """Teach the dev UI to honour an explicit option choice."""
    if MARKER in text:
        return text, []

    occurrences = text.count(ANCHOR)
    if occurrences == 0:
        raise SystemExit(
            "patch-adk-devui-confirm: the bundle no longer contains the confirmation "
            f"response expression.\n  expected exactly once: {ANCHOR!r}\n"
            "  google-adk has probably been upgraded. Read the new `onSend` and "
            "reconcile this patch against it -- a patch that silently does nothing "
            "is worse than a failed build."
        )
    if occurrences > 1:
        raise SystemExit(
            "patch-adk-devui-confirm: the anchor occurs "
            f"{occurrences} times, so the replacement would be a guess.\n"
            "  Re-check which one builds a confirmation response before relaxing "
            "this."
        )

    patched = text.replace(
        ANCHOR,
        "/* %s */%s"
        % (MARKER, REPLACEMENT.format(choice_global=CHOICE_GLOBAL)),
        1,
    )
    return patched, [
        "onSend now prefers window.%s[<call id>] over the prefilled payload"
        % CHOICE_GLOBAL
    ]


#: The inlined block, for recognising and replacing a previous injection.
#:
#: Non-greedy to ``</script>`` rather than to the end of the document, because the
#: shim is the whole body and the page has markup after it.
_SCRIPT_BLOCK = re.compile(
    r"<script>\s*/\*\s*{marker}\b.*?</script>\s*".format(marker=re.escape(MARKER)),
    re.DOTALL,
)


def _script_block(shim: str) -> str:
    return (
        "<script>\n/* %s: injected by patch-adk-devui-confirm.py -- see the script "
        "and AGENTS.md. */\n%s\n</script>\n" % (MARKER, shim.rstrip())
    )


def patch_index(html: str, shim: str) -> tuple[str, list[str]]:
    """Inline the shim, so no extra route has to serve it.

    Idempotent by *content*, not by marker. An earlier version returned early on
    "the marker is already there", which meant a fixed shim could never reach a
    page that already carried the broken one: re-running the patcher reported
    "already patched, nothing to do" while the served page still had the old
    bytes. That is the same failure as a stale build artifact -- it looks done and
    is not -- and it is invisible to every test, because every test starts from a
    page with no marker at all.

    So a second run compares the shim actually on the page against the one on
    disk and rewrites the block when they differ. Only a byte-identical page is a
    no-op.
    """
    block = _script_block(shim)

    existing = _SCRIPT_BLOCK.search(html)
    if existing is not None:
        if existing.group(0) == block:
            return html, []
        patched = html[: existing.start()] + block + html[existing.end() :]
        return patched, [
            "replaced the inlined shim with the current adk-confirm-options.js "
            "(the page carried an older copy)"
        ]

    missing = [needle for needle in EXPECTED if needle not in html]
    if missing:
        raise SystemExit(
            "patch-adk-devui-confirm: this does not look like the ADK dev UI.\n"
            f"  missing: {', '.join(repr(m) for m in missing)}\n"
            "  ADK has probably been upgraded. Re-check the patch against the new\n"
            "  bundle before relaxing this check."
        )

    if html.count("</body>") != 1:
        raise SystemExit(
            "patch-adk-devui-confirm: expected exactly one </body>, found "
            f"{html.count('</body>')}. Refusing to guess where to inject."
        )

    patched = html.replace("</body>", block + "</body>", 1)
    return patched, ["inlined adk-confirm-options.js before </body>"]


def find_default_browser_dir() -> Path | None:
    """The bundled dev UI inside the project's virtualenv."""
    here = Path(__file__).resolve().parent
    for pattern in (
        "text_summarizer/.venv/lib/python*/site-packages/google/adk/cli/browser",
        ".venv/lib/python*/site-packages/google/adk/cli/browser",
        "**/site-packages/google/adk/cli/browser",
    ):
        candidates = sorted(here.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Patch the bundled ADK dev UI to render confirmations as options."
    )
    parser.add_argument(
        "--browser-dir",
        type=Path,
        default=None,
        help="directory holding index.html and main-*.js (default: inside .venv)",
    )
    args = parser.parse_args(argv[1:])

    browser_dir = args.browser_dir or find_default_browser_dir()
    if browser_dir is None:
        print(
            "patch-adk-devui-confirm: could not find the ADK browser bundle. Run "
            "`uv sync` first, or pass --browser-dir.",
            file=sys.stderr,
        )
        return 1

    index = browser_dir / "index.html"
    bundles = _bundle_candidates(browser_dir)
    if not index.is_file():
        print(f"patch-adk-devui-confirm: no such file: {index}", file=sys.stderr)
        return 1
    if not bundles:
        print(
            f"patch-adk-devui-confirm: no main-*.js in {browser_dir}",
            file=sys.stderr,
        )
        return 1

    shim = _load_shim()
    changes: list[str] = []

    for bundle in bundles:
        original = bundle.read_text(encoding="utf-8")
        patched, bundle_changes = patch_bundle(original)
        if bundle_changes:
            bundle.write_text(patched, encoding="utf-8")
            changes.append(f"{bundle.name}: " + "; ".join(bundle_changes))

    original_index = index.read_text(encoding="utf-8")
    patched_index, index_changes = patch_index(original_index, shim)
    if index_changes:
        index.write_text(patched_index, encoding="utf-8")
        changes.extend(index_changes)

    if not changes:
        print(f"patch-adk-devui-confirm: {browser_dir} already patched, nothing to do")
        return 0

    print(f"patch-adk-devui-confirm: patched {browser_dir}")
    for change in changes:
        print(f"  - {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))