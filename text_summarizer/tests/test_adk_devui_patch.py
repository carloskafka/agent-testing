"""The mobile patch applied to the bundled ADK dev UI.

Two things are worth guarding, and they are different in kind:

1. **The patch does what it says.** Pure unit tests over a synthetic
   ``index.html`` -- idempotency, both viewport meta additions, injection point,
   and refusing to touch a file it does not recognise.

2. **The patch still applies to the real bundle.** ``google-adk`` ships a
   prebuilt Angular app, and this project patches a file inside the installed
   wheel. The next ``uv sync`` can replace that file wholesale. If it does, the
   overrides keyed on ``.chat-input-container`` and ``.assistant-panel`` would
   silently stop matching and the fix would quietly do nothing -- no error, no
   failing test, just a broken composer on a phone again.

   So the assumptions the patch makes are asserted against the *installed*
   package. An ADK upgrade that invalidates them fails here, in seconds, with a
   clear message, rather than shipping.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHER = REPO_ROOT / "patch-adk-devui-mobile.py"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("patch_adk_devui", PATCHER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


patcher = _load_patcher()


def _fake_index(**overrides) -> str:
    """A minimal stand-in with the two strings the patch keys on."""
    meta = overrides.get(
        "meta", '<meta name="viewport" content="width=device-width, initial-scale=1">'
    )
    body_rule = overrides.get("body_rule", "body{height:100vh;margin:0;overflow:hidden}")
    return (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        f"{meta}"
        f"<style>{body_rule}</style>"
        "</head><body><app-root></app-root></body></html>"
    )


# --- the patch itself ---------------------------------------------------------


def test_the_patch_adds_both_viewport_parameters():
    """Neither can be done in CSS, and the safe-area CSS is inert without one.

    ``viewport-fit=cover`` is what makes ``env(safe-area-inset-bottom)`` non-zero
    on a notched device; ``interactive-widget=resizes-content`` is what stops
    Chromium's keyboard from covering the composer.
    """
    patched, changes = patcher.patch(_fake_index())
    assert 'name="viewport"' in patched
    assert "viewport-fit=cover" in patched
    assert "interactive-widget=resizes-content" in patched
    assert any("viewport meta" in c for c in changes)


def test_the_patch_keeps_the_original_viewport_parameters():
    """It adds to the meta, it does not replace it."""
    patched, _ = patcher.patch(_fake_index())
    meta = patcher.VIEWPORT_META.search(patched).group("content")
    assert "width=device-width" in meta
    assert "initial-scale=1" in meta


def test_the_patch_injects_100dvh_and_safe_area_handling():
    patched, _ = patcher.patch(_fake_index())
    assert "100dvh" in patched
    assert "env(safe-area-inset-bottom" in patched
    # Guarded by @supports so a browser without dvh keeps vendor behaviour.
    assert "@supports (height: 100dvh)" in patched


def test_the_patch_targets_the_composer_and_the_overflowing_panel():
    """These are the two elements the report was actually about."""
    patched, _ = patcher.patch(_fake_index())
    assert ".chat-input-container" in patched
    assert ".assistant-panel" in patched
    # app-root is stable across ADK builds, unlike the hashed _nghost selectors.
    assert "app-root" in patched


def test_the_patch_is_injected_before_head_closes():
    patched, _ = patcher.patch(_fake_index())
    assert patched.index(patcher.MARKER) < patched.index("</head>")
    assert patched.count("</head>") == 1


def test_the_patch_does_not_rewrite_vendor_css():
    """It injects an override rather than editing the shipped rule.

    Rewriting ``body{height:100vh}`` in place would be shorter but would make
    the vendor string unrecoverable, so removing the patch would mean
    reinstalling the wheel to undo it.
    """
    original = _fake_index()
    patched, _ = patcher.patch(original)
    assert "body{height:100vh;margin:0;overflow:hidden}" in patched


# --- idempotency and refusal --------------------------------------------------


def test_patching_twice_changes_nothing():
    once, first = patcher.patch(_fake_index())
    twice, second = patcher.patch(once)
    assert first and second == [], "second run must report no changes"
    assert twice == once, "second run must be a byte-for-byte no-op"
    assert twice.count(patcher.MARKER) == 1


def test_a_file_without_the_expected_strings_is_refused():
    """Fails the Docker build rather than shipping a patch that does nothing."""
    with pytest.raises(SystemExit) as excinfo:
        patcher.patch("<html><head></head><body>something else</body></html>")
    message = str(excinfo.value)
    assert "does not look like the ADK dev UI" in message
    assert "upgraded" in message


def test_a_file_with_no_viewport_meta_is_refused():
    """The safe-area CSS is pointless without viewport-fit=cover."""
    html = _fake_index().replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1">', ""
    )
    with pytest.raises(SystemExit, match="no viewport meta tag"):
        patcher.patch(html)


def test_ambiguous_injection_point_is_refused():
    """Two </head> closers: an SDK upgrade that restructures the document.

    Injecting into the first of two would be a coin flip, so the patch stops.
    """
    doubled = _fake_index().replace("</head>", "</head></head>", 1)
    with pytest.raises(SystemExit, match="exactly one </head>"):
        patcher.patch(doubled)


def test_a_vendor_upgrade_that_drops_the_body_rule_is_caught():
    """The exact failure mode of an ADK upgrade, simulated."""
    with pytest.raises(SystemExit, match="does not look like"):
        patcher.patch(_fake_index(body_rule="body{margin:0;overflow:hidden}"))


# --- against the actually installed package -----------------------------------


def _installed_browser_dir() -> Path:
    """The real bundle inside the venv, wherever the interpreter lives."""
    import google.adk

    return Path(google.adk.__file__).parent / "cli" / "browser"


def test_the_real_adk_bundle_is_the_one_we_patch():
    """Locate the bundle the Docker RUN step will find."""
    browser = _installed_browser_dir()
    index = browser / "index.html"
    assert index.is_file(), (
        f"ADK dev UI bundle not found at {index}. The Docker patch step would "
        "have nothing to patch -- check the google-adk version."
    )


@pytest.mark.parametrize("needle", patcher.EXPECTED)
def test_the_real_adk_bundle_still_contains_what_the_patch_assumes(needle):
    """The guard that makes an ADK upgrade visible.

    If google-adk is upgraded and the bundle stops matching, the CSS overrides
    for the composer would silently stop applying. This fails in seconds instead
    of the fix quietly regressing on someone's phone.
    """
    index = _installed_browser_dir() / "index.html"
    if not index.is_file():
        pytest.skip("ADK dev UI bundle not present in this environment")
    text = index.read_text(encoding="utf-8")
    assert needle in text, (
        f"google-adk's dev UI no longer contains {needle!r}. "
        "patch-adk-devui-mobile.py needs re-checking against the new bundle -- "
        "run it against the installed index.html and reconcile."
    )


def test_the_real_bundle_patches_cleanly():
    """End to end on the real file: no exceptions, changes reported, idempotent."""
    index = _installed_browser_dir() / "index.html"
    if not index.is_file():
        pytest.skip("ADK dev UI bundle not present in this environment")
    original = index.read_text(encoding="utf-8")
    try:
        patched, changes = patcher.patch(original)
        assert changes, "the real bundle should need patching"
        assert patcher.MARKER in patched
        assert "100dvh" in patched
        # Idempotent on the real file too.
        again, second = patcher.patch(patched)
        assert second == []
        assert again == patched
    finally:
        # Never write to the installed package from a test.
        pass
