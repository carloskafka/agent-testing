"""The confirmation-options patch: the bundle rewrite, and the shim it enables.

Two halves, and they can each fail while the other works, so they are guarded
separately.

**The bundle rewrite** replaces one expression in the bundled dev UI -- the one
that builds a tool confirmation's response -- so that an explicit choice beats the
prefilled payload. That is unavoidable from outside the app: ``confirmationModel``
is an Angular component property with no DOM handle, and the confirmation view has
no payload textarea to type into. Being a rewrite of minified vendor code, it is
made to fail loudly instead: the anchor must be present **exactly once**.

**The shim** (``adk-confirm-options.js``) renders ``options`` as buttons. Its own
logic is tested by node, from this file, against a DOM stub small enough to be
read in one sitting.

The test that matters most is
``test_the_cross_half_check_fails_against_an_unpatched_bundle``. Without it every
other assertion here is compatible with a shim that draws buttons the bundle
ignores -- which is not a broken feature but a *silent* one: the user still sees
JSON text and clicks nothing, exactly as before the patch. That is the shape of
the original bug, so the gate has to be shown to notice it.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHER_PATH = REPO_ROOT / "patch-adk-devui-confirm.py"
SHIM_PATH = REPO_ROOT / "adk-confirm-options.js"
NODE_TEST_PATH = REPO_ROOT / "adk-confirm-options.test.js"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("patch_adk_devui_confirm", PATCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


patcher = _load_patcher()

#: A synthetic bundle containing the one expression the patch rewrites, inside a
#: plausible `onSend`. Kept faithful enough that the node test can slice `onSend`
#: out of it exactly as it does for the real file.
SYNTHETIC_BUNDLE = (
    "class hD{functionCall;confirmationModel={confirmed:!1,payload:'{}'};"
    "get isConfirmationRequest(){return this.functionCall?.name==='adk_request_confirmation'}"
    "onSend(){if(this.isConfirmationRequest){let o={};"
    "try{o=JSON.parse(this.confirmationModel.payload)}catch(s){o={}}"
    + patcher.ANCHOR
    + "this.functionCall.responseStatus='sent';return a;}}"
)


def _run_node(bundle_path: Path | None) -> subprocess.CompletedProcess:
    argv = ["node", str(NODE_TEST_PATH)]
    if bundle_path is not None:
        argv.append(str(bundle_path))
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)


node_required = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is not installed; adk-confirm-options.js cannot be tested here",
)


# --- the bundle rewrite ------------------------------------------------------


def test_the_anchor_is_rewritten_and_the_vendor_expression_survives():
    """It *adds* an alternative and keeps the original as the fallback.

    Deleting the stock expression outright would make a card the shim did not draw
    -- a confirmation from some other tool -- send ``payload: undefined``. Leaving
    it as the ``||`` fallback is what keeps the patch additive, and what makes
    removing it a matter of reinstalling the wheel rather than repairing a bundle.
    """
    patched, changes = patcher.patch_bundle(SYNTHETIC_BUNDLE)
    assert changes, "the synthetic bundle should need patching"
    assert patcher.ANCHOR not in patched, "the stock expression must not be deleted"
    assert "payload:c||o" in patched, "the prefilled payload must remain the fallback"
    assert patcher.MARKER in patched
    assert "isConfirmationRequest" in patched, "vendor code outside the anchor is untouched"


def test_the_rewrite_forces_confirmed_only_when_a_choice_was_made():
    """A button click is a confirmation; a bare Submit is not.

    ``ask_user`` reports ``answered_by: "default_declined"`` for the second and
    ``"user"`` for the first, so collapsing them would make the provenance of
    every answer wrong. Hence ``c?!0:...`` rather than an unconditional ``!0``.
    """
    replacement = patcher.REPLACEMENT.format(choice_global=patcher.CHOICE_GLOBAL)
    assert "confirmed:c?!0:this.confirmationModel.confirmed" in replacement, (
        "a choice must confirm, and the absence of one must leave the checkbox alone"
    )
    assert "payload:c||o" in replacement


def test_a_bundle_without_the_anchor_is_refused():
    """An ADK upgrade, simulated. Fails the Docker build instead of shipping a
    patch that silently does nothing."""
    with pytest.raises(SystemExit) as excinfo:
        patcher.patch_bundle("class hD{onSend(){/* rewritten upstream */}}")
    message = str(excinfo.value)
    assert "no longer contains" in message
    assert "upgraded" in message


def test_an_ambiguous_anchor_is_refused():
    """Two copies of the expression means the replacement is a guess -- and a guess
    about which one builds a user's confirmation answer is not worth making."""
    with pytest.raises(SystemExit, match="so the replacement would be a guess"):
        patcher.patch_bundle(patcher.ANCHOR + "x" + patcher.ANCHOR)


def test_patching_the_bundle_twice_changes_nothing():
    once, first = patcher.patch_bundle(SYNTHETIC_BUNDLE)
    twice, second = patcher.patch_bundle(once)
    assert first and second == [], "second run must report no changes"
    assert twice == once, "second run must be a byte-for-byte no-op"
    assert twice.count(patcher.MARKER) == 1


# --- the shim, and the two halves agreeing -----------------------------------


def test_the_shim_and_the_bundle_patch_name_the_same_global():
    """A rename on either side produces buttons that do nothing.

    The shim writes the choice; the bundle reads it; neither imports the other.
    Nothing but this assertion connects them, and the failure mode is silent --
    buttons render, clicks register, and the answer arrives as a default anyway.
    """
    shim = SHIM_PATH.read_text(encoding="utf-8")
    assert f"var CHOICE_GLOBAL = '{patcher.CHOICE_GLOBAL}';" in shim, (
        "adk-confirm-options.js and patch-adk-devui-confirm.py disagree about the "
        "global the choice is read from"
    )


def test_the_shim_knows_the_bundle_s_checkbox_id_scheme():
    """The shim finds the card by the checkbox's id, so ADK renaming that scheme
    means nothing renders -- with no error to notice it by."""
    shim = SHIM_PATH.read_text(encoding="utf-8")
    assert "'confirmed-checkbox-standalone-'" in shim


def test_a_shim_that_could_close_the_script_tag_is_refused(tmp_path, monkeypatch):
    """Inlined into ``<script>``, a literal ``</script>`` ends the block early and
    the rest of the file becomes visible page text.

    The shim does not contain one, so this is not a test of the shim as written --
    it is a test that the guard exists, against the shape of input that would make
    inlining unsafe. A shim is edited far more often than a patcher is.
    """
    hostile = tmp_path / "adk-confirm-options.js"
    hostile.write_text("var x = 1;\n</script><b>oops</b>\n", encoding="utf-8")
    monkeypatch.setattr(patcher, "_SCRIPT_SRC", hostile)

    with pytest.raises(SystemExit, match="terminate the inline <script>"):
        patcher._load_shim()


def test_a_missing_shim_says_so_rather_than_silently_patching(tmp_path, monkeypatch):
    """The two halves ship together; one without the other is not a smaller patch.

    Without this the script would read a missing file, raise, and leave the bundle
    rewritten and the page unable to produce a choice -- one half working, one
    half silently absent.
    """
    monkeypatch.setattr(patcher, "_SCRIPT_SRC", tmp_path / "not-here.js")
    with pytest.raises(SystemExit, match="neither works alone"):
        patcher._load_shim()


# --- index.html --------------------------------------------------------------


def _fake_index() -> str:
    return (
        "<!DOCTYPE html><html><head><title>ADK</title></head>"
        "<body><app-root></app-root></body></html>"
    )


def test_the_shim_is_inlined_before_the_body_closes():
    patched, changes = patcher.patch_index(_fake_index(), "var shim = 1;")
    assert changes
    assert patched.index(patcher.MARKER) < patched.index("</body>")
    assert patched.count("<script>") == 1
    assert patched.count(patcher.MARKER) == 1


def test_patching_index_twice_changes_nothing():
    once, first = patcher.patch_index(_fake_index(), "var shim = 1;")
    twice, second = patcher.patch_index(once, "var shim = 1;")
    assert first and second == []
    assert twice == once


def test_a_fixed_shim_reaches_a_page_that_carries_the_old_one():
    """Idempotence by content, not by marker.

    The marker check this replaced returned early whenever ``adk-confirm-fix`` was
    present, so once a page had been patched, **a corrected shim could never
    reach it** — the patcher reported "already patched, nothing to do" while the
    served page still ran the old bytes. That is what shipped: the camelCase fix
    below could not be applied to the running container without editing the
    installed package by hand.

    And nothing tested it, because every other test starts from a page with no
    marker at all. This one starts from a page carrying a *different* shim, which is
    the state that is easy to create and easy not to test.
    """
    stale, _ = patcher.patch_index(_fake_index(), "var shim = 1; // the old one")
    fixed, changes = patcher.patch_index(stale, "var shim = 1; // the new one")

    assert changes, "a page with an outdated shim must be re-patched"
    assert "the new one" in fixed
    assert "the old one" not in fixed
    assert fixed.count(patcher.MARKER) == 1, "exactly one injected block"

    # ...and the replacement is itself stable.
    again, second = patcher.patch_index(fixed, "var shim = 1; // the new one")
    assert second == [] and again == fixed


def test_the_shim_that_actually_ships_is_what_the_page_gets():
    """Ties the two files together: the page must carry the shim on disk.

    Reading the shim from ``patch-adk-devui-confirm.py``'s own path rather than a
    fixture is the point. The camelCase defect was invisible because every test
    supplied a *string* to ``patch_index`` instead of the file, so nothing ever
    asked whether the thing that gets deployed is the thing that was tested.
    """
    shim = patcher._load_shim()
    patched, _ = patcher.patch_index(_fake_index(), shim)

    assert "part.functionCall" in patched, (
        "the deployed shim does not read the camelCase wire shape -- this is the "
        "defect that shipped once; see AGENTS.md"
    )


def test_a_document_without_the_expected_strings_is_refused():
    with pytest.raises(SystemExit, match="does not look like the ADK dev UI"):
        patcher.patch_index("<html><head></head><body>nothing</body></html>", "x")


def test_an_ambiguous_injection_point_is_refused():
    doubled = _fake_index().replace("</body>", "</body></body>", 1)
    with pytest.raises(SystemExit, match="exactly one </body>"):
        patcher.patch_index(doubled, "x")


# --- the node suite, and the gate that makes it mean something ---------------


@node_required
def test_the_shim_passes_its_node_suite():
    result = _run_node(bundle_path=None)
    assert result.returncode == 0, (
        "adk-confirm-options.js failed its own tests.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@node_required
def test_the_clicked_option_reaches_the_patched_bundle(tmp_path):
    """The cross-half check: a button click must produce the payload the patched
    bundle actually sends.

    The node test slices ``onSend`` out of the bundle it is given and runs it, so
    this asserts against the bytes the Docker build installs rather than against a
    re-typed copy of the expression, which would only ever test itself.
    """
    patched, _ = patcher.patch_bundle(SYNTHETIC_BUNDLE)
    bundle = tmp_path / "main-TEST.js"
    bundle.write_text(patched, encoding="utf-8")

    result = _run_node(bundle_path=bundle)
    assert result.returncode == 0, (
        "the shim and the bundle patch do not agree about what a button click "
        f"sends.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "# SKIP" not in result.stdout, (
        "the cross-half test was skipped, so this passed without checking anything"
    )


@node_required
def test_the_cross_half_check_fails_against_an_unpatched_bundle(tmp_path):
    """The control.

    Every assertion above is compatible with a shim that draws buttons the bundle
    ignores -- which is not a broken feature but a *silent* one, reproducing the
    original symptom exactly. So the cross-half check is run against an
    **unpatched** bundle and is required to fail.
    """
    bundle = tmp_path / "main-UNPATCHED.js"
    bundle.write_text(SYNTHETIC_BUNDLE, encoding="utf-8")

    result = _run_node(bundle_path=bundle)
    assert result.returncode != 0, (
        "the cross-half check passed against an unpatched bundle, so it is not "
        "testing the patch. Asserting 'a choice is honoured' cannot fail when "
        "nothing honours a choice -- it would have passed for the whole time the "
        "feature did not work."
    )
    assert "# fail 0" not in result.stdout, "the node suite reported no failures to explain this one"


# --- against the actually installed package ----------------------------------


def _installed_browser_dir() -> Path | None:
    try:
        import google.adk
    except ImportError:  # pragma: no cover
        return None
    candidate = Path(google.adk.__file__).parent / "cli" / "browser"
    return candidate if (candidate / "index.html").is_file() else None


def test_the_real_bundle_still_contains_what_the_patch_assumes():
    """The guard that makes an ADK upgrade visible in seconds rather than as a
    feature that quietly stopped working.

    The anchor must be present exactly once -- not merely present, because
    *uniqueness* is the property that makes the rewrite safe and the property a
    vendor bump is most likely to break by refactoring ``onSend``.
    """
    browser = _installed_browser_dir()
    if browser is None:
        pytest.skip("ADK dev UI bundle not present in this environment")
    bundles = patcher._bundle_candidates(browser)
    assert bundles, f"no main-*.js in {browser}"
    for bundle in bundles:
        text = bundle.read_text(encoding="utf-8")
        if patcher.MARKER in text:
            continue  # already patched by the Docker RUN step
        assert text.count(patcher.ANCHOR) == 1, (
            f"{bundle.name} contains the confirmation response expression "
            f"{text.count(patcher.ANCHOR)} times; patch-adk-devui-confirm.py needs "
            "re-checking against the new bundle."
        )


def test_the_real_bundle_patches_cleanly(tmp_path):
    """End to end on the real file: patches, reports a change, and is idempotent.

    Copies rather than edits: a test must never write into the installed package.
    """
    browser = _installed_browser_dir()
    if browser is None:
        pytest.skip("ADK dev UI bundle not present in this environment")

    bundles = patcher._bundle_candidates(browser)
    assert bundles, f"no main-*.js in {browser}"
    shim = patcher._load_shim()

    for bundle in bundles:
        original = bundle.read_text(encoding="utf-8")
        copy = tmp_path / bundle.name
        copy.write_text(original, encoding="utf-8")
        patched, changes = patcher.patch_bundle(original)
        # A bundle the Docker step already patched is legitimately a no-op.
        if not changes:
            assert patcher.MARKER in original
            continue
        assert patcher.MARKER in patched
        assert "payload:c||o" in patched, "the vendor fallback must survive"
        again, second = patcher.patch_bundle(patched)
        assert second == [] and again == patched

    index = browser / "index.html"
    index_copy = tmp_path / "index.html"
    index_copy.write_text(index.read_text(encoding="utf-8"), encoding="utf-8")
    patched_index, index_changes = patcher.patch_index(index.read_text(encoding="utf-8"), shim)
    if index_changes:
        assert patched_index.count(patcher.MARKER) == 1
        assert patched_index.index(patcher.MARKER) < patched_index.index("</body>")
    else:
        assert patcher.MARKER in index.read_text(encoding="utf-8")