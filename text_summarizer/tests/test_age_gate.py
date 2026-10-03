"""The age-gate path: detection, the attestation it needs, and what happens when
the detection is wrong.

The shape of this is the one the rest of ``web_search`` uses and the one this
module's tests exist to protect: an error is *data*, because the model can read a
failure and fall through, whereas an exception ends the turn. So every refusal
below is a JSON payload the model is meant to act on -- here, by asking the user.
"""

from __future__ import annotations

import json
import re
import types

import pytest
from text_summarizer import ask_user as au
from text_summarizer import web_search as ws

GATE_TEXT = (
    "FILME COM CLASSIFICAÇÃO INDICATIVA - 18 ANOS\n"
    "A entrada será permitida apenas para: Pessoas com idade igual ou superior a "
    "18 anos.\nEstou ciente, quero continuar"
)


@pytest.fixture(autouse=True)
def _renderer_configured(monkeypatch):
    """`dismiss_page_text` refuses before it dials, so the seam needs a URL.

    Not a reachable one on purpose: every test here either patches `_post_dismiss`
    or is refused before the HTTP call, and a test that quietly dialled out would
    have a result that depended on the network.
    """
    monkeypatch.setenv("RENDERER_URL", "http://renderer.invalid:8080")


def _ctx(attested: bool = False) -> types.SimpleNamespace:
    """A tool_context whose session state says what the user did."""
    return types.SimpleNamespace(state={ws.ATTESTATION_STATE_KEY: True} if attested else {})


# --- detection ---------------------------------------------------------------


def test_the_gate_is_recognised_in_the_language_it_actually_appears_in():
    """The gate that motivated this is Portuguese, so an English-only list misses it.

    Measured on Ingresso.com's checkout: the notice is ``classificação
    indicativa`` / ``18 anos`` / ``entrada será permitida``, and a list that only
    knew English would have read it as an ordinary page.
    """
    assert ws._age_gate_markers(GATE_TEXT)
    assert ws._age_gate_markers("Age restriction: adults only.")


def test_an_ordinary_page_is_not_a_gate():
    """A false positive is not free -- it costs the user a question.

    This is the control for the test above. Without it, a list that matched
    everything would pass, and the failure mode is an agent that asks every user to
    confirm their age before reading a film listing.
    """
    clean = (
        "Cinemark Osasco - SALA 6\nSAB 03/10 15:20\n"
        "A B C D E F G H\nL L K K J J I I H H G G F\n"
        "Comprar ingresso R$ 30,00"
    )
    assert ws._age_gate_markers(clean) == []


# --- the attestation is enforced, not trusted ---------------------------------


def test_a_model_cannot_attest_on_its_own():
    """``attested=True`` with nothing behind it is refused.

    The reason this cannot be a docstring convention: a model that reads the
    refusal above learns exactly what ``attested=True`` does, so a boolean the
    model supplies is a boolean the model supplies. Only a recorded human answer
    turns it into something the code acts on.
    """
    out = json.loads(ws.web_fetch("https://example.test/", tool_context=_ctx(False), attested=True))
    assert out.get("error")
    assert "ask_user" in out["error"]
    # And nothing was fetched, so the refusal cannot have come *after* the page
    # was already read -- which would be an attestation after the fact.
    assert "url" not in out or "example.test" not in json.dumps(out.get("text", ""))


def test_a_recorded_answer_does_allow_the_fetch():
    """The other direction, because a guard that refuses everything is
    indistinguishable from a working one -- and this one costs the user the page."""
    assert ws._attestation_is_recorded(_ctx(True))
    assert not ws._attestation_is_recorded(_ctx(False))
    assert not ws._attestation_is_recorded(types.SimpleNamespace())


def test_only_a_real_user_answer_records_the_attestation():
    """Three of ask_user's four outcomes are the agent choosing for itself.

    Declined, unrecognised and no-context are all a value the model picked; only
    the branch where a human picked one counts, and that is the whole enforcement
    in one place. Each case below is a way the model could otherwise reach a gated
    page without anyone being asked.
    """
    def ask(confirmed, choice):
        """One call, returning both the answer and the context it happened on.

        The context has to come back too: the attestation is written into the
        session state, so a test that built a fresh one to look inside would be
        inspecting an empty dict and passing for the wrong reason.
        """
        ctx = _ctx()
        ctx.tool_confirmation = types.SimpleNamespace(
            confirmed=confirmed, payload={"choice": choice}
        )
        result = au.ask_user(
            question="Are you 18 or over?",
            options=["yes", "no"],
            default="no",
            consequence="the page stays unread",
            tool_context=ctx,
        )
        return result, ctx

    for confirmed, choice, label in (
        (False, "yes", "declined"),
        (True, "not one of them", "unrecognised answer"),
    ):
        result, ctx = ask(confirmed, choice)
        assert result["answered_by"] != "user", label
        assert not ws._attestation_is_recorded(ctx), (
            "a value the model chose must never count as a human answer"
        )

    # And the fourth: no framework to pause on at all.
    au.ask_user(
        question="Are you 18 or over?",
        options=["yes", "no"],
        default="no",
        consequence="the page stays unread",
        tool_context=None,
    )
    assert not ws._attestation_is_recorded(_ctx())

    # Only the real answer records it.
    result, ctx = ask(True, "yes")
    assert result["answered_by"] == "user"
    assert ws._attestation_is_recorded(ctx)


# --- the gate reaches the model as an instruction -----------------------------


def test_a_gated_page_still_returns_its_text():
    """The page is returned *whole*, with the gate named alongside it.

    This is the correction, and it is measured rather than argued. Ingresso.com's
    gate overlay is visual: ``document.body.innerText`` reads through it, so a plain
    render already carries the session, the room and the seat map (1,289 chars with
    ``SALA`` and ``Cinemark`` in it). Refusing on the markers alone hid a page the
    model could read and cost the user a question nobody needed to answer.

    So the advisory is *added to* the text, never substituted for it.
    """
    out = ws._gate_advisory("Cinemark Osasco - SALA 6 - 19:30", ["18 anos"])
    assert "SALA 6" in out and "19:30" in out, "the content must survive"
    assert "age or eligibility gate" in out


def test_the_advisory_tells_the_model_to_use_what_is_there_first():
    """Asking is conditional, because asking unconditionally is the bug.

    The model is the only party that can tell whether the page is complete for what
    it was asked -- the marker list cannot. So the instruction is "use what is
    there, and ask only if something is missing".
    """
    out = ws._gate_advisory("content", ["18 anos"])
    assert "may already be complete" in out
    assert "Only if something you need is MISSING" in out
    assert "ask_user" in out
    assert "answered_by: user" in out


def test_the_advisory_names_the_markers_it_matched():
    """No unsubstituted placeholder, and the evidence is in the sentence.

    Found live: an earlier version reached the model as "...gate (%s), so what is
    underneath...", which says nothing about *which* gate and reads as a bug in the
    agent rather than a gate.
    """
    out = ws._gate_advisory("content", ["classificação indicativa", "18 anos"])
    assert "%s" not in out and "%(" not in out
    assert "classificação indicativa" in out


def test_an_ordinary_page_gets_no_advisory():
    """The control for all three above.

    Without it, a `_gate_advisory` that fired unconditionally would pass, and the
    symptom would be an agent that tells the user about an age gate on a page with
    no gate on it.
    """
    assert ws._age_gate_markers("Cinemark Osasco - SALA 6 - 19:30 - R$ 30,00") == []


def test_the_marker_list_agrees_with_the_renderers():
    """Two processes, two copies of the vocabulary, one test holding them together.

    The agent and the renderer cannot share a module -- one runs pages a stranger
    wrote, the other holds the vault's keys -- so the list is duplicated on
    purpose. This is what stops the duplication from becoming a drift, and it fails
    the moment either side gains a phrase the other does not have.
    """
    renderer = (
        __import__("pathlib").Path("/home/vboxuser/Downloads/apps/chromium/renderer/renderer.py")
    )
    if not renderer.exists():  # pragma: no cover - the renderer is a separate repo
        pytest.skip("the renderer is not checked out here")
    src = renderer.read_text(encoding="utf-8")
    start = src.index("AGE_MARKERS = (")
    # Bounded at the tuple's own closing paren, not the end of the file: an
    # unbounded slice picks up every other indented string in the module, which
    # makes this test fail on things that have nothing to do with it.
    block = src[start : src.index("\n)\n", start)]
    renderer_markers = set(re.findall(r'^\s+"([^"]+)",', block, re.M))

    assert renderer_markers == set(ws.AGE_GATE_MARKERS), (
        "the two gate lists have drifted:\n"
        f"  only in the agent:   {sorted(set(ws.AGE_GATE_MARKERS) - renderer_markers)}\n"
        f"  only in the renderer: {sorted(renderer_markers - set(ws.AGE_GATE_MARKERS))}"
    )


# --- a wrong guess must not strand the page -----------------------------------


def test_nothing_to_dismiss_falls_back_to_reading(monkeypatch):
    """The gate call can be wrong, and being wrong must not cost the user the page.

    A false-positive detection sends the model to ask a question nobody needs to
    answer; if the *second* call then failed, the agent would have burned a
    question and produced nothing. So "there was no overlay" is answered by reading
    the page the ordinary way.
    """
    monkeypatch.setattr(
        ws,
        "_post_dismiss",
        lambda *a, **k: {"error": "nothing to dismiss", "detail": "no element matched"},
    )
    monkeypatch.setattr(ws, "render_page_text", lambda url, n: "the page, read normally")
    assert ws.dismiss_page_text("https://example.test/", 1000, attested=True) == (
        "the page, read normally"
    )


def test_a_disabled_primitive_falls_back_to_reading(monkeypatch):
    """A deployment that has not opted in must not lose the ability to read a page.

    The dismiss primitive is a *write* and is off unless an operator turns it on.
    If "disabled" ended the turn, enabling the age-gate feature would be a
    regression on every gated page for anyone who had not configured it.
    """
    monkeypatch.setattr(
        ws,
        "_post_dismiss",
        lambda *a, **k: {"error": "disabled", "detail": "set RENDERER_DISMISS_HOSTS"},
    )
    monkeypatch.setattr(ws, "render_page_text", lambda url, n: "the page, read normally")
    assert ws.dismiss_page_text("https://example.test/", 1000, attested=True) == (
        "the page, read normally"
    )


def test_a_real_refusal_is_not_swallowed_by_the_fallback(monkeypatch):
    """The control for the two tests above.

    Both of those would pass against a version that treated *every* refusal as
    "just read it instead", which would quietly turn the age gate back off -- the
    single failure this feature exists to prevent. So the refusals that matter have
    to still reach the model.
    """
    for reason, detail in (
        ("age gate needs the user", "ask the user with ask_user"),
        ("host not allowed", "ingresso.com is not in RENDERER_DISMISS_HOSTS"),
    ):
        monkeypatch.setattr(
            ws, "_post_dismiss", lambda *a, _r=reason, _d=detail, **k: {"error": _r, "detail": _d}
        )
        with pytest.raises(ValueError) as caught:
            ws.dismiss_page_text("https://ingresso.com/", 1000, attested=True)
        assert detail in str(caught.value)