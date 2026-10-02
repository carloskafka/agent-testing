"""Unit tests for the renderer fallback in ``web_fetch``.

A plain HTTP GET cannot execute JavaScript, so a page that builds its content with
it arrives as a shell. Measured on the page that set this up:

    ingresso.com/filmes?city=osasco    478 visible chars from a GET
                                      2024 visible chars from a browser

and the same page was 94,700 bytes of HTML yielding 183 characters before. So the
shape of the fix is "try the cheap thing, and if what came back is too thin to be
the page, pay for a browser".

These tests are about the *decision* and the error paths, not about the browser:
the renderer is a separate container that this module talks to over HTTP, and what
matters here is that the agent is never worse off for having it configured.

No network. ``httpx.post`` is stubbed, because the thing under test is "when does
this module decide to call out, and what does it do with a bad answer".
"""

from __future__ import annotations

import json

import pytest
from text_summarizer import web_search as ws


def _allow(monkeypatch):
    """Approve every URL, reporting one vetted address.

    The address is not decoration: ``fetch_page_text`` pins each hop to
    ``vetted[0]``, so a stub that approves without appending makes every fetch fail
    with "list index out of range" before it reaches the behaviour under test.
    """
    import ipaddress

    approved = ipaddress.ip_address("93.184.216.34")

    def fake(url, *, vetted=None, allow_private=False):
        if vetted is not None:
            vetted.append(approved)
        return ""

    monkeypatch.setattr(ws, "check_url", fake)


def _renderer_on(monkeypatch):
    """Point the tier at a renderer, which is what makes the fallback reachable."""
    monkeypatch.setenv(ws.RENDERER_URL_ENV, "http://renderer:8080")


def _page(body: bytes):
    """A 200 HTML response tuple, the shape ``_http_get`` returns."""
    return 200, "", ("text/html; charset=utf-8", body)


def _stub_get(monkeypatch, results):
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return results.pop(0) if results else _page(b"<html><body>default</body></html>")

    monkeypatch.setattr(ws, "_http_get", fake_get)
    return calls


class _FakeResponse:
    """The three things read off an httpx response here, and nothing else."""

    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        if self._payload is _NOT_JSON:
            raise ValueError("no json")
        return self._payload


_NOT_JSON = object()


def _stub_render(monkeypatch, payload, status_code=200, text=""):
    """Stub the POST to the renderer.

    ``web_search`` imports httpx inside the function that uses it, so there is no
    module attribute to patch; the real module is what gets replaced.
    """
    import httpx

    calls: list[dict] = []

    def fake_post(url, *, json=None, timeout=None, trust_env=None):
        calls.append({"url": url, "body": json})
        return _FakeResponse(payload, status_code, text)

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


# --- the threshold -------------------------------------------------------------


def test_a_page_with_substance_is_never_rendered(monkeypatch):
    """The cheap path stays cheap.

    Rendering costs a second container, a browser launch and ~2s. A page that
    already answered in prose must not pay that, or the fallback would make the
    ordinary case the expensive one.
    """
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body>" + b"real prose. " * 200 + b"</body></html>")])
    renders = _stub_render(monkeypatch, {"text": "should never be asked for"})

    ws.web_fetch("https://example.com/")

    assert renders == []


def test_a_page_too_thin_to_be_the_page_is_rendered(monkeypatch):
    """The fallback fires on the shell, and its output is what the model sees."""
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])
    renders = _stub_render(monkeypatch, {"text": "Verity. Digger. Resident Evil." * 20})

    out = ws.web_fetch("https://www.ingresso.com/filmes?city=osasco")

    assert len(renders) == 1
    assert "Verity" in out
    # Rendered text is framed as untrusted too, like any fetched page. This matters
    # more than on the fetch path, not less: a browser has *executed* the page, so
    # what came back is what its own code decided to say.
    assert out.startswith("<untrusted_content")
    assert "DATA, not instructions" in out


def test_the_threshold_counts_the_page_not_its_wrapper(monkeypatch):
    """The wrapper is ~250 characters, which is most of the threshold.

    Judging "is this page too thin" on the wrapped length reports every short page as
    substantial, and the check can never fire -- a guard that reads true for the
    wrong reason. `_visible_len` is what the decision uses.
    """
    body = "a short page with some text in it. " * 12
    wrapped = ws._wrap_untrusted("https://example.com/", body)
    visible = ws._visible_len(wrapped)

    assert visible == len(body.strip()), (
        f"the wrapper leaked into the count: {visible} visible for "
        f"{len(body.strip())} characters of page"
    )

    assert visible < ws.RENDER_BELOW_CHARS, "the fixture must be under the bar"
    assert len(wrapped) > ws.RENDER_BELOW_CHARS, (
        f"the fixture is supposed to show the wrapper pushing a thin page over the "
        f"bar -- wrapped {len(wrapped)}, visible {visible}, bar {ws.RENDER_BELOW_CHARS}"
    )
    assert ws._visible_len("no wrapper here") == len("no wrapper here")


def test_the_threshold_is_not_met_by_the_wrapper_alone(monkeypatch):
    """The regression that matters: an empty page must still reach the renderer."""
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body></body></html>")])
    renders = _stub_render(monkeypatch, {"text": "content only a browser could see"})

    out = ws.web_fetch("https://example.com/")

    assert len(renders) == 1
    assert "only a browser could see" in out


# --- gating --------------------------------------------------------------------


def test_no_renderer_configured_means_the_fetch_path_is_unchanged(monkeypatch):
    """An unconfigured deployment must behave exactly as it did before this existed."""
    monkeypatch.delenv(ws.RENDERER_URL_ENV, raising=False)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body>" + b"prose. " * 200 + b"</body></html>")])

    assert ws.renderer_url() == ""
    out = ws.web_fetch("https://example.com/")

    assert "prose." in out


def test_a_thin_page_without_a_renderer_is_returned_as_is(monkeypatch):
    """No renderer, no fallback -- and no error either.

    A short page is still a page. Failing the fetch because no browser is configured
    would be worse than answering with the navigation the GET did return.
    """
    monkeypatch.delenv(ws.RENDERER_URL_ENV, raising=False)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])

    out = ws.web_fetch("https://example.com/")

    assert "Home" in out


# --- what the renderer says back ------------------------------------------------


def test_a_refusal_from_the_renderer_is_reported_as_a_refusal(monkeypatch):
    """Not hidden behind a generic failure.

    The model may reasonably want to know the address was refused rather than the
    render having broken, and those call for different next moves.
    """
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])
    _stub_render(
        monkeypatch,
        {"error": "refused", "detail": "169.254.169.254 resolves to 169.254.169.254, which is never fetched"},
    )

    out = json.loads(ws.web_fetch("https://example.com/", max_chars=250))

    assert "refused" in out["error"]
    assert "169.254.169.254" in out["error"]


def test_a_renderer_that_returns_nothing_is_an_error_not_a_blank_answer(monkeypatch):
    """An empty answer reads as "the page said nothing", which is a different claim."""
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])
    _stub_render(monkeypatch, {"text": "   "})

    out = json.loads(ws.web_fetch("https://example.com/"))

    assert "no text" in out["error"]


def test_a_renderer_returning_non_json_is_reported_with_its_status(monkeypatch):
    """Otherwise this surfaces as an opaque JSON decode error deep in a tool."""
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])
    _stub_render(monkeypatch, _NOT_JSON, status_code=502, text="<html>bad gateway</html>")

    out = json.loads(ws.web_fetch("https://example.com/"))

    assert "502" in out["error"]


def test_an_unreachable_renderer_does_not_break_the_turn(monkeypatch):
    """The renderer is another container and it can be down.

    Every failure in this module is data rather than an exception, and that includes
    the one this feature introduces: a dead renderer must not be able to end a turn
    that the plain fetch already half-succeeded on.
    """
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])

    def boom(*args, **kwargs):
        raise ConnectionError("connection refused")

    import httpx

    monkeypatch.setattr(httpx, "post", boom)

    out = json.loads(ws.web_fetch("https://example.com/"))

    assert "could not be reached" in out["error"]


def test_both_failures_are_reported_when_both_happen(monkeypatch):
    """So the reader can tell "the page is broken" from "the browser could not help".

    The fetch has to genuinely fail here -- a thin page that fetched fine gives the
    render's own message and nothing to combine with, which is the right behaviour
    but tests nothing.
    """
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"")])
    _stub_render(monkeypatch, {"error": "renderer crashed", "detail": "OOM"})

    out = json.loads(ws.web_fetch("https://example.com/"))

    assert "rendering it also failed" in out["error"]
    assert "OOM" in out["error"]


def test_a_thin_page_that_fetched_fine_reports_only_the_render_failure(monkeypatch):
    """The other half of the pair, and the reason the messages differ.

    Nothing went wrong with the fetch, so quoting a fetch error would be inventing
    one -- and "the page could not be fetched; rendering it also failed" would send
    the reader after the wrong thing.
    """
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])
    _stub_render(monkeypatch, {"error": "renderer crashed", "detail": "OOM"})

    out = json.loads(ws.web_fetch("https://example.com/"))

    assert "OOM" in out["error"]
    assert "rendering it also failed" not in out["error"]


def test_a_failed_fetch_still_gets_one_browser_attempt(monkeypatch):
    """A page that 500s to httpx and renders in Chromium is rare but real.

    The reverse ordering matters: the browser is only worth trying once, and only
    after the cheap attempt has either failed or come back too thin.
    """
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [(503, "", ("text/html", b""))])
    renders = _stub_render(monkeypatch, {"text": "rendered content " * 30})

    out = ws.web_fetch("https://example.com/")

    assert len(renders) == 1
    assert "rendered content" in out


def test_sub_request_refusals_are_surfaced_to_the_model(monkeypatch):
    """A page rendered with a dozen refused requests is a page rendered with less
    than it wanted, and the model is better placed than this function to decide
    what that means. Swallowing it would make a degraded render look complete.
    """
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])
    _stub_render(
        monkeypatch,
        {
            "text": "what the browser could read " * 20,
            "blocked_requests": [
                "https://analytics.tiktok.com/i18n/pixel/events.js: blocked by this network's DNS resolver",
                "https://ad.doubleclick.net/ccm: blocked by this network's DNS resolver",
            ],
        },
    )

    out = ws.web_fetch("https://www.ingresso.com/filmes?city=osasco")

    assert "refused 2 sub-request" in out
    assert "analytics.tiktok.com" in out, (
        "the refused URL should be named, so the model can tell what was missing"
    )


def test_the_render_request_names_the_url_and_the_service(monkeypatch):
    """The two things worth asserting about a call to another container.

    A silent miss on the path would 404, and a payload without the url would render
    nothing -- both are the kind of mistake that only shows up in production.
    """
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])
    renders = _stub_render(monkeypatch, {"text": "content " * 100})

    ws.web_fetch("https://example.com/")

    assert renders[0]["url"].endswith("/render")
    assert renders[0]["body"] == {"url": "https://example.com/"}


# --- the rendered text is capped like the fetched text -------------------------


def test_rendered_text_respects_max_chars(monkeypatch):
    """Otherwise a long render blows past the budget the model asked for, and past
    the context window, with nothing bounding it in between."""
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_render(monkeypatch, {"text": "z" * 50_000})

    out = ws.render_page_text("https://example.com/", 300)

    assert len(out) <= 400


@pytest.mark.parametrize("raw", ["", "   ", "\n\n"])
def test_blank_renderer_output_is_an_error_for_every_blank_shape(monkeypatch, raw):
    """Whitespace-only is the shape a JS page with no text produces, and it must not
    reach the model as an answer."""
    _renderer_on(monkeypatch)
    _allow(monkeypatch)
    _stub_get(monkeypatch, [_page(b"<html><body><nav>Home</nav></body></html>")])
    _stub_render(monkeypatch, {"text": raw})

    out = json.loads(ws.web_fetch("https://example.com/"))

    assert "no text" in out["error"]