"""A URL the model was shown must be a URL the model may cite.

The rule the `**Sources**` renderer enforces is that a `[web]` line naming a URL
the search never returned is dropped -- the same rule a vault note title matching
no file is dropped, and for the same reason: a citation asserts the thing exists
and is relevant, so one that cannot be substantiated is a fabrication.

That rule only works if the allow-list is the set of URLs the model was actually
*shown*. It was not: `record_returned_urls` registered the page that was fetched
and nothing else, while `[links on this page]` handed the model a page full of
other URLs. So the model read a link, cited it, and had the citation silently
removed.

This is not a marginal case. On the Ingresso listing the fetched page is the
film index and the useful URLs are the per-session checkout links inside it, so
the one citation a reader needed was the one the allow-list forbade.
"""

from __future__ import annotations

import types

from text_summarizer import web_search as ws

LISTING_HTML = """
<html><body>
  <a href="/filmes/verity">Verity</a>
  <a href="https://checkout.ingresso.com/?sessionId=87205315&amp;partnership=home">Comprar</a>
  <a href="/cinemas">Cinemas</a>
  <a href="https://checkout.ingresso.com/?sessionId=87205316&amp;partnership=home">Comprar</a>
</body></html>
"""

PAGE_URL = "https://www.ingresso.com/filmes?city=osasco"


def _ctx() -> types.SimpleNamespace:
    return types.SimpleNamespace(state={}, invocation_id="inv-1")


def _allowlisted(ctx) -> set[str]:
    recorded = ctx.state.get(ws.WEB_URLS_STATE_KEY) or {}
    return set(recorded.get("inv-1") or ())


def test_a_link_shown_on_the_page_can_be_cited():
    """The whole point: the checkout link is in the allow-list after a fetch."""
    ctx = _ctx()
    shown: list[str] = []
    block = ws._format_links(LISTING_HTML, PAGE_URL, shown=shown)
    ws.record_returned_urls(ctx, shown)

    assert "checkout.ingresso.com/?sessionId=87205315" in block
    assert any("87205315" in u for u in _allowlisted(ctx)), (
        "the model was shown this URL, so it must be allowed to cite it"
    )


def test_the_page_itself_is_still_registered():
    """Registering the links must not replace registering the page."""
    ctx = _ctx()
    shown: list[str] = []
    ws._format_links(LISTING_HTML, PAGE_URL, shown=shown)
    ws.record_returned_urls(ctx, [PAGE_URL, *shown])
    assert PAGE_URL in _allowlisted(ctx)


def test_a_url_dropped_by_the_cap_is_not_citable():
    """The allow-list is what was *shown*, not what existed.

    Asserting the converse matters as much as the direct case: a version that
    registered every href it found before capping would let the model cite a link
    it never saw, which is the fabrication this whole mechanism exists to prevent.
    """
    ctx = _ctx()
    shown: list[str] = []
    ws._format_links(LISTING_HTML, PAGE_URL, limit=1, shown=shown)
    ws.record_returned_urls(ctx, shown)

    assert len(shown) == 1
    assert len(_allowlisted(ctx)) == 1, "nothing that was cut may be citable"


def test_the_collection_is_empty_when_a_page_has_no_links():
    """A prose page must not create an empty registration."""
    ctx = _ctx()
    shown: list[str] = []
    assert ws._format_links("<p>just words</p>", PAGE_URL, shown=shown) == ""
    assert shown == []
    ws.record_returned_urls(ctx, shown)
    assert _allowlisted(ctx) == set()


def test_web_fetch_actually_registers_the_links_it_shows(monkeypatch):
    """The wiring, not the mechanism.

    Every test above calls `record_returned_urls` by hand, so they all pass against
    a `web_fetch` that registers only the page URL -- which is exactly the defect
    this file exists to catch. Verified by mutation: reverting the call site to
    `record_returned_urls(tool_context, [url])` left all four green.

    So this one drives the real `web_fetch` with the fetch stubbed, and asserts on
    the state the agent would later be checked against.
    """
    monkeypatch.setenv("RENDERER_URL", "http://renderer.invalid:8080")

    def fake_fetch(url, max_chars, shown=None):
        # Long enough that the cheap path is taken. A stub returning a short page
        # sends `web_fetch` off to the renderer, and the test would then be
        # asserting on a dial failure rather than on the allow-list.
        text = "Verity - 19:30. R$ 30,00. " + ("Sessao disponivel. " * 60)
        links = ws._format_links(LISTING_HTML, url, shown=shown)
        return f"{text}\n\n{links}"

    monkeypatch.setattr(ws, "fetch_page_text", fake_fetch)

    ctx = _ctx()
    out = ws.web_fetch(PAGE_URL, tool_context=ctx)

    # The model can see the link...
    assert "87205315" in out
    # ...and is allowed to cite it.
    assert any("87205315" in u for u in _allowlisted(ctx)), (
        "web_fetch showed the model a checkout link and left it uncitable; the "
        "renderer will silently drop the citation line for it"
    )