"""A booking link the model can read, tie to a time, and cite.

Found live on session `fae42db3-3955-4abe-8fd1-b2003e84749d`. The user narrowed to
one screening — Resident Evil, 21:15, Kinoplex Osasco — and the agent handed back
`<https://www.ingresso.com/filme/resident-evil?city=osasco>`, the **film page**, when
the answer they needed was the checkout URL for that specific session.

The page had it. The model could not use it. `_format_links` reads an anchor's
*inner text* as its label, and these anchors have none — the session lives in a
sibling JSON-LD block — so what reached the model was:

    - https://checkout.ingresso.com/?sessionId=87205315&partnership=home
    - https://checkout.ingresso.com/?sessionId=87206124&partnership=home
    - https://checkout.ingresso.com/?sessionId=87210538&partnership=home

Five opaque IDs, a page full of session times, and **nothing joining them**. The
model cited the page it fetched because that was the only URL it could attribute.
Five bare URLs look like a working links block, which is why this survived a test
suite that checked links were *citable* and never checked they were *usable*.

The fix reads schema.org `ScreeningEvent` — the standard vocabulary for exactly
this, used by cinemas, airlines and event listings alike — where each record is a
complete booking unit in three separate fields: `startDate`, `location.name` and
`offers.url`.
"""

from __future__ import annotations

import json

from text_summarizer import web_search as ws

# One record as Ingresso.com emits it, trimmed to the fields that matter.
SINGLE = """<script type="application/ld+json">%s</script>"""


def _ld(*events: dict) -> str:
    return SINGLE % json.dumps({"@context": "https://schema.org", "@graph": list(events)})


def _event(when: str, cinema: str, session: str) -> dict:
    return {
        "@type": "ScreeningEvent",
        "startDate": f"2026-10-03T{when}:00-03:00",
        "workPerformed": {"@type": "Movie", "name": "Resident Evil"},
        "location": {
            "@type": "MovieTheater",
            "name": cinema,
            "address": {"@type": "PostalAddress", "streetAddress": "AV DOS AUTONOMISTAS, 1400"},
        },
        "offers": {
            "@type": "Offer",
            "url": f"https://checkout.ingresso.com/?sessionId={session}&partnership=home",
            "priceCurrency": "BRL",
        },
    }


REAL = _ld(
    _event("15:20", "Cinemark Osasco", "87205315"),
    _event("20:20", "Cinemark Osasco", "87206124"),
    _event("22:40", "Cinemark Osasco", "87205517"),
    _event("19:10", "Kinoplex Osasco", "87210538"),
    _event("21:15", "Kinoplex Osasco", "87210684"),
)


# --- the shape the model needs -----------------------------------------------


def test_a_session_carries_its_time_its_cinema_and_its_booking_url():
    """The whole fix in one assertion: the three fields are joined.

    The failure this replaces is *not* a missing URL. It is a URL the model cannot
    attribute to a time, so it cannot put the right one on the right fact.
    """
    block, urls = ws._format_sessions(REAL)
    assert "- 21:15 Kinoplex Osasco  https://checkout.ingresso.com/?sessionId=87210684" in block
    assert len(urls) == 5


def test_sessions_are_sorted_by_time():
    """Unordered sessions make the model match a time to a link by eye."""
    block, _ = ws._format_sessions(REAL)
    times = [line.split()[1] for line in block.splitlines() if line.startswith("- ")]
    assert times == sorted(times)


def test_the_date_is_dropped_from_every_line():
    """Printing `2026-10-03T21:15` invites reading a timestamp as a time of day.

    That is how a listing of today's sessions turns into an answer about the 3rd --
    the page is about one day, so repeating the date on every line adds nothing and
    costs the model a chance to misread it.
    """
    block, _ = ws._format_sessions(REAL)
    assert "2026-10-03" not in block
    assert "T21:15" not in block


def test_a_page_with_no_sessions_yields_nothing():
    """The control. A block that appeared everywhere would be noise on every page."""
    assert ws._format_sessions("<html><body><p>hello</p></body></html>") == ("", [])
    assert ws._format_sessions("") == ("", [])


def test_ordinary_json_ld_is_left_alone():
    """The control for the parser's appetite: a Movie node is not a screening."""
    markup = (
        '<script type="application/ld+json">'
        + json.dumps({"@type": "Movie", "name": "Resident Evil"})
        + "</script>"
    )
    assert ws._format_sessions(markup) == ("", [])


# --- the shapes JSON-LD actually arrives in -----------------------------------


def test_all_three_json_ld_shapes_are_read():
    """A bare object, a top-level array, and an `@graph` wrapper all occur in the wild.

    Matching the three by pattern would work until the next site nests one level
    differently; walking the structure does not care which it is.
    """
    ev = _event("21:15", "Kinoplex Osasco", "87210684")
    for doc in (ev, [ev], {"@graph": [ev]}, {"@graph": [{"@graph": [ev]}]}):
        markup = f'<script type="application/ld+json">{json.dumps(doc)}</script>'
        assert len(ws._format_sessions(markup)[1]) == 1, json.dumps(doc)[:60]


def test_malformed_json_ld_is_skipped_not_raised():
    """A broken script tag is a normal thing to find. The page still renders."""
    broken = '<script type="application/ld+json">{not json,,,</script>'
    assert ws._format_sessions(broken) == ("", [])


def test_deeply_nested_json_does_not_recurse_without_bound():
    """Attacker-controlled input, so the depth bound is the point.

    Unbounded recursion on a hostile page is a crash the model reads as "the page
    was broken" -- and it is a crash in the *fetch*, so it costs the turn.
    """
    node: dict = {"@type": "ScreeningEvent"}
    for _ in range(200):
        node = {"@graph": [node]}
    assert len(list(ws._iter_ld_nodes(node))) < 20


# --- incomplete records are worse than none -----------------------------------


def test_a_screening_with_no_offer_url_is_dropped():
    """Half a record is the defect, not the fix."""
    ev = _event("21:15", "Kinoplex Osasco", "87210684")
    del ev["offers"]
    assert ws._format_sessions(_ld(ev)) == ("", [])


def test_a_screening_with_no_location_is_dropped():
    """A URL with no cinema beside it is the opaque sessionId this exists to replace."""
    ev = _event("21:15", "Kinoplex Osasco", "87210684")
    del ev["location"]
    assert ws._format_sessions(_ld(ev)) == ("", [])


def test_a_screening_with_no_start_date_is_dropped():
    ev = _event("21:15", "Kinoplex Osasco", "87210684")
    del ev["startDate"]
    assert ws._format_sessions(_ld(ev)) == ("", [])


def test_an_off_scheme_offer_url_is_dropped():
    """`offers.url` is page-controlled, so it gets the same vetting as an href."""
    ev = _event("21:15", "Kinoplex Osasco", "87210684")
    ev["offers"]["url"] = "javascript:alert(1)"
    assert ws._format_sessions(_ld(ev)) == ("", [])


# --- the invariant that ties it to citations ---------------------------------


def test_every_booking_url_shown_is_also_citable():
    """Shown is not enough; the allow-list is what makes it *sayable*.

    This is the coupling that made the first version useless: the URLs were printed
    and not registered, so the citation renderer dropped every `[web]` line naming
    one. An answer that silently loses the link the user asked for is worse than an
    answer with no link, because it reads as complete.
    """
    shown: list[str] = []
    block = ws._links_and_sessions(REAL, "https://www.ingresso.com/filme/x", shown)
    assert "87210684" in block
    assert any("87210684" in u for u in shown), "shown but uncitable"


def test_with_nowhere_to_register_the_sessions_are_not_printed():
    """Better no block than an unusable one.

    ``shown=None`` means the caller cannot record the URLs, so printing them would
    put an uncitable booking link in front of the model. The block is suppressed
    rather than shown-and-dropped.
    """
    assert "sessions on this page" not in ws._links_and_sessions(
        REAL, "https://www.ingresso.com/filme/x", None
    )


# --- the wiring, which is where the first version failed ----------------------


def test_the_renderer_path_emits_sessions_not_just_the_plain_http_path(monkeypatch):
    """Ingresso renders its sessions in JavaScript, so only this path matters.

    The first version wired the block into `fetch_page_text` — the plain-HTTP path
    — and left `render_page_text` untouched. It looked correct in both places by
    inspection, and silently did nothing on the only site that publishes
    `ScreeningEvent`, which is exactly the site the renderer exists for.

    This asserts both paths go through the one helper, because "wired in one place
    and forgotten in the other" is not a mistake you can catch by reading.
    """
    import inspect

    for fn in (ws.fetch_page_text, ws.render_page_text, ws.dismiss_page_text):
        src = inspect.getsource(fn)
        assert "_links_and_sessions(" in src, (
            f"{fn.__name__} does not use the shared helper; a session block wired "
            "into one path and not another is the bug this test is for"
        )
        assert "_format_sessions(" not in src, (
            f"{fn.__name__} calls _format_sessions directly instead of the helper"
        )


def test_an_end_to_end_fetch_offers_the_right_session_url(monkeypatch):
    """The whole thing through `web_fetch`, which is what the model calls."""
    monkeypatch.setattr(
        ws,
        "fetch_page_text",
        lambda url, max_chars, shown=None: (
            "Sessões\nCinemark Osasco 15:20\nKinoplex Osasco 21:15\n\n"
            + ws._links_and_sessions(REAL, url, shown)
        ),
    )

    class Ctx:
        invocation_id = "e2e"

        def __init__(self):
            self.state = {}

    ctx = Ctx()
    out = ws.web_fetch("https://www.ingresso.com/filme/resident-evil", tool_context=ctx)
    assert "- 21:15 Kinoplex Osasco  https://checkout.ingresso.com/?sessionId=87210684" in out

    allowed = set((ctx.state.get(ws.WEB_URLS_STATE_KEY) or {}).get("e2e") or ())
    assert any("87210684" in u for u in allowed), (
        "the answer will name this URL in a [web] line; if it is not allow-listed "
        "the renderer drops the citation and the user gets no link"
    )