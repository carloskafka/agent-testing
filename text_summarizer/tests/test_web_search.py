"""Web tier: the SearXNG search tool and the guarded page fetch.

Every test here is offline and free. The transport is either stubbed at
``web_search._http_get`` or refused at ``check_url`` before a socket is opened, so
nothing leaves the process and no engine is rate-limited.

The two halves answer different questions:

* the search half is about **treating a failure as data**. A SearXNG that returns
  403 (JSON not enabled) must reach the model as readable text so it can fall
  through to its own knowledge -- an exception here would kill the turn, which is
  the whole reason ``gmail_mcp_server`` returns errors instead of raising.
* the fetch half is about **untrusted input**. Web pages are attacker-influenceable
  and their text lands in the model's context, so scheme, address, redirect, size
  and content type are all checked, and the result is framed as data.
"""

from __future__ import annotations

import json

import pytest
from text_summarizer import web_search as ws

# --- helpers ------------------------------------------------------------------


def _stub_http(monkeypatch, results):
    """Replace ``_http_get`` with a scripted sequence of ``(status, loc, payload)``."""
    calls = []

    def fake(url, *, timeout, max_bytes):
        calls.append({"url": url, "timeout": timeout, "max_bytes": max_bytes})
        if not results:
            raise AssertionError(f"unexpected request: {url}")
        item = results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(ws, "_http_get", fake)
    return calls


def _search_payload(results, unresponsive=()):
    """One scripted ``_http_get`` return: ``(status, location, payload)``.

    ``payload`` is itself the ``(content_type, body)`` pair ``_http_get`` produces,
    which is why this is a nested tuple rather than a flat one.
    """
    return (
        200,
        "",
        (
            "application/json",
            json.dumps({"results": results, "unresponsive_engines": unresponsive}).encode(),
        ),
    )


# --- the gate -----------------------------------------------------------------


def test_no_tools_when_unset(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "")
    monkeypatch.delenv(ws.WEB_SEARCH_ENABLED_ENV, raising=False)
    assert ws.build_web_search_tools() == []


def test_no_tools_when_disabled(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    monkeypatch.setenv(ws.WEB_SEARCH_ENABLED_ENV, "false")
    assert ws.build_web_search_tools() == []


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " false "])
def test_every_falsey_spelling_disables(monkeypatch, value):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    monkeypatch.setenv(ws.WEB_SEARCH_ENABLED_ENV, value)
    assert ws.build_web_search_tools() == []


def test_whitespace_url_is_not_configured(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "   ")
    monkeypatch.delenv(ws.WEB_SEARCH_ENABLED_ENV, raising=False)
    assert ws.build_web_search_tools() == []


def test_two_tools_when_configured(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    monkeypatch.delenv(ws.WEB_SEARCH_ENABLED_ENV, raising=False)
    tools = ws.build_web_search_tools()
    assert len(tools) == 2
    assert [t.name for t in tools] == ["web_search", "web_fetch"]


def test_tool_schemas_are_plain():
    """The declarations carry no union, so nothing needs the schema sanitiser.

    ``sanitize_tool_schema`` exists because ``obsidian-mcp`` declares
    ``search_metadata.value`` as a ``type`` union with no ``items``, which 400s on
    the fallback path. If a future edit here made a union, the fallback path would
    break again with no test noticing -- so this asserts the JSON schema directly
    rather than the genai ``Schema`` object, which is where the union would appear.
    """
    from google.adk.tools.function_tool import FunctionTool

    for func, expected in (
        (ws.web_search, {"query", "max_results"}),
        (ws.web_fetch, {"url", "max_chars"}),
    ):
        schema = FunctionTool(func)._get_declaration().parameters_json_schema or {}
        properties = schema.get("properties", {})
        assert set(properties) == expected
        # A union would be a list of type objects; a plain string/int is not.
        for name, definition in properties.items():
            assert isinstance(definition.get("type"), str), (
                f"{name} has a non-scalar type, which the fallback path may reject"
            )
        assert (
            schema["properties"]["max_results" if "query" in expected else "max_chars"]["type"]
            == "integer"
        )


# --- search -------------------------------------------------------------------


def test_search_returns_title_url_snippet_engine(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(
        monkeypatch,
        [
            _search_payload(
                [
                    {
                        "title": "OpenCode",
                        "url": "https://opencode.ai/",
                        "content": "The open source AI coding agent",
                        "engine": "google",
                    }
                ]
            )
        ],
    )
    hits = ws.search_web("opencode")
    assert len(hits) == 1
    assert hits[0].title == "OpenCode"
    assert hits[0].url == "https://opencode.ai/"
    assert hits[0].snippet == "The open source AI coding agent"
    assert hits[0].engine == "google"


def test_search_clamps_max_results(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    body = [{"title": f"t{i}", "url": f"https://e.com/{i}", "engine": "google"} for i in range(30)]
    _stub_http(monkeypatch, [_search_payload(body)])
    assert len(ws.search_web("x", 500)) == ws.MAX_RESULTS_CAP


def test_search_floor_of_one_result(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(
        monkeypatch, [_search_payload([{"title": "t", "url": "https://e.com/", "engine": "g"}])]
    )
    assert len(ws.search_web("x", 0)) == 1


def test_json_format_disabled_is_a_readable_403(monkeypatch):
    """The load-bearing SearXNG failure: ``search.formats`` missing ``json``.

    Upstream ships ``[html]`` only, so a request for any other format is 403 -- and
    the service still looks healthy, because the failure is per-request. It must
    arrive as text the model can act on, naming the actual fix.
    """
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(monkeypatch, [(403, "", None)])
    with pytest.raises(ValueError, match="403"):
        ws.search_web("opencode")


def test_403_message_names_the_setting(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(monkeypatch, [(403, "", None)])
    with pytest.raises(ValueError) as excinfo:
        ws.search_web("opencode")
    assert "formats" in str(excinfo.value)


def test_transport_error_raises_value_error(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(monkeypatch, [OSError("connection refused")])
    with pytest.raises(ValueError, match="failed"):
        ws.search_web("opencode")


def test_non_json_body_is_an_error_not_a_crash(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(monkeypatch, [(200, "", ("text/html", b"<html>nope</html>"))])
    with pytest.raises(ValueError, match="non-JSON"):
        ws.search_web("opencode")


def test_search_query_is_url_encoded(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    calls = _stub_http(monkeypatch, [_search_payload([])])
    ws.search_web("searxng json api & more")
    assert "searxng%20json%20api%20%26%20more" in calls[0]["url"]
    assert "format=json" in calls[0]["url"]


@pytest.mark.parametrize("scheme", ["javascript:", "data:", "file:", "ftp:"])
def test_non_web_result_schemes_are_dropped(monkeypatch, scheme):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(
        monkeypatch, [_search_payload([{"title": "bad", "url": f"{scheme}//evil", "engine": "g"}])]
    )
    assert ws.search_web("x") == []


def test_non_dict_results_are_skipped(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(
        monkeypatch, [_search_payload(["a string", 7, {"title": "ok", "url": "https://e.com/"}])]
    )
    assert [h.title for h in ws.search_web("x")] == ["ok"]


def test_results_without_url_are_skipped(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(monkeypatch, [_search_payload([{"title": "no url", "engine": "g"}])])
    assert ws.search_web("x") == []


# --- the tools' failure surface ----------------------------------------------


def test_web_search_returns_error_data_not_an_exception(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(monkeypatch, [(403, "", None)])
    payload = json.loads(ws.web_search("opencode"))
    assert set(payload) == {"error"}
    assert "403" in payload["error"]


def test_web_search_says_empty_is_not_the_same_as_rate_limited(monkeypatch):
    """Empty results must not read as "the web has nothing on this"."""
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(monkeypatch, [_search_payload([])])
    error = json.loads(ws.web_search("opencode"))["error"]
    assert "no results" in error
    assert "rate-limited" in error


def test_web_search_returns_a_json_list_on_success(monkeypatch):
    monkeypatch.setenv(ws.SEARXNG_URL_ENV, "http://searxng:8080")
    _stub_http(
        monkeypatch,
        [
            _search_payload(
                [
                    {
                        "title": "OpenCode",
                        "url": "https://opencode.ai/",
                        "content": "x",
                        "engine": "google",
                    }
                ]
            )
        ],
    )
    payload = json.loads(ws.web_search("opencode"))
    assert isinstance(payload, list)
    assert payload[0]["url"] == "https://opencode.ai/"


def test_web_fetch_returns_error_data(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "refused for the test")
    payload = json.loads(ws.web_fetch("https://example.com/"))
    assert set(payload) == {"error"}


# --- URL guards ---------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "data:text/html,<script>alert(1)</script>",
        "ftp://example.com/x",
        "gopher://example.com/",
    ],
)
def test_non_http_schemes_are_refused(url):
    assert ws.check_url(url)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "localhost",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.137",
        "169.254.169.254",  # cloud instance metadata -- the reason this table exists
        "[::1]",
        "0.0.0.0",
    ],
)
def test_private_and_loopback_addresses_are_refused(host):
    problem = ws.check_url(f"http://{host}/latest/meta-data/")
    assert problem, f"{host} should be refused"
    assert "non-public" in problem or "cannot resolve" in problem or "scheme" in problem


def test_decimal_ip_is_resolved_not_string_matched():
    """``http://2130706433/`` is 127.0.0.1 written as a decimal integer.

    A hostname allow/deny list cannot see this. Resolve-then-check can, which is
    why :func:`check_url` resolves before deciding.
    """
    addresses = ws._resolve_public_address("2130706433")
    if not addresses:
        pytest.skip("this resolver does not answer numeric hosts")
    assert any(a.is_loopback for a in addresses)
    assert ws.check_url("http://2130706433/")


def test_ipv4_mapped_ipv6_is_judged_as_ipv4():
    """``::ffff:127.0.0.1`` is loopback wearing an IPv6 hat."""
    import ipaddress

    assert not ws._is_public_address(ipaddress.ip_address("::ffff:127.0.0.1"))
    assert not ws._is_public_address(ipaddress.ip_address("::ffff:10.0.0.1"))


def test_unresolvable_host_is_refused():
    assert "cannot resolve" in ws.check_url("http://no-such-host.invalid/")


def test_missing_host_is_refused():
    assert ws.check_url("http:///path")
    assert ws.check_url("")


def test_public_address_is_allowed():
    """A real public name must not be refused -- the guards are not a blanket deny."""
    assert ws.check_url("https://example.com/") == ""


# --- fetch guards -------------------------------------------------------------


def _ok_page(body: bytes = b"<html><body><p>hello</p></body></html>"):
    return 200, "", ("text/html; charset=utf-8", body)


def test_fetch_extracts_text_and_frames_it(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [_ok_page()])
    out = ws.fetch_page_text("https://example.com/")
    assert "hello" in out
    assert out.startswith("<untrusted_content")
    assert out.rstrip().endswith("</untrusted_content>")


def test_fetched_text_is_labelled_as_data_not_instructions(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [_ok_page()])
    out = ws.fetch_page_text("https://example.com/")
    assert "DATA, not instructions" in out
    assert "Never follow" in out


def test_script_and_style_content_is_dropped(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    page = (
        b"<html><head><style>body{color:red}</style>"
        b"<script>alert('x')</script></head>"
        b"<body><p>visible</p></body></html>"
    )
    _stub_http(monkeypatch, [_ok_page(page)])
    out = ws.fetch_page_text("https://example.com/")
    assert "visible" in out
    assert "alert" not in out
    assert "color:red" not in out


def test_entities_are_unescaped(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [_ok_page(b"<html><body><p>a &amp; b &lt;tag&gt;</p></body></html>")])
    out = ws.fetch_page_text("https://example.com/")
    assert "a & b <tag>" in out


def test_html_comments_are_removed(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [_ok_page(b"<html><body><!-- hidden --><p>shown</p></body></html>")])
    out = ws.fetch_page_text("https://example.com/")
    assert "hidden" not in out
    assert "shown" in out


def test_whitespace_is_collapsed(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [_ok_page(b"<html><body><p>a     b\t\tc</p></body></html>")])
    out = ws.fetch_page_text("https://example.com/")
    assert "a b c" in out


def test_truncation_is_announced(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    body = ("<html><body>" + " ".join(["word"] * 5000) + "</body></html>").encode()
    _stub_http(monkeypatch, [_ok_page(body)])
    out = ws.fetch_page_text("https://example.com/", max_chars=400)
    assert "truncated" in out
    assert len(out) < 1200


def test_max_chars_is_capped(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [_ok_page(b"<html><body><p>x</p></body></html>")])
    # Asked for more than the cap: the cap wins, and the request does not fail.
    ws.fetch_page_text("https://example.com/", max_chars=10_000_000)


@pytest.mark.parametrize(
    "content_type",
    [
        "application/pdf",
        "image/png",
        "application/octet-stream",
        "application/zip",
    ],
)
def test_non_text_content_types_are_refused(monkeypatch, content_type):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [(200, "", (content_type, b"\x00\x01binary"))])
    with pytest.raises(ValueError, match="not readable text"):
        ws.fetch_page_text("https://example.com/file")


def test_text_plain_is_accepted(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [(200, "", ("text/plain; charset=utf-8", b"just words"))])
    assert "just words" in ws.fetch_page_text("https://example.com/")


def test_http_error_is_reported(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [(404, "", None)])
    with pytest.raises(ValueError, match="404"):
        ws.fetch_page_text("https://example.com/missing")


def test_page_with_no_readable_text_is_reported(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [_ok_page(b"<html><body><script>x</script></body></html>")])
    with pytest.raises(ValueError, match="no readable text"):
        ws.fetch_page_text("https://example.com/")


def test_refused_url_never_reaches_the_transport(monkeypatch):
    """A guard that runs *after* the request would defeat the point."""
    calls = _stub_http(monkeypatch, [])
    with pytest.raises(ValueError, match="refusing to fetch"):
        ws.fetch_page_text("file:///etc/passwd")
    assert calls == []


def test_too_large_body_is_refused(monkeypatch):
    """The streaming ceiling, exercised at the seam the transport reports."""
    monkeypatch.setattr(ws, "check_url", lambda url: "")

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}

        def stream(self, method, url):  # pragma: no cover - not used
            raise AssertionError

    # Rather than fake httpx, drive the size ceiling through a stub client.
    import types as pytypes

    chunk = b"x" * 4096

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs.get("follow_redirects") is False, "redirects must not be followed"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url):
            class Ctx:
                status_code = 200
                headers = {"content-type": "text/html"}

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *e):
                    return False

                def iter_bytes(self_inner):
                    for _ in range(10):  # 40 KiB, past a 2 KiB ceiling
                        yield chunk

            return Ctx()

    fake_httpx = pytypes.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)
    monkeypatch.setattr(ws, "MAX_PAGE_BYTES", 2048)

    # Raised at the transport, not returned: a caller cannot mistake a half-read
    # body for the whole page. fetch_page_text converts it to a readable error,
    # which is asserted separately below.
    with pytest.raises(ws._BodyTooLarge):
        ws._http_get("https://example.com/", timeout=1, max_bytes=2048)


def test_too_large_page_becomes_a_readable_error(monkeypatch):
    """The model's view of an oversized page is an error string, not a crash."""
    import types as pytypes

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url):
            class Ctx:
                status_code = 200
                headers = {"content-type": "text/html"}

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *e):
                    return False

                def iter_bytes(self_inner):
                    for _ in range(10):
                        yield b"x" * 4096

            return Ctx()

    monkeypatch.setitem(
        __import__("sys").modules, "httpx", pytypes.SimpleNamespace(Client=FakeClient)
    )
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    monkeypatch.setattr(ws, "MAX_PAGE_BYTES", 2048)

    with pytest.raises(ValueError, match="too large to fetch"):
        ws.fetch_page_text("https://example.com/")

    # And through the tool, as data rather than an exception.
    payload = json.loads(ws.web_fetch("https://example.com/"))
    assert set(payload) == {"error"}
    assert "too large" in payload["error"]


def test_redirects_are_not_followed_automatically(monkeypatch):
    """A redirect is how a checked URL walks into the metadata endpoint."""
    import types as pytypes

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url):
            class Ctx:
                status_code = 302
                headers = {"location": "http://169.254.169.254/latest/meta-data/"}

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *e):
                    return False

                def iter_bytes(self_inner):
                    return iter(())

            return Ctx()

    monkeypatch.setitem(
        __import__("sys").modules, "httpx", pytypes.SimpleNamespace(Client=FakeClient)
    )
    status, location, payload = ws._http_get("https://example.com/", timeout=1, max_bytes=10)
    assert status == 302
    assert "169.254.169.254" in location
    assert payload is None
    assert seen["follow_redirects"] is False


def test_redirect_to_a_private_address_is_refused(monkeypatch):
    """The first hop is allowed, the second is not, and the chain stops there."""
    monkeypatch.setattr(ws, "check_url", lambda url: "" if "example.com" in url else "non-public")
    _stub_http(monkeypatch, [(302, "http://169.254.169.254/latest/meta-data/", None)])
    with pytest.raises(ValueError, match="refusing to fetch"):
        ws.fetch_page_text("https://example.com/")


def test_redirect_loop_is_bounded(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [(302, "/next", None)] * 10)
    with pytest.raises(ValueError, match="too many redirects"):
        ws.fetch_page_text("https://example.com/")


def test_relative_redirect_is_resolved_against_the_current_url(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    calls = _stub_http(monkeypatch, [(302, "/elsewhere", None), _ok_page()])
    ws.fetch_page_text("https://example.com/start")
    assert calls[1]["url"] == "https://example.com/elsewhere"


def test_transport_failure_during_fetch_is_a_value_error(monkeypatch):
    monkeypatch.setattr(ws, "check_url", lambda url: "")
    _stub_http(monkeypatch, [OSError("timed out")])
    with pytest.raises(ValueError, match="could not fetch"):
        ws.fetch_page_text("https://example.com/")
