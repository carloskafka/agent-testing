"""Read-only Gmail: the MCP server's logic and the toolset that spawns it.

These were at 0% coverage, which is worth explaining because the number was
actively misleading. ``agent.py`` builds its tool list at *import* time::

    tools = [*build_obsidian_tools(), *build_gmail_tools()]

and the old ``conftest.py`` blanked only ``LANGFUSE_PUBLIC_KEY``. So
``build_gmail_tools()`` took its happy path during collection on any machine whose
``.env`` had Gmail credentials, and its ``return []`` path on CI. Same suite,
different code exercised, coverage that moved with the developer's secrets.
``conftest.py`` now pins every variable, and ``test_gmail.py`` is what actually
exercises the code.

No network and no Google credentials: the Gmail client is never constructed. The
fake below stands in for ``googleapiclient``'s fluent ``.users().messages()...``
chain, which is just method calls ending in ``.execute()``.

**The most valuable thing in here** is ``_decoded_body``. It decides *what text the
model actually sees*, it walks nested MIME structures recursively, and nothing
tested any of that -- a regression there would not throw, it would quietly feed
the model empty emails.
"""

from __future__ import annotations

import base64
import sys
from types import SimpleNamespace

import pytest

from text_summarizer import gmail_mcp_server as server
from text_summarizer import gmail_tools


# --- fakes --------------------------------------------------------------------


def _b64(text: str) -> str:
    """Base64url the way Gmail sends it: standard alphabet, padding intact."""
    return base64.urlsafe_b64encode(text.encode()).decode()


class _FakeGmail:
    """Stands in for the Gmail API client.

    ``execute()`` returns queued values in call order, which is what the real
    client does: the list tools call it once for the id list and then once per
    message to hydrate it. Recorded call paths accumulate across the whole fluent
    chain ("users.messages.list"), so a test can assert on the exact call.
    """

    def __init__(self, *results):
        self._queue = list(results)
        self.calls: list[tuple[str, dict]] = []

    def users(self):
        return self._node("users.")

    def _node(self, prefix: str):
        fake = self

        class _Node:
            def __getattr__(self, method):
                if method.startswith("_"):
                    raise AttributeError(method)

                def inner(**kwargs):
                    fake.calls.append((prefix + method, kwargs))
                    return fake._node(prefix + method + ".")

                return inner

            def execute(self):
                if not fake._queue:
                    raise AssertionError("Gmail fake ran out of queued responses")
                nxt = fake._queue.pop(0)
                # A queued exception is *raised*, which is the whole point: the
                # real client raises HttpError rather than returning it, and a
                # fake that returned it would never exercise the except clauses.
                if isinstance(nxt, BaseException):
                    raise nxt
                return nxt

        return _Node()


def _http_error(status: int = 404, content: bytes = b"Not Found") -> Exception:
    """A real ``HttpError`` -- the code catches that type specifically."""
    from googleapiclient.errors import HttpError

    # HttpError reads resp.status and resp.reason as attributes.
    resp = SimpleNamespace(status=status, reason="Not Found", headers={})
    return HttpError(resp, content)


@pytest.fixture
def fake_service(monkeypatch):
    """Install a fake ``_service()`` and return a factory for the results."""

    def install(*results):
        fake = _FakeGmail(*results)
        monkeypatch.setattr(server, "_service", lambda: fake)
        return fake

    return install


@pytest.fixture(autouse=True)
def _reset_service_cache():
    """The module caches the client in a global; never leak it between tests."""
    original = server._service_cache
    server._service_cache = None
    yield
    server._service_cache = original


def _message(msg_id="m1", thread_id="t1", **headers) -> dict:
    payload = {"headers": [{"name": k, "value": v} for k, v in headers.items()]}
    return {"id": msg_id, "threadId": thread_id, "snippet": "a snippet", "payload": payload}


def _text_part(text: str, mime: str = "text/plain") -> dict:
    return {"mimeType": mime, "body": {"data": _b64(text)}}


# --- _env ----------------------------------------------------------------------


def test_env_returns_the_stripped_value(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "  spaced  ")
    assert server._env("GOOGLE_CLIENT_ID") == "spaced"


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_env_rejects_blank_values_by_name(monkeypatch, value):
    """The message must name the variable, or the user cannot act on it."""
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", value)
    with pytest.raises(ValueError, match="GOOGLE_REFRESH_TOKEN is not set"):
        server._env("GOOGLE_REFRESH_TOKEN")


def test_env_rejects_a_completely_absent_var(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_CLIENT_SECRET is not set"):
        server._env("GOOGLE_CLIENT_SECRET")


# --- _header ------------------------------------------------------------------


def test_header_match_is_case_insensitive():
    """Gmail returns "From", but a forwarded message can differ."""
    msg = {"payload": {"headers": [{"name": "SUBJECT", "value": "Hello"}]}}
    assert server._header(msg, "Subject") == "Hello"


def test_a_missing_header_is_empty_not_none():
    assert server._header({"payload": {"headers": []}}, "Cc") == ""


def test_a_message_with_no_payload_at_all_is_handled():
    assert server._header({}, "From") == ""


def test_the_first_matching_header_wins():
    msg = {
        "payload": {
            "headers": [
                {"name": "Received", "value": "first"},
                {"name": "Received", "value": "second"},
            ]
        }
    }
    assert server._header(msg, "Received") == "first"


# --- _decoded_body: the part that decides what the model sees -------------------


def test_plain_base64_body_is_decoded():
    payload = {"body": {"data": _b64("hello there")}}
    assert server._decoded_body(payload) == "hello there"


def test_undecodable_data_yields_empty_rather_than_raising():
    """A corrupt payload must not take the tool down."""
    assert server._decoded_body({"body": {"data": "!!!not base64!!!"}}) == ""


def test_unpadded_base64_is_decoded_rather_than_silently_dropped():
    """Padding is restored before decoding.

    Any body whose byte length is not a multiple of 3 needs "=" characters, and
    b64decode raises without them. The bare `except` around it would then return
    "", so the model would be handed a bodyless email with nothing logged and
    nothing raised -- the worst possible failure for this function, whose whole
    job is deciding what text the model sees.
    """
    unpadded = _b64("a body whose length is not a multiple of three").rstrip("=")
    assert unpadded != _b64("a body whose length is not a multiple of three")
    assert server._decoded_body({"body": {"data": unpadded}}) == (
        "a body whose length is not a multiple of three"
    )


def test_invalid_utf8_is_replaced_not_fatal():
    payload = {"body": {"data": base64.urlsafe_b64encode(b"\xff\xfe bad").decode()}}
    out = server._decoded_body(payload)
    assert isinstance(out, str) and out  # no exception, some text back


def test_a_payload_with_neither_data_nor_parts_is_empty():
    assert server._decoded_body({}) == ""
    assert server._decoded_body({"body": {}}) == ""
    assert server._decoded_body({"body": None, "parts": []}) == ""


def test_multipart_keeps_only_text_parts():
    """An attachment must not be base64-decoded into the prompt as noise."""
    payload = {
        "parts": [
            _text_part("the body"),
            {"mimeType": "image/png", "body": {"data": _b64("\x89PNG binary")}},
        ]
    }
    assert server._decoded_body(payload) == "the body"


def test_multipart_joins_every_text_part():
    payload = {
        "parts": [
            _text_part("plain version", "text/plain"),
            _text_part("<p>html version</p>", "text/html"),
        ]
    }
    assert server._decoded_body(payload) == "plain version\n<p>html version</p>"


def test_nested_multipart_is_walked_recursively():
    """multipart/mixed -> multipart/alternative -> text/plain is the common shape."""
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {"mimeType": "application/pdf", "body": {"data": _b64("%PDF-1.4")}},
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    _text_part("the real text", "text/plain"),
                    _text_part("<b>rich</b>", "text/html"),
                ],
            },
        ]
    }
    assert server._decoded_body(payload) == "the real text\n<b>rich</b>"


def test_empty_text_parts_are_dropped_from_the_join():
    """Otherwise a blank alternative leaves a leading blank line in the prompt."""
    payload = {"parts": [_text_part(""), _text_part("only this")]}
    assert server._decoded_body(payload) == "only this"


def test_data_on_a_multipart_wins_over_its_parts():
    """Documented Gmail behaviour: a top-level body.data short-circuits."""
    payload = {"body": {"data": _b64("top level")}, "parts": [_text_part("nested")]}
    assert server._decoded_body(payload) == "top level"


# --- _summary / _full ----------------------------------------------------------


def test_summary_projects_the_fields_the_agent_uses():
    summary = server._summary(
        _message(From="a@b.c", To="d@e.f", Subject="Hi", Date="Mon, 1 Jan 2026")
    )
    assert summary == {
        "id": "m1",
        "thread_id": "t1",
        "from": "a@b.c",
        "to": "d@e.f",
        "subject": "Hi",
        "date": "Mon, 1 Jan 2026",
        "snippet": "a snippet",
    }


def test_summary_tolerates_a_message_with_nothing_in_it():
    """Metadata-only fetches can come back nearly empty; must not raise."""
    summary = server._summary({"id": "m2"})
    assert summary["id"] == "m2"
    assert summary["from"] == ""
    assert summary["snippet"] == ""


def test_full_falls_back_to_the_snippet_when_there_is_no_body():
    """Common for HTML-only mail: the body is empty but the snippet is not."""
    full = server._full({"id": "m1", "snippet": "just a snippet", "payload": {"headers": []}})
    assert full["body"] == "just a snippet"


def test_full_prefers_a_real_body_over_the_snippet():
    full = server._full(
        {"id": "m1", "snippet": "snip", "payload": {"body": {"data": _b64("real body")}}}
    )
    assert full["body"] == "real body"


# --- the list tools ------------------------------------------------------------


def test_get_latest_messages_hydrates_each_id(monkeypatch, fake_service):
    fake = fake_service(
        {"messages": [{"id": "a"}, {"id": "b"}]},
        _message("a", From="one@x.com", Subject="First"),
        _message("b", From="two@x.com", Subject="Second"),
    )
    out = server.gmail_get_latest_messages(max_results=2)
    assert [m["subject"] for m in out] == ["First", "Second"]
    # Metadata only: the list path must not pull full bodies.
    gets = [kw for name, kw in fake.calls if name.endswith(".messages.get")]
    assert all(kw["format"] == "metadata" for kw in gets)


def test_get_latest_messages_asks_for_metadata_not_full(monkeypatch, fake_service):
    fake = fake_service({"messages": [{"id": "a"}]}, _message("a"))
    server.gmail_get_latest_messages()
    get_call = next(kw for name, kw in fake.calls if name.endswith(".messages.get"))
    assert get_call["format"] == "metadata"


def test_search_passes_the_query_through(monkeypatch, fake_service):
    fake = fake_service({"messages": [{"id": "a"}]}, _message("a"))
    server.gmail_search("from:jobs subject:offer is:unread")
    list_call = next(kw for name, kw in fake.calls if name.endswith(".messages.list"))
    assert list_call["q"] == "from:jobs subject:offer is:unread"


@pytest.mark.parametrize(
    "requested,expected", [(0, 1), (-5, 1), (5, 5), (50, 50), (9999, 50)]
)
def test_max_results_is_clamped_to_a_sane_range(
    monkeypatch, fake_service, requested, expected
):
    """The model picks this number. 0 or 9999 must not reach the API."""
    fake = fake_service({"messages": []})
    server.gmail_search("q", max_results=requested)
    list_call = next(kw for name, kw in fake.calls if name.endswith(".messages.list"))
    assert list_call["maxResults"] == expected


def test_a_non_numeric_max_results_becomes_a_readable_error(monkeypatch, fake_service):
    """int() raises ValueError here, which the tool turns into data, not a crash."""
    fake_service({"messages": []})
    out = server.gmail_search("q", max_results="banana")
    assert len(out) == 1 and "error" in out[0]
    assert "banana" in out[0]["error"]


# --- the error contract --------------------------------------------------------


@pytest.mark.parametrize("tool", ["gmail_get_latest_messages", "gmail_search"])
def test_list_tools_return_an_api_error_as_data(monkeypatch, fake_service, tool):
    """The model has to be able to read this, so it is data, not an exception."""
    fake_service(_http_error(500, b"backend exploded"))
    out = getattr(server, tool)() if tool == "gmail_get_latest_messages" else getattr(server, tool)("q")
    assert len(out) == 1 and "error" in out[0]
    assert "500" in out[0]["error"]


@pytest.mark.parametrize("tool", ["gmail_get_latest_messages", "gmail_search"])
def test_list_tools_return_a_missing_credential_as_data(monkeypatch, fake_service, tool):
    def boom():
        raise ValueError("GOOGLE_REFRESH_TOKEN is not set - Gmail tools unavailable")

    monkeypatch.setattr(server, "_service", boom)
    out = getattr(server, tool)() if tool == "gmail_get_latest_messages" else getattr(server, tool)("q")
    assert out == [{"error": "GOOGLE_REFRESH_TOKEN is not set - Gmail tools unavailable"}]


def test_read_returns_an_api_error_as_data(fake_service):
    fake_service(_http_error(404))
    assert "error" in server.gmail_read("nope")


def test_read_returns_headers_and_body_and_asks_for_full_format(monkeypatch, fake_service):
    """The one tool that returns a body, so the only one that must ask for `full`.

    The list tools deliberately fetch `format="metadata"` to keep the prompt small.
    A copy-paste slip to "metadata" here would silently give the model bodiless
    emails that still look like real results.
    """
    fake = fake_service(
        {
            "id": "m1",
            "threadId": "t1",
            "snippet": "snip",
            "payload": {
                "headers": [
                    {"name": "From", "value": "boss@x.com"},
                    {"name": "Subject", "value": "Deadline"},
                ],
                "body": {"data": _b64("Please ship by Friday.")},
            },
        }
    )
    out = server.gmail_read("m1")
    assert out["subject"] == "Deadline"
    assert out["from"] == "boss@x.com"
    assert out["body"] == "Please ship by Friday."
    get_call = next(kw for name, kw in fake.calls if name.endswith(".messages.get"))
    assert get_call["format"] == "full"
    assert get_call["id"] == "m1"


def test_read_walks_a_multipart_body(monkeypatch, fake_service):
    """End to end through the tool: the forwarded-email shape must not come back empty."""
    fake_service(
        {
            "id": "m1",
            "snippet": "see attached",
            "payload": {
                "headers": [],
                "mimeType": "multipart/mixed",
                "parts": [
                    {"mimeType": "application/pdf", "body": {"data": _b64("%PDF-1.4")}},
                    {
                        "mimeType": "multipart/alternative",
                        "parts": [
                            _text_part("the actual message", "text/plain"),
                            _text_part("<b>the actual message</b>", "text/html"),
                        ],
                    },
                ],
            },
        }
    )
    out = server.gmail_read("m1")
    assert "the actual message" in out["body"]


@pytest.mark.parametrize("tool", ["gmail_read", "gmail_get_thread"])
def test_single_message_tools_do_not_crash_on_a_missing_credential(monkeypatch, tool):
    """The bug this file was written for.

    ``gmail_read`` and ``gmail_get_thread`` caught only ``HttpError``, while the
    two list tools caught ``(HttpError, ValueError)``. ``ValueError`` is what
    ``_env`` raises for an unset credential, so a missing credential made these two
    raise out of the tool and the model saw a traceback instead of
    ``{"error": "GOOGLE_REFRESH_TOKEN is not set"}`` -- with no way to recover on
    the next turn.
    """
    def boom():
        raise ValueError("GOOGLE_REFRESH_TOKEN is not set - Gmail tools unavailable")

    monkeypatch.setattr(server, "_service", boom)
    result = getattr(server, tool)("some-id")
    # gmail_read returns one message so it is a dict; gmail_get_thread returns a
    # list so it stays a list. The payload inside is what matters and must match.
    payload = result["error"] if isinstance(result, dict) else result[0]["error"]
    assert payload == "GOOGLE_REFRESH_TOKEN is not set - Gmail tools unavailable"


def test_get_thread_returns_an_api_error_as_data(fake_service):
    fake_service(_http_error(404))
    out = server.gmail_get_thread("nope")
    assert isinstance(out, list), "a thread tool must stay a list even on error"
    assert len(out) == 1 and "error" in out[0]


def test_get_thread_hydrates_every_message_in_order(monkeypatch, fake_service):
    """The docstring promises oldest first; the API returns them in that order
    and the mapping must not reshuffle them."""
    fake_service(
        {
            "messages": [
                _message("m1", From="a@x.com", **{"Subject": "one"}),
                _message("m2", From="b@x.com", **{"Subject": "two"}),
            ]
        }
    )
    out = server.gmail_get_thread("t1")
    assert [m["subject"] for m in out] == ["one", "two"]


def test_an_empty_thread_is_an_empty_list_not_an_error(fake_service):
    fake_service({"messages": []})
    assert server.gmail_get_thread("t1") == []


def test_a_thread_with_a_null_message_list_is_handled(fake_service):
    fake_service({"messages": None})
    assert server.gmail_get_thread("t1") == []


# --- _service -----------------------------------------------------------------


def test_the_client_is_built_once_and_cached(monkeypatch):
    """A refresh costs a network round trip; every tool must share one client."""
    built = []

    class _Creds:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def refresh(self, request):
            built.append("refresh")

    monkeypatch.setattr(server, "Credentials", _Creds)
    monkeypatch.setattr(
        server, "build", lambda *a, **kw: (built.append("build"), "client")[1]
    )
    for name in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"):
        monkeypatch.setenv(name, f"{name}-value")

    server._service_cache = None
    first = server._service()
    second = server._service()

    assert first is second
    assert built.count("build") == 1, "client rebuilt on every call"
    assert built.count("refresh") == 1


def test_the_client_requests_readonly_scope(monkeypatch):
    """A broader scope would be a privilege escalation for a read-only feature."""
    captured = {}

    class _Creds:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def refresh(self, request):
            pass

    monkeypatch.setattr(server, "Credentials", _Creds)
    monkeypatch.setattr(server, "build", lambda *a, **kw: "client")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "the-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "the-secret")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "the-refresh-token")

    server._service_cache = None
    server._service()
    assert captured["scopes"] == ["https://www.googleapis.com/auth/gmail.readonly"]
    assert captured["refresh_token"] == "the-refresh-token"
    assert captured["client_id"] == "the-id"


# --- gmail_tools.build_gmail_tools ---------------------------------------------


@pytest.fixture
def gmail_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "refresh")
    return monkeypatch


def test_no_tools_without_any_credentials(monkeypatch):
    for name in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert gmail_tools.build_gmail_tools() == []


@pytest.mark.parametrize(
    "missing",
    ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"],
)
def test_all_those_vars_are_required_not_just_two(monkeypatch, gmail_env, missing):
    """It is an AND, and it is easy to relax one arm by accident."""
    monkeypatch.setenv(missing, "")
    assert gmail_tools.build_gmail_tools() == []


def test_whitespace_does_not_count_as_configured(monkeypatch, gmail_env):
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "   ")
    assert gmail_tools.build_gmail_tools() == []


def test_a_toolset_is_built_when_all_three_are_set(gmail_env):
    built = gmail_tools.build_gmail_tools()
    assert len(built) == 1


def test_credentials_are_forwarded_to_the_child_process(gmail_env):
    """The whole reason the `env=` dict exists.

    The MCP stdio client inherits only a whitelist (``HOME``, ``PATH``, ...) from
    the parent, so without this the spawned server sees the three vars as unset
    and *every* call fails at runtime with "GOOGLE_REFRESH_TOKEN is not set". It
    would still work in a test, because no test spawned the server.
    """
    env = gmail_tools.build_gmail_tools()[0].connection_params.server_params.env
    assert env["GOOGLE_CLIENT_ID"] == "id"
    assert env["GOOGLE_CLIENT_SECRET"] == "secret"
    assert env["GOOGLE_REFRESH_TOKEN"] == "refresh"


def test_the_child_runs_this_interpreter_against_the_in_repo_server(gmail_env):
    """Anything else and the child either cannot import googleapiclient or
    silently runs a *different* copy of the server."""
    params = gmail_tools.build_gmail_tools()[0].connection_params.server_params
    assert params.command == sys.executable
    assert params.args[0].endswith("gmail_mcp_server.py")
    assert "text_summarizer" in params.args[0]


def test_the_tool_filter_names_exactly_the_four_read_only_tools(gmail_env):
    """A typo in a filter name yields zero tools, silently."""
    toolset = gmail_tools.build_gmail_tools()[0]
    assert set(toolset.tool_filter) == {
        "gmail_get_latest_messages",
        "gmail_search",
        "gmail_read",
        "gmail_get_thread",
    }


def test_no_write_capable_tool_is_exposed(gmail_env):
    """Read-only is a product promise, not a default."""
    for name in gmail_tools.build_gmail_tools()[0].tool_filter:
        assert not any(
            verb in name for verb in ("send", "delete", "trash", "draft", "modify", "label")
        ), name


def test_it_degrades_to_no_tools_when_the_mcp_extra_is_missing(monkeypatch, gmail_env):
    """A missing google-adk[mcp] must disable the feature, not crash the import."""
    monkeypatch.setitem(sys.modules, "mcp", None)
    assert gmail_tools.build_gmail_tools() == []


def test_the_server_module_registers_the_four_tools_the_filter_asks_for():
    """Ties the two halves together.

    ``gmail_tools`` filters by name, so a tool renamed in the server would leave
    the filter pointing at nothing and the agent would silently lose it.
    """
    import inspect

    declared = {
        name
        for name, obj in vars(server).items()
        if inspect.isfunction(obj) and name.startswith("gmail_")
    }
    assert declared == {
        "gmail_get_latest_messages",
        "gmail_search",
        "gmail_read",
        "gmail_get_thread",
    }
