"""The two untested functions behind ``gmail_oauth.py``'s one-time token mint.

This module is excluded from coverage on purpose (``[tool.coverage.run] omit``:
"a one-time interactive browser flow"), and that exclusion is right for
``main()`` -- it opens a browser and reads from stdin. It is wrong for the two
functions underneath it, which are the parts that can fail silently:

* ``_load_client_config`` decides *where the credentials come from*. Getting it
  wrong does not raise: it builds a flow with an empty client id, Google answers
  ``invalid_client`` at the token endpoint, and the user is looking at an OAuth
  error with no way to tell that the helper, not their Google project, is at
  fault.
* ``_exchange_authorization_code`` exists **only** as a workaround.
  ``google_auth_oauthlib``'s ``flow.fetch_token`` raises when the token response
  carries a ``scope`` different from the one requested, and Google echoes every
  scope the client has ever been granted -- so a developer who has already
  consented to Drive cannot mint a Gmail token at all, and the library's error
  says nothing about why. Posting to the token endpoint directly is the fix, and
  a test that only checked the happy path would not notice it being reverted to
  ``fetch_token``: on a fresh Google account the two are indistinguishable, and
  the breakage appears only for the people who need the tool most.

The refresh token this produces is the third of the three variables
``build_gmail_tools()`` requires, so a failure here presents to the user as "the
Gmail tools are not available" with nothing in ``.env`` to explain it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from text_summarizer import gmail_oauth
from text_summarizer.gmail_oauth import (
    TOKEN_URI,
    _exchange_authorization_code,
    _load_client_config,
)

#: The shape Google hands out for a Desktop-app credential. ``installed`` is
#: Google's spelling, not ours, and the other branch below is the one that matters.
DESKTOP_SECRET_FILE = {
    "installed": {
        "client_id": "file-id.apps.googleusercontent.com",
        "client_secret": "file-secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": TOKEN_URI,
        "redirect_uris": ["http://localhost"],
    }
}


# --- fakes --------------------------------------------------------------------


class _Response:
    """The three attributes ``_exchange_authorization_code`` reads off a response."""

    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def _flow(*, redirect_uri="http://localhost", client_config=None, code_verifier="the-verifier"):
    """A stand-in for ``InstalledAppFlow`` with only the attributes that are read.

    ``client_config`` and ``redirect_uri`` are read directly; ``code_verifier`` is
    read with a fallback, because a Desktop-app flow may not carry PKCE.

    ``fetch_token`` is a booby trap. The whole reason this function posts to the
    token endpoint itself is that ``fetch_token`` raises on a scope mismatch, and
    the regression that undoes the workaround is a single line that looks
    perfectly reasonable in review. Calling it here turns that from "fails for
    users who already granted other scopes" into a red test on every machine.
    """

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError(
            "flow.fetch_token() must not be used: it raises whenever Google echoes a "
            "scope other than the one requested, which is exactly the case this "
            "function bypasses"
        )

    return SimpleNamespace(
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        client_config=DESKTOP_SECRET_FILE if client_config is None else client_config,
        fetch_token=_must_not_be_called,
    )


def _capture_post(monkeypatch, response: _Response) -> list[dict]:
    """Replace ``requests.post`` and return the list that records the calls."""
    calls: list[dict] = []

    def fake_post(url, *, data=None, timeout=None, **kwargs):
        calls.append({"url": url, "data": data, "timeout": timeout})
        return response

    monkeypatch.setattr(gmail_oauth.requests, "post", fake_post)
    return calls


# --- _load_client_config -------------------------------------------------------


def test_a_client_secret_file_is_used_verbatim(tmp_path):
    """The file is Google's, so nothing in it is second-guessed.

    A rewrite here would be tempting -- normalising ``redirect_uris``, say -- and
    it would break the Desktop-app clients that ship two or three URIs, or a
    ``web``-type credential with the keys at the top level instead of nested.
    """
    path = tmp_path / "client_secret.json"
    path.write_text(json.dumps(DESKTOP_SECRET_FILE), encoding="utf-8")

    assert _load_client_config(str(path)) == DESKTOP_SECRET_FILE


def test_the_file_wins_over_the_environment(tmp_path, monkeypatch):
    """``if path:`` is checked before the env vars, and that precedence is a decision.

    A developer who has both -- a ``GOOGLE_CLIENT_ID`` in ``.env`` and a freshly
    downloaded ``client_secret.json`` -- should get the file they just pointed at.
    Silently preferring the environment would make the flag look broken.
    """
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    path = tmp_path / "client_secret.json"
    path.write_text(json.dumps(DESKTOP_SECRET_FILE), encoding="utf-8")

    assert _load_client_config(str(path))["installed"]["client_id"].startswith("file-id")


def test_the_environment_is_built_into_a_desktop_app_config(monkeypatch):
    """The no-file path has to produce the same shape the file path does.

    ``InstalledAppFlow.from_client_config`` is handed this dict, so a key it does
    not recognise here is an OAuth failure two steps later with the real API --
    which is why the two branches are asserted to agree rather than just to exist.
    """
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")

    installed = _load_client_config(None)["installed"]

    assert installed["client_id"] == "env-id"
    assert installed["client_secret"] == "env-secret"
    assert installed["token_uri"] == TOKEN_URI, (
        "the same endpoint the exchange function posts to; a divergence would mean "
        "the minted token comes from somewhere other than where it was requested"
    )
    # Google requires a redirect URI to match one registered for the client, and
    # "http://localhost" (no port) is what a Desktop-app client has by default.
    assert installed["redirect_uris"] == ["http://localhost"]


def test_surrounding_whitespace_is_not_a_credential(monkeypatch):
    """``.env`` values arrive with stray spaces more often than not.

    A whitespace-only id would otherwise pass the ``if not client_id`` guard and
    produce a request with ``client_id="   "``.
    """
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "   ")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")

    with pytest.raises(SystemExit):
        _load_client_config(None)


def test_half_a_credential_pair_is_no_credential(monkeypatch):
    """Both halves or neither: Google's error for one is opaque, ours is not."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "")

    with pytest.raises(SystemExit) as excinfo:
        _load_client_config(None)

    message = str(excinfo.value)
    assert "--client-secret-file" in message and "GOOGLE_CLIENT_ID" in message, (
        "the message has to name both ways out, or the user cannot tell which one they meant to use"
    )


def test_no_credential_source_at_all_is_a_hard_stop(monkeypatch):
    """A non-zero exit, not an OAuth attempt with empty credentials.

    ``conftest.py`` pins all three Google vars to ``""`` so this is the state
    every test run starts from; without the raise, a developer following the
    module docstring without either source would get a browser prompt and then
    ``invalid_client``.
    """
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "")

    with pytest.raises(SystemExit):
        _load_client_config(None)


# --- _exchange_authorization_code ----------------------------------------------


def test_the_token_request_carries_every_field_google_requires(monkeypatch):
    """The five arguments are not interchangeable, and Google rejects silently.

    ``redirect_uri`` in particular has to be byte-identical to the one used to
    obtain the code -- a mismatch is ``redirect_uri_mismatch``, and a mismatch
    caused by omitting it looks like a replayed or expired code.
    ``code_verifier`` is sent as ``""`` rather than omitted when the flow has
    none, because a Desktop-app flow registered without PKCE rejects the request
    when the parameter is present and empty on some clients.
    """
    calls = _capture_post(monkeypatch, _Response(payload={"refresh_token": "r"}))

    _exchange_authorization_code(_flow(), "the-code")

    assert len(calls) == 1, "exactly one request, or the code is consumed twice"
    call = calls[0]
    assert call["url"] == TOKEN_URI
    assert call["data"] == {
        "code": "the-code",
        "client_id": "file-id.apps.googleusercontent.com",
        "client_secret": "file-secret",
        "redirect_uri": "http://localhost",
        "grant_type": "authorization_code",
        "code_verifier": "the-verifier",
    }
    # 60s: the console flow has a human in it, so a slow network should not look
    # like a hung process -- but it must not be short enough to fail a retry.
    assert call["timeout"] == 60


def test_a_flow_without_a_code_verifier_sends_an_empty_one(monkeypatch):
    """PKCE is optional here, and its absence must not become a missing key.

    A Desktop-app flow may have no ``code_verifier`` at all. ``flow.code_verifier
    or ""`` exists so the POST body always has the key; sending ``None`` would
    reach Google as the literal string ``"None"``.
    """
    calls = _capture_post(monkeypatch, _Response(payload={"refresh_token": "r"}))

    _exchange_authorization_code(_flow(code_verifier=None), "the-code")

    assert calls[0]["data"]["code_verifier"] == ""


@pytest.mark.parametrize(
    ("client_config", "expected_id", "expected_secret"),
    [
        (DESKTOP_SECRET_FILE, "file-id.apps.googleusercontent.com", "file-secret"),
        # The flat shape, as a Web-application credential is downloaded. Same
        # lookup, different nesting, and picking the wrong one is an
        # ``invalid_client`` from the real API.
        (
            {"client_id": "flat-id", "client_secret": "flat-secret"},
            "flat-id",
            "flat-secret",
        ),
    ],
)
def test_the_client_id_is_read_from_either_config_shape(
    monkeypatch, client_config, expected_id, expected_secret
):
    """``cfg.get("client_id") or cfg["installed"]["client_id"]`` -- both arms.

    The flat key wins when present, which is why this asserts the value rather
    than just the shape: a rewrite that inverted the precedence would pass a test
    that only checked the nested credential.
    """
    calls = _capture_post(monkeypatch, _Response(payload={"refresh_token": "r"}))

    _exchange_authorization_code(_flow(client_config=client_config), "the-code")

    assert calls[0]["data"]["client_id"] == expected_id
    assert calls[0]["data"]["client_secret"] == expected_secret


def test_a_flat_config_that_shadows_the_nested_one_wins(monkeypatch):
    """Both keys present: the flat one is read, deterministically.

    Not a case Google produces, so the behaviour is arbitrary -- but it is
    *chosen*, and pinning it means a future edit has to decide deliberately
    rather than by accident of ``or`` short-circuiting somewhere else.
    """
    calls = _capture_post(monkeypatch, _Response(payload={"refresh_token": "r"}))
    shadowed = {
        "client_id": "flat-id",
        "client_config_flat": True,
        "installed": {"client_id": "nested-id", "client_secret": "nested-secret"},
    }

    _exchange_authorization_code(_flow(client_config=shadowed), "the-code")

    assert calls[0]["data"]["client_id"] == "flat-id"


def test_the_response_is_returned_as_parsed(monkeypatch):
    """The whole dict, not just ``refresh_token``.

    ``main()`` prints ``token["refresh_token"]``, but a caller that also wants
    the scope or the expiry -- which is what tells them whether the account
    consented to read-only or something wider -- can only get it from here.
    """
    payload = {
        "access_token": "a",
        "refresh_token": "r",
        "scope": "https://www.googleapis.com/auth/gmail.readonly",
        "expires_in": 3599,
    }
    _capture_post(monkeypatch, _Response(payload=payload))

    assert _exchange_authorization_code(_flow(), "the-code") == payload


def test_a_rejected_code_stops_with_the_status_and_the_body(monkeypatch):
    """Google's own words, not a summary of them.

    ``invalid_grant`` and ``redirect_uri_mismatch`` need different fixes, and the
    two are indistinguishable from the status code alone.
    """
    _capture_post(
        monkeypatch,
        _Response(status_code=400, text='{"error": "redirect_uri_mismatch"}'),
    )

    with pytest.raises(SystemExit) as excinfo:
        _exchange_authorization_code(_flow(), "the-code")

    message = str(excinfo.value)
    assert "400" in message
    assert "redirect_uri_mismatch" in message


def test_a_200_without_a_refresh_token_stops_rather_than_continuing(monkeypatch):
    """The most confusing response Google sends, made legible.

    It happens when the account was already consented, because Google then omits
    ``refresh_token`` on a re-authorization -- the flow "succeeded" and there is
    nothing to put in ``.env``. Returning the token dict here would hand
    ``main()`` a ``KeyError`` on the very next line, with the response that
    explains the cause already discarded.
    """
    _capture_post(monkeypatch, _Response(payload={"access_token": "a", "expires_in": 3599}))

    with pytest.raises(SystemExit) as excinfo:
        _exchange_authorization_code(_flow(), "the-code")

    assert "refresh_token" in str(excinfo.value)
    assert "access_token" in str(excinfo.value), (
        "the response has to be quoted back: the user cannot diagnose this without it"
    )


def test_a_flow_with_no_redirect_uri_is_a_programming_error(monkeypatch):
    """``RuntimeError``, deliberately not ``SystemExit``.

    Two different failure classes exit by two different routes. Google's
    rejections are user-facing and carry Google's reason, so they end the process
    with an explanation. A flow with no ``redirect_uri`` means ``main()`` failed
    to set it up -- a bug in this file -- and flattening it into ``SystemExit``
    would present a code defect as a credential problem.
    """
    calls = _capture_post(monkeypatch, _Response(payload={"refresh_token": "r"}))

    with pytest.raises(RuntimeError, match="redirect_uri"):
        _exchange_authorization_code(_flow(redirect_uri=None), "the-code")

    assert calls == [], "no request may be sent without a redirect URI to match"


def test_a_minted_token_works_as_the_only_credential_the_tools_need(monkeypatch):
    """The end of the chain, stated as a contract.

    ``build_gmail_tools()`` requires exactly this one value from ``.env``; a token
    that came back without it, or with the wrong scope, leaves the agent
    reporting "no Gmail tools" rather than a Gmail problem. Asserted against
    ``SCOPES`` because ``include_granted_scopes="false"`` in the console flow is
    what keeps the request narrow, and this is the only place that value is
    visible to a test.
    """
    _capture_post(
        monkeypatch,
        _Response(payload={"refresh_token": "r", "scope": " ".join(gmail_oauth.SCOPES)}),
    )

    token = _exchange_authorization_code(_flow(), "the-code")

    assert token["refresh_token"]
    assert token["scope"].split() == gmail_oauth.SCOPES, (
        "the only scope this project ever asks for is read-only Gmail; anything "
        "wider in the response came from the account, not from SCOPES"
    )
