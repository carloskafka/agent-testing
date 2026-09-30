"""Tests for the ``/vault`` route that makes **Sources** titles clickable.

The renderer emits ``/vault/<path>`` links; this is the half that has to serve
them. The interesting property is not that a note is reachable -- it is that the
route is rooted at the *resolved* vault and therefore cannot be walked out of,
because a link the agent renders is a link whose text an LLM chose.

``get_fast_api_app`` is stubbed out: building the real one instantiates the
session/artifact/memory services and touches ``.adk/``, none of which this test
is about.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from text_summarizer import serve


def _stub_adk_app(monkeypatch) -> None:
    """Replace the real ADK app with a bare FastAPI.

    Patched on ``google.adk.cli.fast_api`` rather than on ``serve``, because
    ``build_app`` imports the name at call time -- a patch on the module that
    calls it would be silently ignored.
    """
    import google.adk.cli.fast_api as fast_api

    monkeypatch.setattr(fast_api, "get_fast_api_app", lambda **kwargs: FastAPI())


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A vault, wired in as ``serve.VAULT_ROOT``."""
    (tmp_path / "Topics").mkdir()
    (tmp_path / "Topics" / "Email.md").write_text("# Email\n", encoding="utf-8")
    monkeypatch.setattr(serve, "VAULT_ROOT", str(tmp_path))
    _stub_adk_app(monkeypatch)
    return str(tmp_path)


def test_a_note_is_served_under_the_vault_prefix(vault):
    client = TestClient(serve.build_app(vault, "127.0.0.1", 8000))
    response = client.get("/vault/Topics/Email.md")
    assert response.status_code == 200
    assert response.text == "# Email\n"


def test_a_missing_note_is_a_404_not_the_vault_listing(vault):
    client = TestClient(serve.build_app(vault, "127.0.0.1", 8000))
    assert client.get("/vault/Topics/Nope.md").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/vault/../Dockerfile",
        "/vault/../../etc/passwd",
        "/vault/%2e%2e/%2e%2e/etc/passwd",
        "/vault/..%2f..%2fetc%2fpasswd",
    ],
)
def test_the_route_cannot_be_walked_out_of(vault, path):
    """A link's text comes from the model, so the path must be re-derived.

    ``StaticFiles`` resolves and rejects anything that leaves its root; these
    assert that, because a traversal here would serve the whole container.
    """
    client = TestClient(serve.build_app(vault, "127.0.0.1", 8000))
    response = client.get(path)
    assert response.status_code == 404
    assert "root:" not in response.text


def test_the_vault_is_not_served_when_the_root_is_missing(monkeypatch):
    """A misconfigured mount degrades to unlinked titles, not to a broken server."""
    monkeypatch.setattr(serve, "VAULT_ROOT", "/nonexistent/vault")
    _stub_adk_app(monkeypatch)
    client = TestClient(serve.build_app(".", "127.0.0.1", 8000))
    assert client.get("/vault/Topics/Email.md").status_code == 404


def test_the_prefix_matches_what_the_renderer_emits(vault):
    """The link the renderer writes and the route that serves it must agree.

    Both sides are derived from the same constant precisely so that a rename
    cannot leave the renderer pointing at a path nothing serves.
    """
    from text_summarizer.sources import VAULT_WEB_PREFIX, note_href

    assert note_href("Email", vault).startswith(VAULT_WEB_PREFIX + "/")
    client = TestClient(serve.build_app(vault, "127.0.0.1", 8000))
    assert client.get(note_href("Email", vault)).status_code == 200
