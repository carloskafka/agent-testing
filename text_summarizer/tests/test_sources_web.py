"""The ``[web]`` source kind: parsing, rendering, and the URL-permission rule.

Split out from ``test_sources.py`` rather than appended to it, because every test
here is about the web tier and ``test_sources.py`` is the obsidian-tier suite.
The obsidian assertions in that file are the regression guard that makes these
safe: if adding a kind had changed how an obsidian block renders, they would fail.

Three properties are load-bearing and each has a test below:

* **idempotency** — a rendered block re-parses into the same entries, which is
  what lets ``render_sources`` run twice without growing the answer;
* **one block** — mixed kinds collapse into a single canonical block, because the
  renderer replaces the whole run of source lines rather than appending;
* **only URLs the search returned** — a ``[web]`` line citing anything the tools
  did not return is dropped, which is what stops the tier printing invented URLs.
"""

from __future__ import annotations

import pytest
from text_summarizer.sources import (
    MODEL_TOKEN,
    SOURCES_HEADING,
    VAULT_TOKEN,
    WEB_TOKEN,
    normalised_url,
    render_sources,
    strip_sources_block,
    summary_only,
)

ALLOWED = ["https://example.com/p", "https://en.wikipedia.org/wiki/Dog"]

KW = {"vault_name": "ck", "model_name": "gemini", "web_provider": "searxng"}


def _obsidian_line(title: str = "Dogs Overview", reason: str = "same topic") -> str:
    return f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[{title}]]: {reason}"


def _web_line(url: str, reason: str = "cited claim") -> str:
    return f"[web][{WEB_TOKEN}][{MODEL_TOKEN}]<{url}>: {reason}"


def _render(raw: str, allowed=ALLOWED, **overrides):
    kwargs = dict(KW)
    kwargs.update(overrides)
    return render_sources(raw, allowed_web_urls=allowed, **kwargs)


# --- parsing ------------------------------------------------------------------


def test_web_line_parses_to_kind_url_and_reason():
    out = _render("- a bullet\n" + _web_line("https://example.com/p"))
    assert "- [web][searxng][gemini]<https://example.com/p>: cited claim" in out


def test_bare_url_without_angle_brackets_parses():
    line = f"[web][{WEB_TOKEN}][{MODEL_TOKEN}]https://example.com/p: bare"
    out = _render("- x\n" + line)
    assert "<https://example.com/p>" in out
    assert "bare" in out


def test_web_line_without_a_kind_tag_still_parses():
    """A model that omits ``[web]`` must still round-trip, not be dropped."""
    out = _render(f"- x\n[{VAULT_TOKEN}][{MODEL_TOKEN}]<https://example.com/p>: untagged")
    assert "- [web][searxng][gemini]<https://example.com/p>" in out


def test_obsidian_line_is_unaffected():
    out = _render(f"- x\n{_obsidian_line()}")
    assert "- [obsidian][ck][gemini][[Dogs Overview]]: same topic" in out


def test_a_vault_note_titled_like_a_url_stays_a_vault_note():
    """``[[https://…]]`` is a wikilink, so it is a vault target by definition.

    Treating it as a web citation would drop it, because the URL would not be in
    the permitted set -- a real note lost to a formatting coincidence.
    """
    raw = f"- x\n[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[https://example.com/p]]: about a url"
    out = _render(raw, allowed=[])
    assert "[[https://example.com/p]]" in out
    assert "[web]" not in out


def test_reason_is_preserved_for_both_kinds():
    out = _render(
        f"- x\n{_obsidian_line(reason='vault reason')}\n{_web_line('https://example.com/p', 'web reason')}"
    )
    assert "[[Dogs Overview]]: vault reason" in out
    assert "<https://example.com/p>: web reason" in out


# --- idempotency and the single block ----------------------------------------


def test_rendering_is_idempotent_on_a_mixed_block():
    raw = "\n".join(["- x", _obsidian_line(), _web_line("https://example.com/p")])
    once = _render(raw)
    assert _render(once) == once
    assert _render(_render(once)) == once


def test_mixed_kinds_produce_exactly_one_block():
    raw = "\n".join(["- x", _obsidian_line(), _web_line("https://example.com/p")])
    assert _render(raw).count(SOURCES_HEADING) == 1


def test_one_bullet_per_source():
    raw = "\n".join(
        [
            "- x",
            _obsidian_line(),
            _web_line("https://example.com/p"),
            _web_line("https://en.wikipedia.org/wiki/Dog"),
        ]
    )
    out = _render(raw)
    assert sum(1 for line in out.split("\n") if line.startswith("- [")) == 3


def test_a_second_heading_does_not_duplicate_the_block():
    raw = "\n".join(
        [
            "- x",
            SOURCES_HEADING,
            _obsidian_line(),
            SOURCES_HEADING,
            _web_line("https://example.com/p"),
        ]
    )
    assert _render(raw).count(SOURCES_HEADING) == 1


def test_obsidian_rendering_is_byte_identical_without_web():
    """The regression guard: adding a kind must not move an obsidian block.

    ``test_sources.py`` covers the obsidian grammar in depth; this is the
    assertion that the web work left it alone, including the backslash-escaped
    linked spelling and the reason separator.
    """
    raw = f"- x\n[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: same topic"
    expected = "- x\n**Sources**\n- [obsidian][ck][gemini][[Dogs Overview]]: same topic"
    assert _render(raw, vault_root=None) == expected


def test_obsidian_href_spelling_is_unchanged(tmp_path):
    vault = tmp_path / "Second Brain"
    vault.mkdir()
    (vault / "Dogs.md").write_text("---\naliases:\n  - Dogs Overview\n---\nbody\n")
    raw = f"- x\n[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: same topic"
    out = _render(raw, vault_root=str(tmp_path))
    assert r"[\[\[Dogs Overview\]\]](/vault/Second%20Brain/Dogs.md)" in out


# --- dedupe -------------------------------------------------------------------


def test_the_same_url_twice_collapses_to_one_line():
    raw = "\n".join(
        [
            "- x",
            _web_line("https://example.com/p", "first"),
            _web_line("https://example.com/p", "second"),
        ]
    )
    out = _render(raw)
    assert out.count("<https://example.com/p>") == 1
    assert "first" in out and "second" not in out


def test_a_url_and_a_note_of_the_same_name_are_kept_separate():
    """Dedupe is keyed on ``(kind, target)``, so the two do not collapse."""
    raw = "\n".join(
        [
            "- x",
            _web_line("https://example.com/p", "web"),
            f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[https://example.com/p]]: vault",
        ]
    )
    out = _render(raw)
    assert "[web][searxng]" in out
    assert "[[https://example.com/p]]" in out


# --- the permission rule ------------------------------------------------------


def test_a_url_the_search_never_returned_is_dropped():
    out = _render("- x\n" + _web_line("https://invented.example/never-seen"))
    assert "invented.example" not in out


def test_all_web_lines_are_dropped_when_nothing_was_returned():
    raw = "- x\n" + _web_line("https://example.com/p")
    assert "[web]" not in _render(raw, allowed=[])


def test_an_empty_permitted_set_means_no_web_citations_at_all():
    """The cache-hit / tools-absent case: no search ran, so nothing may be cited."""
    raw = "\n".join(["- x", _obsidian_line(), _web_line("https://example.com/p")])
    out = _render(raw, allowed=[])
    assert "[web]" not in out
    assert "[[Dogs Overview]]" in out


def test_an_obsidian_line_survives_even_with_an_empty_permitted_set():
    raw = f"- x\n{_obsidian_line()}"
    assert "[[Dogs Overview]]" in _render(raw, allowed=[])


def test_omitting_the_argument_accepts_every_web_line():
    """``None`` means "do not filter", which is what the unit tests of the renderer
    elsewhere in the suite rely on; an empty *collection* means "allow none"."""
    raw = "- x\n" + _web_line("https://anything.example/")
    out = render_sources(raw, allowed_web_urls=None, **KW)
    assert "[web][searxng]" in out


@pytest.mark.parametrize(
    "spelling",
    [
        "https://example.com/p",
        "https://example.com/p/",
        "https://EXAMPLE.com/p",
        "https://example.com/p?utm_source=news",
        "https://example.com/p#section",
    ],
)
def test_permitted_url_matching_tolerates_ordinary_spelling_differences(spelling):
    """A false negative here would silently drop a real citation."""
    out = _render("- x\n" + _web_line(spelling), allowed=["https://example.com/p"])
    assert "[web][searxng]" in out, f"{spelling} should have matched"


def test_a_different_path_is_still_refused():
    out = _render(
        "- x\n" + _web_line("https://example.com/other"), allowed=["https://example.com/p"]
    )
    assert "[web]" not in out


def test_a_non_http_web_line_is_dropped_even_if_listed():
    out = _render(
        f"- x\n[web][{WEB_TOKEN}][{MODEL_TOKEN}]<javascript:alert(1)>: x",
        allowed=["javascript:alert(1)"],
    )
    assert "javascript" not in out


# --- summary_only must strip web lines too ------------------------------------


def test_summary_only_removes_web_lines():
    """Otherwise web citations inflate ``quality.bullet_count``."""
    raw = "\n".join(
        [
            "- Dogs are domesticated mammals.",
            SOURCES_HEADING,
            _obsidian_line(),
            _web_line("https://example.com/p"),
        ]
    )
    stripped = summary_only(_render(raw))
    assert "- Dogs are domesticated mammals." in stripped
    assert "[web]" not in stripped
    assert "[[Dogs Overview]]" not in stripped
    assert stripped.count("\n- ") == 0


def test_summary_only_leaves_text_after_the_block():
    raw = "\n".join(
        [
            "- x",
            SOURCES_HEADING,
            _web_line("https://example.com/p"),
            "",
            'Saved to the second brain as "Dogs Summary".',
        ]
    )
    stripped = summary_only(_render(raw))
    assert 'Saved to the second brain as "Dogs Summary".' in stripped


def test_strip_and_render_agree_on_the_block_extent():
    raw = "\n".join(["- x", SOURCES_HEADING, _web_line("https://example.com/p"), "tail"])
    rendered = _render(raw)
    assert SOURCES_HEADING not in strip_sources_block(rendered)
    assert "tail" in strip_sources_block(rendered)


# --- web URLs never reach the vault index -------------------------------------


def test_a_web_url_is_not_resolved_against_the_vault(tmp_path):
    """``note_href`` is a title -> vault-path derivation, pinned against traversal.

    A web citation's URL is its own and there is nothing to recompute, so it must
    never be passed through that index -- otherwise a URL matching a note title
    would be turned into a link to a note that does not exist.
    """
    vault = tmp_path / "Second Brain"
    vault.mkdir()
    (vault / "evil.md").write_text("body\n")
    url = "https://evil.md/"
    raw = "- x\n" + _web_line(url)
    out = _render(raw, allowed=[url], vault_root=str(tmp_path))
    assert f"<{url}>" in out
    assert "/vault/" not in out


def test_web_url_is_emitted_verbatim_not_percent_encoded():
    """A URL is already a URL; re-encoding it would change what it points at."""
    url = "https://example.com/a b?q=1&r=2"
    out = _render("- x\n" + _web_line(url), allowed=[url])
    assert f"<{url}>" in out


def test_normalised_url_is_the_canonical_spelling():
    assert normalised_url("https://Example.COM/") == "https://example.com/"
    assert normalised_url("https://example.com") == "https://example.com/"
    assert normalised_url("https://example.com/p/") == "https://example.com/p"
    assert normalised_url("https://example.com/p#frag") == "https://example.com/p"


@pytest.mark.parametrize("bad", ["", "javascript:alert(1)", "file:///etc/passwd", "not a url"])
def test_normalised_url_rejects_anything_that_is_not_an_http_url(bad):
    assert normalised_url(bad) == ""
