"""Unit tests for the deterministic ``**Sources**`` renderer.

The renderer is the only thing standing between an LLM's free-form output and
the project's provenance guarantee, so it is tested as pure text-in/text-out:
no LLM call, no vault, no network.

Run with::

    cd ~/Downloads/apps/agent-testing
    PYTHONPATH=/tmp/opencode/testlibs LANGFUSE_PUBLIC_KEY= \
      text_summarizer/.venv/bin/python -m pytest text_summarizer/tests -q
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from _helpers import SERVED_MODEL, _event, _point_vault_at
from text_summarizer.sources import (
    MODEL_TOKEN,
    SOURCES_HEADING,
    UNKNOWN,
    VAULT_TOKEN,
    note_href,
    render_sources,
    resolve_vault_name,
    served_model_from_events,
    strip_sources_block,
)

BULLETS = "- Dogs are domesticated mammals.\n- They come in many breeds.\n- Dogs are social animals."


# --- normal substitution -----------------------------------------------------


def test_substitutes_vault_and_model_into_single_source():
    model_text = (
        f"{BULLETS}\n\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: same topic"
    )
    out = render_sources(
        model_text, vault_name="ck", model_name="gemini-3.5-flash-lite"
    )
    assert out == (
        f"{BULLETS}\n\n"
        f"{SOURCES_HEADING}\n"
        "- [obsidian][ck][gemini-3.5-flash-lite][[Dogs Overview]]: same topic"
    )


def test_substitutes_multi_source_block_in_place():
    model_text = (
        f"{BULLETS}\n\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Wolves]]: related canid\n\n"
        'Saved to the second brain as "Dogs Summary".'
    )
    out = render_sources(
        model_text, vault_name="ck", model_name="openrouter/google/gemma-4-26b-a4b-it:free"
    )
    assert out == (
        f"{BULLETS}\n\n"
        f"{SOURCES_HEADING}\n"
        "- [obsidian][ck][openrouter/google/gemma-4-26b-a4b-it:free][[Dogs Overview]]: primary\n"
        "- [obsidian][ck][openrouter/google/gemma-4-26b-a4b-it:free][[Wolves]]: related canid\n\n"
        'Saved to the second brain as "Dogs Summary".'
    )


def test_each_source_is_its_own_bullet():
    """One list item per note, whatever shape the model wrote them in.

    Bare lines collapse into a single run-on paragraph in the chat UI, which is
    what the block looked like before the bullet was part of the format.
    """
    model_text = (
        f"{BULLETS}\n\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Wolves]]: related canid\n"
    )
    out = render_sources(model_text, vault_name="ck", model_name="m1")
    source_lines = out.split(f"{SOURCES_HEADING}\n", 1)[1].splitlines()
    assert source_lines == [
        "- [obsidian][ck][m1][[Dogs Overview]]: primary",
        "- [obsidian][ck][m1][[Wolves]]: related canid",
    ]


def test_rendering_is_idempotent():
    """Re-rendering an already-rendered block must not change it or double it.

    The block is re-parsed and re-emitted on every render, so a second pass has
    to recognise its own bulleted output -- the case that matters once a response
    is stored and then read back.
    """
    model_text = (
        f"{BULLETS}\n\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary\n"
    )
    once = render_sources(model_text, vault_name="ck", model_name="m1")
    assert render_sources(once, vault_name="ck", model_name="m1") == once
    assert once.count(SOURCES_HEADING) == 1


def test_model_emitted_heading_is_replaced_not_duplicated():
    """The model is told not to write a heading; if it does, there is still one."""
    model_text = (
        f"{BULLETS}\n\n"
        "## Sources\n"
        f"- [obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary\n"
    )
    out = render_sources(model_text, vault_name="ck", model_name="m1")
    assert out.count(SOURCES_HEADING) == 1
    assert "## Sources" not in out
    assert out.endswith("- [obsidian][ck][m1][[Dogs Overview]]: primary\n")


def test_duplicate_headings_collapse_to_one_block():
    model_text = (
        f"{BULLETS}\n\n"
        "**Sources**\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary\n\n"
        "## Sources\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Wolves]]: related canid\n"
    )
    out = render_sources(model_text, vault_name="ck", model_name="m1")
    assert out.count(SOURCES_HEADING) == 1
    assert "[[Dogs Overview]]" in out
    assert "[[Wolves]]" in out


def test_legacy_vault_prefixed_lines_are_normalized():
    """Pre-existing `## Sources` + `[vault] [[Note]]` output is adopted, not lost."""
    model_text = (
        f"{BULLETS}\n\n"
        "## Sources\n"
        "[vault] [[Dogs Overview]]\n"
        "- [[Wolves]]: related canid\n"
    )
    out = render_sources(model_text, vault_name="ck", model_name="m1")
    assert out == (
        f"{BULLETS}\n\n"
        f"{SOURCES_HEADING}\n"
        "- [obsidian][ck][m1][[Dogs Overview]]\n"
        "- [obsidian][ck][m1][[Wolves]]: related canid\n"
    )


def test_duplicate_notes_are_deduped_case_insensitively():
    model_text = (
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: a\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[dogs overview]]: b\n"
    )
    out = render_sources(model_text, vault_name="ck", model_name="m1")
    assert out.count("[[Dogs Overview]]") == 1
    assert "[[dogs overview]]" not in out


# --- unknown / missing provenance --------------------------------------------


def test_unknown_model_renders_as_literal_unknown_not_a_guess():
    model_text = f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary"
    out = render_sources(model_text, vault_name="ck", model_name=None)
    assert out.endswith("- [obsidian][ck][unknown][[Dogs Overview]]: primary")
    assert UNKNOWN in out
    # Nothing plausible was invented.
    assert "gemini" not in out and "gemma" not in out and "qwen" not in out


def test_unknown_vault_and_model_together():
    model_text = f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary"
    out = render_sources(model_text, vault_name="", model_name="")
    assert out.endswith("- [obsidian][unknown][unknown][[Dogs Overview]]: primary")


def test_served_model_is_read_from_the_last_complete_model_event():
    events = [
        _event("tool call step", model_version="first-model"),
        _event("intermediate", model_version="first-model"),
        _event("final answer", model_version="openrouter/qwen/qwen3.8-27b:free"),
    ]
    assert served_model_from_events(events) == "openrouter/qwen/qwen3.8-27b:free"


def test_served_model_is_none_for_a_cache_replay():
    """A cached LlmResponse carries no model_version -- that is the signal."""
    events = [_event("replayed cached text", model_version=None)]
    assert served_model_from_events(events) is None
    out = render_sources(
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary",
        vault_name="ck",
        model_name=served_model_from_events(events),
    )
    assert "- [obsidian][ck][unknown][[Dogs Overview]]: primary" in out


def test_served_model_ignores_user_and_empty_events():
    events = [
        _event("the prompt", author="user", model_version="not-a-model"),
        _event("", model_version="also-not-a-model"),
        _event("answer", model_version="real-model"),
    ]
    assert served_model_from_events(events) == "real-model"


def test_stray_tokens_in_prose_are_scrubbed():
    out = render_sources(
        f"{BULLETS} mentioning {VAULT_TOKEN} and {MODEL_TOKEN} inline.",
        vault_name="ck",
        model_name="m1",
    )
    assert VAULT_TOKEN not in out
    assert MODEL_TOKEN not in out
    assert out == f"{BULLETS} mentioning and inline."


# --- no-op paths -------------------------------------------------------------


def test_text_without_a_sources_block_is_returned_unchanged():
    out = render_sources(BULLETS, vault_name="ck", model_name="m1")
    assert out == BULLETS


def test_empty_text_is_returned_unchanged():
    assert render_sources("", vault_name="ck", model_name="m1") == ""
    assert render_sources("   \n ", vault_name="ck", model_name="m1") == "   \n "


def test_heading_without_usable_source_line_is_dropped():
    model_text = f"{BULLETS}\n\n**Sources**\n\nSaved to the second brain."
    out = render_sources(model_text, vault_name="ck", model_name="m1")
    assert SOURCES_HEADING not in out
    assert "Saved to the second brain." in out


def test_strip_sources_block_keeps_text_after_the_block():
    rendered = (
        f"{BULLETS}\n\n"
        f"{SOURCES_HEADING}\n"
        "- [obsidian][ck][m1][[Dogs Overview]]: primary\n\n"
        'Saved to the second brain as "Dogs Summary".'
    )
    assert strip_sources_block(rendered) == (
        f"{BULLETS}\n\nSaved to the second brain as \"Dogs Summary\"."
    )


def test_strip_sources_block_lets_metrics_ignore_source_lines():
    """Source lines must not be counted as summary bullets.

    This matters more now that source lines *are* bullets: without the strip,
    every cited note would inflate the bullet count (and so ``quality.bullet_score``).
    """
    rendered = (
        f"{BULLETS}\n\n{SOURCES_HEADING}\n"
        "- [obsidian][ck][m1][[Dogs Overview]]: x\n"
        "- [obsidian][ck][m1][[Wolves]]: y"
    )
    assert strip_sources_block(rendered).count("\n- ") == 2


# --- vault name resolution ---------------------------------------------------


def test_vault_name_from_env_wins():
    identity = resolve_vault_name(
        env={"VAULT_NAME": "ck", "SECOND_BRAIN_VAULT": "/vault"},
        mcp_reported="vault",
    )
    assert identity.name == "ck"
    assert identity.source == "VAULT_NAME"


def test_vault_name_falls_back_to_the_mcp_reported_name():
    identity = resolve_vault_name(
        env={"SECOND_BRAIN_VAULT": "/somewhere/else"}, mcp_reported="ck"
    )
    assert identity.name == "ck"
    assert identity.source == "mcp:vault_info"


def test_vault_name_falls_back_to_the_vault_root_basename(tmp_path):
    vault = tmp_path / "ck"
    vault.mkdir()
    identity = resolve_vault_name(env={}, vault_root=str(vault))
    assert identity.name == "ck"
    assert identity.source == "vault-path"


def test_vault_name_is_unknown_when_the_root_does_not_exist(tmp_path):
    """A configured path that isn't a directory must not render its last segment.

    Guards against a typo'd or missing mount silently labeling every source line
    with a name that looks real but is not.
    """
    identity = resolve_vault_name(env={}, vault_root=str(tmp_path / "not-mounted"))
    assert identity.name == UNKNOWN
    assert identity.source == "unknown"


def test_vault_name_is_unknown_for_a_bare_root_path():
    identity = resolve_vault_name(env={}, vault_root="/")
    assert identity.name == UNKNOWN
    assert identity.source == "unknown"


def test_vault_name_is_sanitized_for_the_token_grammar():
    identity = resolve_vault_name(env={"VAULT_NAME": "[ck]"})
    assert identity.name == "ck"
    identity = resolve_vault_name(env={"VAULT_NAME": "my vault"})
    assert identity.name == "my-vault"


def test_mcp_vault_info_payload_shapes():
    from text_summarizer.sources import mcp_vault_name_from_events

    payload = json.dumps(
        {"vault_name": "ck", "vault_path": "/tmp/opencode/vault-probe"}
    )
    for wrapped in (
        {"result": payload},
        {"result": {"vault_name": "ck"}},
        payload,
    ):
        part = SimpleNamespace(text=None, function_response=SimpleNamespace(
            name="vault_info", response=wrapped
        ))
        assert mcp_vault_name_from_events([_event(parts=[part])]) == "ck"

    other = SimpleNamespace(text=None, function_response=SimpleNamespace(
        name="vault_list", response={"result": "[]"}
    ))
    assert mcp_vault_name_from_events([_event(parts=[other])]) is None
    assert mcp_vault_name_from_events([]) is None


# --- note frontmatter provenance --------------------------------------------


def test_a_pre_existing_note_never_gains_an_invented_model(monkeypatch, tmp_path):
    """A note written before provenance existed must render as ``unknown``.

    The previous version of this test built a ``legacy`` frontmatter string, asserted
    that the word ``generated_by_model`` was absent from *its own literal*, and then
    never passed it to anything. It could not fail: the interesting behaviour --
    "what does the system do when it reads such a note?" -- was never exercised.

    So this walks the real path with a real note on disk:

    1. ``find_cached_summary`` finds it by fingerprint and replays the body;
    2. nothing back-fills the frontmatter -- the note is byte-identical afterwards,
       so a model that happens to be serving now cannot claim it;
    3. the provenance resolved for that replay has no model, because a cache hit
       short-circuits *before* the model is called and so the session holds no
       model event at all;
    4. and that absence is what makes the renderer say ``unknown`` rather than the
       currently-serving model. Step 5 is the anchor: the same resolver, given one
       model event, does return a name -- so "unknown" is a fact about this input,
       not a constant.
    """
    from text_summarizer import second_brain

    vault = Path(_point_vault_at(monkeypatch, tmp_path))
    brain = vault / second_brain.BRAIN_DIR
    brain.mkdir(parents=True, exist_ok=True)

    prompt = "Summarize: dogs are social animals."
    # Step 1 asserts this note is *reachable by the cache*, and the cache refuses
    # anything older than `CACHE_MAX_AGE_DAYS`. A hardcoded filename made the
    # test expire on a calendar rather than on a behaviour -- it passed until the
    # date rolled past the window, then failed here and in two unrelated files.
    today = date.today().isoformat()
    legacy = (
        f"---\ntags:\n  - Dogs\ndate: {today}\n"
        f"source_fingerprint: {second_brain.source_fingerprint(prompt)}\n"
        "aliases:\n  - Dogs\n---\n\n"
        "- Dogs are social animals.\n\n## From the vault\n[[Dogs Overview]]\n"
        "\n## Related\n- [[Dogs]]\n"
    )
    note = brain / f"{today} - dogs.md"
    note.write_text(legacy, encoding="utf-8")

    # 1. The real lookup: this note is reachable by the cache.
    replayed = second_brain.find_cached_summary(prompt)
    assert replayed is not None
    assert replayed.startswith("- Dogs are social animals.")
    assert "generated_by_model" not in replayed, "the frontmatter is stripped on replay"

    # 2. Never back-filled.
    assert note.read_text(encoding="utf-8") == legacy

    # 3. A cache hit short-circuits in before_model_callback, so the session holds
    #    the prompt and the replayed text and no model event whatsoever.
    replay_events = [
        _event(prompt, author="user"),
        _event(replayed, model_version=None),
    ]
    assert served_model_from_events(replay_events) is None
    provenance = second_brain.note_provenance(_session(replay_events))
    assert "generated_by_model" not in provenance

    # 4. Which is exactly what the renderer turns into `unknown`.
    out = render_sources(
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[{today} - dogs]]: pre-existing",
        vault_name="ck",
        model_name=served_model_from_events(replay_events),
    )
    assert out.endswith(f"- [obsidian][ck][unknown][[{today} - dogs]]: pre-existing")

    # 5. The anchor. One model event on the session and the same resolver names the
    #    backend, so step 3 is a property of a replay -- not a stub that always
    #    answers `unknown`.
    served_events = replay_events + [_event("- a fresh answer", model_version=SERVED_MODEL)]
    assert served_model_from_events(served_events) == SERVED_MODEL
    backfilled = render_sources(
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[{today} - dogs]]: pre-existing",
        vault_name="ck",
        model_name=served_model_from_events(served_events),
    )
    assert backfilled.endswith(
        f"- [obsidian][ck][{SERVED_MODEL}][[{today} - dogs]]: pre-existing"
    )
    assert backfilled != out, "the two cases must not render identically"


def _session(events) -> SimpleNamespace:
    """A ``ToolContext``-shaped object exposing only ``session.events``.

    That is the entire surface ``second_brain._session_events`` reads, and the
    whole point of the shape is that it is *only* that: a real ``ToolContext`` also
    carries ``state``, ``functions`` and a live invocation, none of which
    ``note_provenance`` looks at.
    """
    return SimpleNamespace(session=SimpleNamespace(events=list(events)))


@pytest.mark.parametrize("tool_context", [None, object()])
def test_note_provenance_omits_unresolvable_fields(tool_context):
    from text_summarizer.second_brain import note_provenance

    provenance = note_provenance(tool_context)
    assert "generated_by_model" not in provenance
    # The vault is still resolvable from configuration alone.
    assert provenance.get("generated_in_vault")


# --- clickable source titles -------------------------------------------------


def linked(title: str, href: str) -> str:
    """The exact markdown the renderer emits for a clickable note title.

    Built from ``chr(92)`` and concatenated rather than written as a literal:
    the escaping is the whole point of the format, and a hand-written ``\\\\[``
    hides the off-by-one that costs a bracket. What it produces is::

        [\\[\\[Note\\]\\](/vault/Topics/Note.md)

    i.e. a markdown link whose text is the escaped wikilink, so the brackets
    reach the screen instead of being eaten as link syntax.
    """
    bs = chr(92)
    return "".join(
        (
            "[", bs, "[",          # [ + escaped [
            bs, "[", title,        # escaped [ + the note title
            bs, "]", bs, "]",      # escaped ] + escaped ]
            "]",                   # closes the markdown link
            "(", href, ")",
        )
    )


@pytest.fixture
def vault(tmp_path):
    """A small vault: a topic note, a dated note, and an aliased note."""
    (tmp_path / "Topics").mkdir()
    (tmp_path / "Topics" / "Email.md").write_text("# Email\n", encoding="utf-8")
    brain = tmp_path / "Second Brain"
    brain.mkdir()
    (brain / "2026-09-25 - dogs.md").write_text("# Dogs\n", encoding="utf-8")
    (brain / "2026-09-26 - voyager-1-interstellar-space-mission.md").write_text(
        "---\n"
        "tags:\n  - Space\n"
        "date: 2026-09-26\n"
        "aliases:\n"
        "  - Voyager 1 Interstellar Space Mission\n"
        "  - Voyager 1\n"
        "---\n\n"
        "# Voyager 1\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def test_a_resolved_title_renders_a_link(vault):
    model_text = f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Email]]: covers inbox"
    out = render_sources(
        model_text, vault_name="ck", model_name="m1", vault_root=vault
    )
    assert out.endswith(
        f"- [obsidian][ck][m1]{linked('Email', '/vault/Topics/Email.md')}: covers inbox"
    )


def test_a_link_still_displays_the_wikilink_brackets(vault):
    """The visible text has to stay ``[[Email]]``, or the format is not preserved.

    The brackets are backslash-escaped so the markdown parser shows them instead
    of consuming them as link syntax. Drop the escaping and the title still
    links, but silently reads as ``[Email]``.

    Asserted on the *link text* -- the span between the link's own delimiters --
    because that is what a CommonMark renderer puts inside the ``<a>``. The
    delimiters are not displayed at all, so unescaping the whole line would be
    asserting the wrong thing.
    """
    model_text = f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Email]]: covers inbox"
    out = render_sources(
        model_text, vault_name="ck", model_name="m1", vault_root=vault
    )
    emitted = out.split("[m1]", 1)[1].split(":")[0]
    assert emitted == linked("Email", "/vault/Topics/Email.md")

    link_text = emitted[1 : emitted.rindex("](")]
    assert link_text.replace("\\", "") == "[[Email]]"


def test_an_unresolvable_title_stays_a_plain_wikilink(vault):
    """A title with no file behind it must not become a link that 404s."""
    model_text = f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Ghost Note]]: hallucinated"
    out = render_sources(
        model_text, vault_name="ck", model_name="m1", vault_root=vault
    )
    assert out.endswith("- [obsidian][ck][m1][[Ghost Note]]: hallucinated")
    assert "/vault/" not in out


def test_no_vault_root_means_no_links():
    """With no readable vault every title stays unlinked, as it was before."""
    model_text = f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Email]]: covers inbox"
    out = render_sources(model_text, vault_name="ck", model_name="m1")
    assert out.endswith("- [obsidian][ck][m1][[Email]]: covers inbox")


def test_titles_in_subdirectories_link_to_their_path(vault):
    model_text = (
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}]"
        "[[2026-09-25 - dogs]]: primary"
    )
    out = render_sources(
        model_text, vault_name="ck", model_name="m1", vault_root=vault
    )
    # The space is percent-encoded, or the browser truncates the URL there.
    assert "(/vault/Second%20Brain/2026-09-25%20-%20dogs.md)" in out


def test_title_lookup_ignores_case_but_not_spelling(vault):
    assert note_href("email", vault) == "/vault/Topics/Email.md"
    assert note_href("Emai", vault) is None


def test_rendering_is_idempotent_with_links(vault):
    """A linked block must re-parse into the same entries, links and all.

    The href is dropped on re-parse and recomputed from the vault, so a note
    that is renamed or moved re-points instead of keeping a dead path.
    """
    model_text = (
        f"{BULLETS}\n\n"
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Email]]: covers inbox\n"
    )
    once = render_sources(
        model_text, vault_name="ck", model_name="m1", vault_root=vault
    )
    twice = render_sources(once, vault_name="ck", model_name="m1", vault_root=vault)
    assert once == twice
    assert once.count(SOURCES_HEADING) == 1
    assert once.count("/vault/Topics/Email.md") == 1


def test_a_renamed_note_repoints_instead_of_keeping_a_dead_link(vault, tmp_path):
    model_text = f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Email]]: covers inbox"
    once = render_sources(
        model_text, vault_name="ck", model_name="m1", vault_root=vault
    )
    (tmp_path / "Topics" / "Email.md").rename(tmp_path / "Topics" / "Email (old).md")
    (tmp_path / "Topics" / "Email.md").write_text("# Email\n", encoding="utf-8")
    twice = render_sources(once, vault_name="ck", model_name="m1", vault_root=vault)
    assert twice == once  # same title, same resolved file -> same link


def test_note_href_refuses_to_walk_out_of_the_vault(vault):
    """Titles are matched against real files, so no path can be injected."""
    assert note_href("../../../etc/passwd", vault) is None
    assert note_href("", vault) is None
    assert note_href("Email", None) is None
    assert note_href("Email", "/nonexistent/vault") is None


def test_duplicate_titles_resolve_deterministically(tmp_path):
    """Obsidian allows the same note name in two folders; the link must not flap."""
    (tmp_path / "Topics").mkdir()
    (tmp_path / "Archive").mkdir()
    (tmp_path / "Topics" / "Email.md").write_text("a", encoding="utf-8")
    (tmp_path / "Archive" / "Email.md").write_text("b", encoding="utf-8")
    root = str(tmp_path)
    assert note_href("Email", root) == note_href("Email", root)


# --- alias resolution --------------------------------------------------------


def test_a_cited_alias_links_to_the_note_that_declares_it(vault):
    """The model cites the note's alias, not its ``<date> - <slug>`` filename.

    Every note this agent writes carries ``aliases: [<the title the model
    chose>]`` while the file is named after a slug of it, so the readable name
    the model cites and the name on disk are different strings. Obsidian
    resolves both; an index built from filenames alone left most citations
    unlinked. Found live: a turn citing
    ``[[Voyager 1 Interstellar Space Mission]]`` produced no link until the
    alias was indexed.
    """
    model_text = (
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}]"
        "[[Voyager 1 Interstellar Space Mission]]: prior note"
    )
    out = render_sources(
        model_text, vault_name="ck", model_name="m1", vault_root=vault
    )
    assert out.endswith(
        linked(
            "Voyager 1 Interstellar Space Mission",
            "/vault/Second%20Brain/2026-09-26%20-%20voyager-1-interstellar-space-mission.md",
        )
        + ": prior note"
    )
    # The filename spelling resolves to the same note.
    by_name = render_sources(
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}]"
        "[[2026-09-26 - voyager-1-interstellar-space-mission]]: by name",
        vault_name="ck",
        model_name="m1",
        vault_root=vault,
    )
    assert "2026-09-26%20-%20voyager-1-interstellar-space-mission.md" in by_name


def test_an_inline_alias_list_is_understood(tmp_path):
    """``aliases: [A, B]`` is valid YAML and valid Obsidian."""
    (tmp_path / "n.md").write_text("---\naliases: [Inline One, Inline Two]\n---\n")
    assert note_href("Inline One", str(tmp_path)) == "/vault/n.md"
    assert note_href("Inline Two", str(tmp_path)) == "/vault/n.md"


def test_a_filename_beats_a_colliding_alias(tmp_path):
    """Obsidian resolves a real filename first, and so does the index."""
    (tmp_path / "Topics").mkdir()
    (tmp_path / "Archive").mkdir()
    (tmp_path / "Topics" / "Report.md").write_text("# real\n", encoding="utf-8")
    (tmp_path / "Archive" / "old.md").write_text(
        "---\naliases:\n  - Report\n---\n", encoding="utf-8"
    )
    assert note_href("Report", str(tmp_path)) == "/vault/Topics/Report.md"


def test_aliases_are_not_read_out_of_the_note_body(tmp_path):
    """Only the frontmatter declares an alias; a body mention is not one."""
    (tmp_path / "n.md").write_text(
        "# Note\n\nSome prose mentioning aliases:\n  - Not An Alias\n"
    )
    assert note_href("Not An Alias", str(tmp_path)) is None


def test_a_note_with_no_frontmatter_is_fine(tmp_path):
    (tmp_path / "plain.md").write_text("# Plain\n")
    assert note_href("Plain", str(tmp_path)) == "/vault/plain.md"
