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
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from _helpers import SERVED_MODEL, _event, _point_vault_at
from text_summarizer.chart import chart_dir
from text_summarizer.sources import (
<<<<<<< Updated upstream
=======
    CHART_FENCE,
    CHART_IMAGE_PREFIX,
    CHART_TOKEN,
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======


# --- the hourly curve ---------------------------------------------------------
#
# Asked for as "a graph in the answer", and what makes it possible is that it is drawn
# from the tool's payload in code rather than asked of the model. That is the same
# decision as the **Sources** block and the name stamp, and for a sharper reason: a model
# asked to plot 24 hourly readings is not summarising them, it is generating 24 more.
# The free-tier primary that serves most turns here emits token debris on ordinary prose
# (`com最大值 de 19,4 °C`), which is what "draw the chart yourself" looks like in practice.
#
# So the model writes `@@ADK_CHART@@` on a line and this module replaces it. What the
# tests below hold onto is the properties that make that safe: the curve lands where it
# was asked for, a second pass reproduces it byte for byte, a turn with no series loses
# the placeholder rather than keeping a gap, and -- the one a presentation artefact can
# silently break -- none of it moves a quality score.


def _series(temps=None, rain=None, hours=24, day="2026-10-05"):
    """A payload the live API returned for Osasco on 2026-10-04, in this module's names.

    A night near 16 °C rising to 22.9 °C at 13:00, and rain chance climbing through the
    evening -- so the two rows have visibly different shapes, which is what makes the
    per-series scaling testable rather than vacuous.
    """
    return {
        "unit": "hour",
        "hours": hours,
        "time": [f"{day}T{h:02d}:00" for h in range(hours)],
        "temperature_c": temps if temps is not None else [16.1 + (h % 7) for h in range(hours)],
        "rain_chance_pct": rain if rain is not None else [0] * (hours - 6) + [20, 30, 40, 44, 40, 30],
    }


NOW = {"hour": "16:00", "temperature_c": 19.2, "rain_chance_pct": 24}

#: Spelled out rather than imported for the same reason the test file spells out other
#: shapes: an assertion about "the 16-hex name" is a claim about the *format*, and a copy
#: of the pattern is the only way to make a change to ``chart.py`` fail here.
_NAME_RE = re.compile(r"^[0-9a-f]{16}$")


def _alt(image: str) -> str:
    """The alt text of a rendered chart, or ``""`` for anything that is not one."""
    match = re.fullmatch(r"!\[([^\]\n]*)\]\(" + re.escape(CHART_IMAGE_PREFIX) + r"[^)\n]*\)", image)
    return match.group(1) if match else ""


def _heading(alt: str) -> str:
    """The alt text up to the reading of now -- the part naming the place and the day."""
    return alt.split(" - now ", 1)[0]


def _svg_of(image: str) -> str:
    """The stored SVG a rendered chart points at."""
    href = image[image.index("](") + 2 : -1]
    assert href.startswith(CHART_IMAGE_PREFIX), href
    return (Path(chart_dir()) / href.rsplit("/", 1)[-1]).read_text(encoding="utf-8")


def test_the_chart_is_substituted_where_the_model_put_the_token():
    """In place, not appended.

    The model puts the token where the graph belongs -- under the prose that explains it,
    above the source line -- and appending instead would pile the curve onto the end of
    the answer, after the citation, where nothing the model wrote refers to it.
    """
    answer = (
        "- Max 22.9 °C.\n- Rain 44% of the day.\n\n"
        f"{CHART_TOKEN}\n\n"
        f"{SOURCES_HEADING}\n- [obsidian][ck][m1][[Osasco]]: readings"
    )
    rendered = render_sources(
        answer,
        vault_name="ck",
        model_name=SERVED_MODEL,
        chart=render_chart(_series(), NOW, "Osasco"),
    )
    lines = [line for line in rendered.split("\n") if line.strip()]
    assert lines[0] == "- Max 22.9 °C."
    assert lines[2].startswith("![")
    assert CHART_IMAGE_PREFIX in lines[2]
    assert lines[-1].startswith("- [obsidian]")
    assert CHART_TOKEN not in rendered
    assert CHART_IMAGE_PREFIX in rendered.split(SOURCES_HEADING)[0]


def test_a_chart_token_with_no_series_behind_it_is_dropped_and_leaves_no_gap(monkeypatch):
    """A turn where no weather tool ran has no graph, and must not look like it lost one.

    Two halves, and the second is the one that is easy to miss. Removing the placeholder
    on its own leaves the blank line above and below it, so the answer grows a double gap
    exactly where the reader was told to expect a curve.
    """
    answer = f"- Max 22.9 °C.\n\n{CHART_TOKEN}\n\nSaved.\n"
    assert render_sources(answer, chart=None) == "- Max 22.9 °C.\n\nSaved.\n"
    assert render_sources(answer, chart="") == "- Max 22.9 °C.\n\nSaved.\n"
    # And the same for an empty render, which is what a series too short produces.
    assert render_sources(answer, chart=render_chart(None)) == "- Max 22.9 °C.\n\nSaved.\n"


def test_gap_collapsing_does_not_reformat_an_answer_that_never_mentioned_a_chart():
    """The control: the collapse is scoped to the placeholder, not a general tidy-up.

    A renderer that collapsed every run of blank lines would pass the test above while
    quietly reformatting every answer the agent produces -- and that is a change in the
    bytes a user reads, made for a cosmetic reason, in the one function whose job is
    provenance.
    """
    assert render_sources("- a\n\n\n\nb\n") == "- a\n\n\n\nb\n"


def test_rendering_is_idempotent_with_a_chart_present():
    """A second pass reproduces the answer byte for byte, chart included.

    The Sources block earns this by re-parsing its own output; the chart has to earn it
    too, and it does so by carrying no sentinel of its own. A block that still contained
    one would be substituted a second time on the next pass and grow a copy -- which is
    the duplication `after_agent_callback` exists to prevent, arriving through a
    different door.
    """
    chart = render_chart(_series(), NOW, "Osasco")
    kwargs = dict(vault_name="ck", model_name=SERVED_MODEL, chart=chart)
    answer = f"- Max 22.9 °C.\n\n{CHART_TOKEN}\n\n{SOURCES_HEADING}\n- [obsidian][ck][m1][[Osasco]]: x"
    once = render_sources(answer, **kwargs)
    assert render_sources(once, **kwargs) == once
    assert once.count(CHART_IMAGE_PREFIX) == 1


def test_the_chart_does_not_move_any_quality_score():
    """A row of glyphs is not prose the model wrote, and must not be scored as one.

    `quality.bullet_score` counts bullets and `lexical_recall` compares the answer with
    the *user's* words. Both read `summary_only`, so the chart has to be stripped there
    rather than left in -- otherwise weather turns score differently from every other turn
    for no reason a reader of the dashboard could see, which is exactly the kind of drift
    that makes a score useless for the instruction edits it exists to measure.
    """
    from text_summarizer.agent import _score_generation

    user = "como esta o tempo em Osasco hoje?"
    prose = "- Max 22.9 °C.\n- Min 16.1 °C.\n- Chuva 44%."
    chart = render_chart(_series(), NOW, "Osasco")
    without = render_sources(prose, vault_name="ck", model_name=SERVED_MODEL)
    with_chart = render_sources(
        prose + f"\n\n{CHART_TOKEN}", vault_name="ck", model_name=SERVED_MODEL, chart=chart
    )
    assert with_chart != without
    assert _score_generation(user, summary_only(without)) == _score_generation(
        user, summary_only(with_chart)
    )


def test_a_single_reading_is_not_a_chart():
    """One point is a number wearing a graph's clothes."""
    assert render_chart({"time": ["2026-10-05T16:00"], "temperature_c": [19.2]}, None) == ""
    assert render_chart({"time": [], "temperature_c": []}, None) == ""


def test_the_strip_marker_is_the_image_url_not_a_word_the_model_writes():
    """Why the marker is invisible, now that the chart is an image.

    The obvious key for stripping a rendered chart is a line of prose -- "Hourly -", a
    heading -- and that is exactly what cannot be used: a model writes prose, so a
    stripper keyed on a phrase eats the rest of the answer the first time the phrase
    appears without a chart. The second half matters more now that the chart *is* an
    image: a model asked about a chart may well write the word, and the alt text of the
    rendered image carries the title too -- so a stripper keyed on the phrase would
    delete the image the moment the answer was scored, and the alt text is what a text
    consumer reads.
    """
    prose = "- Max 22.9 °C.\n\nHourly - Osasco, 2026-10-05\n°C 16.1..22.9\n- Min 16.1 °C."
    assert summary_only(prose) == prose, "a phrase the model wrote was treated as a chart"

    chart = render_chart(_series(), NOW, "Osasco")
    rendered = f"- Max 22.9 °C.\n\n{chart}\n- Min 16.1 °C."
    stripped = summary_only(rendered)
    assert stripped == "- Max 22.9 °C.\n\n- Min 16.1 °C."
    assert "Hourly -" not in stripped


def test_an_image_the_model_wrote_is_not_stripped_as_if_it_were_a_chart():
    """The counterpart, and the one this change made newly possible.

    The renderer now emits an ``<img>``, so the strip key is a URL shape -- and a model can
    emit images too. A stripper that matched "a markdown image" would delete every
    diagram a model put in an answer, and ``summary_only`` is what
    ``quality.lexical_recall`` reads, so the loss would be invisible in the scores.

    Two shapes, because a URL prefix alone is not narrow enough: a different host and a
    different path both fail the match, and the digest is required to be exactly the 16
    hex characters ``chart.chart_href`` produces.
    """
    for image in (
        "![a chart](/chart/chartjs.png)",
        "![a chart](https://example.test/deadbeefdeadbeef.svg)",
        "![a chart](/vault/deadbeefdeadbeef.svg)",
        "![a chart](/chart/DEADBEEFDEADBEEF.svg)",
    ):
        answer = f"- Here it is.\n\n{image}\n- That is the forecast."
        assert summary_only(answer) == answer, image


def test_a_code_block_the_model_wrote_is_not_stripped_as_if_it_were_a_chart():
    """The other half of the marker argument: a *narrow* marker still has to be narrow.

    Keying the strip on "a fenced block" rather than on ``adk-chart`` passes the test
    above for the wrong reason -- it removes the prose case by removing every code block,
    which is a far larger loss. And the loss is silent and measurable: ``summary_only`` is
    what ``quality.lexical_recall`` and ``format_and_recall`` read, so an answer that
    quotes code would have its quoted code deleted before scoring. The scores would
    simply read a little different on code answers, with nothing to say why.

    Mutating ``_is_chart_fence`` to ``line.strip().startswith("```")`` leaves the suite
    green, which is why this test exists and not a comment.
    """
    answer = (
        "- Use `dict` or `defaultdict`.\n\n"
        "```python\n"
        "counts = defaultdict(int)\n"
        "for word in text.split():\n"
        "    counts[word] += 1\n"
        "```\n\n"
        "- Both are in the stdlib."
    )
    assert summary_only(answer) == answer


def test_a_fenced_block_with_its_own_info_string_is_still_not_a_chart():
    """``adk-chart`` is the whole marker; a fence labelled anything else is the model's.

    The mutation a reader of the previous test might reasonably make is a substring
    match -- ``"chart" in line`` -- which would eat every block whose language happens
    to be ``mermaid`` or ``chartjs``. So the comparison is exact, and this is the test
    that says so rather than leaving it to the implementation.
    """
    for info in ("mermaid", "chartjs", "adk-charts", "adk-chart-2", "not-adk-chart"):
        answer = f"- A diagram.\n\n```{info}\ngraph TD;\nA-->B;\n```\n\n- That is it."
        assert summary_only(answer) == answer, info


def test_an_unterminated_chart_fence_does_not_leak_the_rest_of_the_answer_into_the_strip():
    """A half-rendered block is a truncated model response, and the strip must still end.

    Treating an unterminated fence as running to the end of the text is the choice that
    keeps the strip from leaving a dangling `````adk-chart`` fence in the middle of an
    otherwise well-formed answer.
    """
    assert summary_only("- a\n\n```adk-chart\nHourly - x\n°C ▁▂▃") == "- a\n"


def test_the_chart_is_one_line_of_an_image_that_fits_the_message_column():
    """A fixed 720px-wide SVG on a line of its own, and never a wrapping block.

    The ASCII version's bound was 80 *characters*, because the curve was text and text
    wraps: a chart that wraps stops being a chart, since the axis stops lining up with the
    rows under it. An ``<img>`` cannot wrap -- it is one replaced element with its own
    intrinsic width -- so the same failure is gone, and what replaces the character bound
    is that the width has to fit the dev UI's ~800px column, or the browser scales it down
    and the reader squints.

    The renderer emits ``width``/``viewBox`` from one constant rather than measuring
    anything, because an ``<img>`` has no viewport to measure. So the claim is that the
    constant is inside the column.
    """
    chart = render_chart(_series(), NOW, "Osasco")
    assert "\n" not in chart, "the image must occupy a single line, or the strip misses it"
    href = chart[chart.index("](") + 2 : -1]
    assert href.startswith(CHART_IMAGE_PREFIX), href
    name = href.rsplit("/", 1)[-1].removesuffix(".svg")
    assert _NAME_RE.fullmatch(name), href
    root = Path(chart_dir()) / f"{name}.svg"
    assert root.is_file(), "the image is referenced but was never stored"
    header = root.read_text(encoding="utf-8").split(">")[0]
    assert f'width="{720}"' in header
    assert 720 <= 800, "wider than the dev UI's message column, so it scales down unreadably"


def test_the_same_forecast_renders_one_file_rather_than_one_per_turn():
    """The content-addressed name, which is also what makes rendering idempotent.

    Two turns describing the same hours must not accumulate identical images: the name is
    a digest of the payload, so the second turn rewrites the same bytes to the same path
    and the store stays proportional to *distinct* forecasts. A random token would fail
    this -- and would also break ``render_sources``'s idempotence, which the linked-
    ``Sources`` re-parse depends on.

    Counted against the store as it stands rather than against an empty one: the module
    writes nothing outside ``CHART_DIR`` but it is not emptied between tests, so an
    absolute count would be a claim about the order the suite ran in.
    """
    store = Path(chart_dir())
    before = {path.name for path in store.glob("*.svg")} if store.is_dir() else set()

    first = render_chart(_series(), NOW, "Osasco")
    after_first = {path.name for path in store.glob("*.svg")}
    second = render_chart(_series(), NOW, "Osasco")

    assert first == second, "the same payload rendered two different URLs"
    assert {path.name for path in store.glob("*.svg")} == after_first, (
        "a repeat of the same forecast added a file"
    )
    assert len(after_first - before) == 1

    # And a *different* forecast is a different file, which is the other half: a digest
    # that ignored its inputs would pass the assertion above while serving one graph for
    # every forecast ever drawn.
    other = render_chart(_series(temps=[20.0] * 24), NOW, "Osasco")
    assert other != first
    assert len({path.name for path in store.glob("*.svg")} - before) == 2


def test_a_stray_chart_token_never_reaches_the_reader():
    """The sentinel machinery covers all four tokens, not the three it started with.

    ``_TOKENS`` is built from a tuple for exactly this: adding a token without adding it
    there would leave the scrubber ignorant of it, and a turn with no series would show
    the reader `@@ADK_CHART@@` where a graph should have been.
    """
    assert CHART_TOKEN not in render_sources(f"- a\n\n{CHART_TOKEN}\n", chart=None)
    assert CHART_TOKEN not in render_sources(f"{CHART_TOKEN} - a")


def test_the_hour_reading_is_in_the_alt_text_and_the_drawing_marks_it():
    """Two consumers, two channels, and neither is a column index.

    The ASCII version refused to draw a caret under the current hour because the caret has
    to sit under exactly the right *character column*, and that depends on whichever
    monospace face the reader's browser falls back to -- a property this module cannot
    know. In an SVG the hour is a real x-coordinate, so the drawing can mark it (a dashed
    rule on the first panel) and the misalignment is gone.

    What it cannot do is *reach* the reader who never loads an image -- the ``adk run``
    CLI, a text browser, a screen reader. That reader gets the alt text, which is the whole
    point of carrying the reading twice: one channel is lost silently when the image does
    not load, and neither one alone is enough.
    """
    chart = render_chart(_series(), NOW, "Osasco")
    alt = _alt(chart)
    assert "now 16:00" in alt and "19.2" in alt and "24%" in alt

    svg = _svg_of(chart)
    assert "stroke-dasharray" in svg, "the drawing does not mark where now is"
    assert ">now 16:00</text>" in svg
    # The rule is drawn once, on the first panel: two would read as two annotations.
    assert svg.count("stroke-dasharray") == 1

    # With no `current` there is nothing to mark, and nothing claiming to be the present.
    assert "now" not in _alt(render_chart(_series(), None, "Osasco"))
    assert "stroke-dasharray" not in _svg_of(render_chart(_series(), "not a dict", "Osasco"))


def test_the_current_hour_is_floored_and_a_missing_one_is_never_invented():
    """An absent reading is a shorter line, never a printed `None`.

    The floor itself is `weather`'s property and is tested there; what is asserted here is
    the rendering, in both channels -- a `current` carrying only an hour must still mark
    that hour in the drawing and say ``now 16:00`` in the alt text, and must not invent a
    temperature or a rain chance to go with it.
    """
    chart = render_chart(_series(), {"hour": "16:00"}, "Osasco")
    assert _alt(chart) == "Hourly - Osasco, 2026-10-05, local time - now 16:00"
    assert ">now 16:00</text>" in _svg_of(chart)

    for junk in ("not a dict", {"hour": 17}, {"hour": "99:99"}, {"hour": ""}, {}, None):
        rendered = render_chart(_series(), junk, "Osasco")
        assert "now" not in _alt(rendered), junk
        assert "stroke-dasharray" not in _svg_of(rendered), junk


def test_the_chart_is_titled_with_the_place_and_the_day_it_covers():
    """A curve with no place on it is a chart of nowhere.

    The date is stated rather than inferred from the axis, because the axis carries hours
    only and a reader comparing two answers has to see which day each one belongs to. It is
    in both channels -- the alt text and the drawn title -- because the alt text is what
    survives when the image does not.
    """
    chart = render_chart(_series(day="2026-10-07"), NOW, "Londres")
    alt = _alt(chart)
    assert "Londres" in alt and "2026-10-07" in alt and "local time" in alt
    assert ">Hourly - Londres, 2026-10-07</text>" in _svg_of(chart)

    # No place recorded: the title drops it rather than inventing one.
    assert "Osasco" not in _alt(render_chart(_series(), NOW, None))


def test_a_non_string_place_is_dropped_from_the_title_rather_than_printed():
    """A heading is either a place's name or nothing.

    ``render_chart`` is public and its ``place`` argument is whatever the caller had to
    hand, so f-stringing it unguarded would print ``Hourly - 17`` or ``Hourly - ['Osasco']``
    -- and neither reads as a bug. A reader has no way to tell it is one, which is what
    makes it worth a guard rather than a code review.

    Coercion rather than a type check, because ``None`` and ``""`` already mean "no
    place" here and a caller that passes a str is fine whether or not it is a real city
    name -- ``render_chart`` cannot tell a bad name from a good one, and pretending it
    can would only move the failure somewhere it can no longer be seen.

    The same coercion feeds the SVG title and the fingerprint, so a non-string cannot leak
    into the drawn heading *or* split the store: one name for one shape.
    """
    for absent in (None, 17, 0.5, ["Osasco"], {"name": "Osasco"}, (1, 2)):
        rendered = render_chart(_series(), NOW, absent)
        assert _heading(_alt(rendered)) == "Hourly - 2026-10-05, local time", absent
        assert _alt(rendered) == _alt(render_chart(_series(), NOW, None)), absent

    # The empty string and a blank one are the same "no place", not a stray separator.
    for blank in ("", "   ", "\n"):
        assert _heading(_alt(render_chart(_series(), NOW, blank))) == (
            "Hourly - 2026-10-05, local time"
        ), repr(blank)

    # And a non-string is not merely hidden from the title: it does not become a
    # different image, because the fingerprint coerces it the same way.
    assert render_chart(_series(), NOW, 17) == render_chart(_series(), NOW, None)


def test_strip_chart_block_is_a_no_op_on_text_that_has_no_chart():
    """So the hot path for every non-weather turn costs one substring test."""
    text = "- a\n- b\n\n**Sources**\n- [obsidian][ck][m1][[x]]: y"
    assert strip_chart_block(text) is text or strip_chart_block(text) == text
>>>>>>> Stashed changes
