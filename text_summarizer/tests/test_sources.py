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
from types import SimpleNamespace

import pytest

from text_summarizer.sources import (
    MODEL_TOKEN,
    SOURCES_HEADING,
    UNKNOWN,
    VAULT_TOKEN,
    render_sources,
    resolve_vault_name,
    served_model_from_events,
    strip_sources_block,
)

BULLETS = "- Dogs are domesticated mammals.\n- They come in many breeds.\n- Dogs are social animals."


def _event(text="", *, author="text_summarizer", model_version=None, parts=None):
    """A minimal stand-in for an ADK ``Event`` (which subclasses LlmResponse)."""
    return SimpleNamespace(
        author=author,
        content=SimpleNamespace(parts=parts if parts is not None else (
            [SimpleNamespace(text=text, function_response=None)] if text else [SimpleNamespace(text=None, function_response=None)]
        )),
        model_version=model_version,
        partial=False,
    )


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
        "[obsidian][ck][gemini-3.5-flash-lite][[Dogs Overview]]: same topic"
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
        "[obsidian][ck][openrouter/google/gemma-4-26b-a4b-it:free][[Dogs Overview]]: primary\n"
        "[obsidian][ck][openrouter/google/gemma-4-26b-a4b-it:free][[Wolves]]: related canid\n\n"
        'Saved to the second brain as "Dogs Summary".'
    )


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
    assert out.endswith("[[Dogs Overview]]: primary\n")


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
        "[obsidian][ck][m1][[Dogs Overview]]\n"
        "[obsidian][ck][m1][[Wolves]]: related canid\n"
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
    assert out.endswith("[obsidian][ck][unknown][[Dogs Overview]]: primary")
    assert UNKNOWN in out
    # Nothing plausible was invented.
    assert "gemini" not in out and "gemma" not in out and "qwen" not in out


def test_unknown_vault_and_model_together():
    model_text = f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[Dogs Overview]]: primary"
    out = render_sources(model_text, vault_name="", model_name="")
    assert out.endswith("[obsidian][unknown][unknown][[Dogs Overview]]: primary")


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
    assert "[unknown]" in out


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
        "[obsidian][ck][m1][[Dogs Overview]]: primary\n\n"
        'Saved to the second brain as "Dogs Summary".'
    )
    assert strip_sources_block(rendered) == (
        f"{BULLETS}\n\nSaved to the second brain as \"Dogs Summary\"."
    )


def test_strip_sources_block_lets_metrics_ignore_source_lines():
    """Source lines must not be counted as summary bullets."""
    rendered = f"{BULLETS}\n\n{SOURCES_HEADING}\n[obsidian][ck][m1][[Dogs Overview]]: x"
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


def test_pre_existing_note_without_generated_by_model_is_treated_as_unknown():
    """A note written before provenance existed must not gain an invented model."""
    legacy = (
        "---\ntags:\n  - Dogs\ndate: 2026-09-25\n"
        "source_fingerprint: " + "a" * 64 + "\naliases:\n  - Dogs\n---\n\n"
        "- Dogs are social animals."
    )
    assert "generated_by_model" not in legacy
    # Nothing in the renderer invents one for it.
    out = render_sources(
        f"[obsidian][{VAULT_TOKEN}][{MODEL_TOKEN}][[2026-09-25 - dogs]]: pre-existing",
        vault_name="ck",
        model_name="gemini-3.5-flash-lite",
    )
    assert out.endswith(
        "[obsidian][ck][gemini-3.5-flash-lite][[2026-09-25 - dogs]]: pre-existing"
    )


@pytest.mark.parametrize("tool_context", [None, object()])
def test_note_provenance_omits_unresolvable_fields(tool_context):
    from text_summarizer.second_brain import note_provenance

    provenance = note_provenance(tool_context)
    assert "generated_by_model" not in provenance
    # The vault is still resolvable from configuration alone.
    assert provenance.get("generated_in_vault")
