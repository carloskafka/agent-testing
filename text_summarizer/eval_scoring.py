"""Eval-set aware scoring: reproduce ADK's response_match_score (ROUGE-1).

The ADK eval loop computes ``response_match_score`` via
``google.adk.evaluation.final_response_match_v1._calculate_rouge_1_scores``,
which is the ROUGE-1 f-measure between the agent's final response and the
golden answer stored in an eval set. That module needs ``google-adk[eval]``
(which provides ``rouge_score``).

This module browser-loads ``tests/eval/*.test.json`` (single eval files)
and ``tests/eval/*.evalset.json`` (eval sets) into a lookup keyed by the
normalized user prompt. When a live call's user text matches a known eval
case, we compute the *exact same* ROUGE-1 score and expose it so the agent
callback can push it to Langfuse under the same metric name the eval loop
reports.

Everything is defensive: if ``rouge_score`` (or the ADK helper) is not
available, or an eval set cannot be read, matching falls back to ``None``
and the callback simply skips the score.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

try:
    from google.adk.evaluation.final_response_match_v1 import (
        _calculate_rouge_1_scores,
    )
except Exception:  # pragma: no cover - eval extra not installed
    _calculate_rouge_1_scores = None

_EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "eval")


def _normalize(text: str) -> str:
    """Collapse whitespace for prompt matching (keeps word identity)."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _user_prompt(invocation: dict) -> str:
    content = invocation.get("user_content") or {}
    parts = content.get("parts") or []
    return "".join(
        p.get("text", "") or "" for p in parts if isinstance(p, dict) and p.get("text")
    )


def _golden_response(invocation: dict) -> str:
    final = invocation.get("final_response") or {}
    parts = final.get("parts") or []
    return "\n".join(
        p.get("text", "") or "" for p in parts if isinstance(p, dict) and p.get("text")
    )


def _load_eval_file(path: str) -> list[dict]:
    """Extract (user_prompt, golden_final_response) pairs from a test/evalset file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cases = data.get("eval_cases", [])
    found = []
    for case in cases:
        for invocation in case.get("conversation", []):
            prompt = _user_prompt(invocation)
            golden = _golden_response(invocation)
            if prompt and golden:
                found.append((_normalize(prompt), golden))
    return found


@lru_cache(maxsize=1)
def load_eval_cases() -> dict[str, str]:
    """Return {normalized_user_prompt: golden_final_response} across all eval files."""
    lookup: dict[str, str] = {}
    if not os.path.isdir(_EVAL_DIR):
        return lookup
    for name in sorted(os.listdir(_EVAL_DIR)):
        if not name.endswith((".test.json", ".evalset.json")):
            continue
        for prompt, golden in _load_eval_file(os.path.join(_EVAL_DIR, name)):
            lookup[prompt] = golden
    return lookup


def response_match_score(user_text: str) -> float | None:
    """Return the golden answer for a live prompt, or None if unmatched.

    This mirrors ADK's lookup: the eval loop scores each invocation against the
    golden final response stored in its eval case. Here we return the golden
    response to compare against; the f-measure is computed downstream.
    """
    if not (prompt := _normalize(user_text or "")):
        return None
    return load_eval_cases().get(prompt)


def rouge1_fmeasure(model_text: str, golden: str) -> float | None:
    """ROUGE-1 f-measure, byte-for-byte identical to the ADK eval loop.

    Returns None when rouge_score / the ADK helper are unavailable.
    """
    if _calculate_rouge_1_scores is None or not model_text or not golden:
        return None
    try:
        return float(_calculate_rouge_1_scores(model_text, golden).fmeasure)
    except Exception:  # pragma: no cover - never break agent on scoring
        return None


def response_match_for_agent(user_text: str, model_text: str) -> float | None:
    """Full pipeline: match prompt to eval case, then compute ROUGE-1."""
    golden = response_match_score(user_text)
    if golden is None:
        return None
    return rouge1_fmeasure(model_text, golden)