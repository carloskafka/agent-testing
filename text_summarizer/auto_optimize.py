"""
Auto-Optimization Loop for ADK Agent Instructions
==================================================

Automatically improves agent instructions by running eval -> critiquing -> rewriting -> re-eval.

Usage:
    python auto_optimize.py
    python auto_optimize.py --max-iterations 5
    python auto_optimize.py --optimizer-model gemini-3.5-flash-lite

Model provider (same DI as agent.py):
    MODEL_PROVIDER=gemini   python auto_optimize.py
    MODEL_PROVIDER=openrouter python auto_optimize.py
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

AGENT_DIR = Path(__file__).parent
AGENT_FILE = AGENT_DIR / "agent.py"
EVAL_FILE = AGENT_DIR / "tests" / "eval" / "simple_test.test.json"
CONFIG_FILE = AGENT_DIR / "tests" / "eval" / "test_config.json"
HISTORY_FILE = AGENT_DIR / "optimization_history.json"

#: Instruction rules the eval grades, and which a rewrite must not delete.
#:
#: 7, 8 and 9 are the retrieve-first / save / log trio named in every golden
#: answer's ``intermediate_data.tool_uses``
#: (``[save_summary_to_second_brain, log_conversation]``), which
#: ``tests/eval/test_config.json`` scores as ``tool_trajectory_avg_score`` at
#: threshold 1.0. 11 is the Gmail rule; it is not in ``tool_uses``, but a trim
#: that keeps 7-9 by accident and drops the rest is the same failure one question
#: away, so it is pinned here too.
#:
#: The reason this guard exists: the loop's only decision is ROUGE-1 word overlap
#: (``response_match_score``), which *rises* when instructions get shorter and more
#: generic. Rules 7-11 are long and are exactly what a trim removes. So a rewrite
#: can delete them, score better, and be KEPT -- while the agent quietly stops
#: writing to the second brain and stops citing the vault, and
#: ``tool_trajectory_avg_score`` goes to 0.00. ROUGE-1 cannot see that loss, so
#: nothing in the loop's own arithmetic would refuse it.
REQUIRED_RULES = (7, 8, 9, 11, 12, 13)


def run_eval():
    """Run adk eval and return the response_match_score."""
    # CACHE_ENABLED=false, forced rather than inherited. `adk eval` is not
    # hermetic: it calls the real agent, which calls save_summary_to_second_brain
    # against the real vault. So the baseline run writes a note per eval prompt,
    # and the *next* run on the same eval set hits the cache -- skipping the
    # model and the tool calls -- while still reporting a high
    # response_match_score against the stored summary. Every iteration after the
    # first would then be graded against its own predecessor instead of the
    # instructions under test, and the loop would report gains that are really
    # just the cache. See gotcha 10 in AGENTS.md.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "google.adk.cli",
            "eval",
            "text_summarizer",
            str(EVAL_FILE),
            "--config_file_path",
            str(CONFIG_FILE),
            "--print_detailed_results",
        ],
        capture_output=True,
        text=True,
        cwd=str(AGENT_DIR.parent),
        env={**os.environ, "CACHE_ENABLED": "false"},
        timeout=120,
    )
    output = result.stdout + result.stderr
    match = re.search(r"response_match_score.*?Score:\s*([\d.]+)", output)
    if match:
        return float(match.group(1)), output
    return None, output


def get_current_instructions():
    """Read current agent instructions from agent.py."""
    content = AGENT_FILE.read_text()
    match = re.search(r'instruction="""(.*?)"""', content, re.DOTALL)
    return match.group(1).strip() if match else None


def set_instructions(new_instructions):
    """Update agent instructions in agent.py."""
    content = AGENT_FILE.read_text()
    new_content = re.sub(
        r'instruction="""(.*?)"""',
        f'instruction="""{new_instructions}"""',
        content,
        flags=re.DOTALL,
    )
    AGENT_FILE.write_text(new_content)


def missing_required_rules(instructions):
    """The REQUIRED_RULES numbers that ``instructions`` does not define.

    Returns the numbers, ascending, so a caller can name them. A rule counts as
    present when the instructions open a line with that number and a period
    (``^7. ``), which is how every rule in the block is written. Renumbering the
    block therefore reads as "all the rules are missing", which is the safe
    direction to fail: a guard that cannot find rule 8 must not let the rewrite
    through.
    """
    if not instructions:
        return list(REQUIRED_RULES)
    return [
        number
        for number in REQUIRED_RULES
        if not re.search(rf"^{number}\. ", instructions, re.MULTILINE)
    ]


def get_optimizer_model_name(provider):
    """Resolve the optimizer model name based on the provider."""
    if provider == "openrouter":
        alias = os.environ.get("MODEL_ALIAS", "")
        openrouter_models = {
            "gemma": "google/gemma-4-26b-a4b-it:free",
            "qwen": "qwen/qwen3.8-27b:free",
            "nvidia": "nvidia/nemotron-3.5-lightning:free",
        }
        return openrouter_models.get(alias, "google/gemma-4-26b-a4b-it:free")
    return "gemini-3.5-flash-lite"


def critique_and_rewrite(
    current_instructions, current_score, test_input, expected_output, actual_output, provider
):
    """Use an LLM to critique current instructions and suggest improvements."""
    prompt = f"""You are a prompt engineer optimizing an AI agent's instructions.

CURRENT INSTRUCTIONS:
{current_instructions}

CURRENT EVALUATION SCORE: {current_score} (ROUGE-1 word overlap, max 1.0)

TEST CASE INPUT:
{test_input}

EXPECTED OUTPUT:
{expected_output}

ACTUAL OUTPUT FROM AGENT:
{actual_output}

TASK:
Analyze why the score is {current_score} and rewrite the instructions to improve it.

FOCUS ON:
1. Using words that appear in the expected output (to increase ROUGE-1 overlap)
2. Making the format match the expected output more closely
3. Being specific about what the agent should produce

RULES:
- Keep the instructions under 500 words (the current block is ~490 across 11 rules).
  Trimming to a much shorter budget is the failure this loop cannot otherwise see:
  the rules below are long, and deleting them RAISES ROUGE-1 while the agent stops
  writing to the second brain.
- Rule 7 (retrieve-first, the **Sources** line format), rule 8 (call
  save_summary_to_second_brain), rule 9 (call log_conversation) and rule 11 (the
  Gmail tools) are load-bearing. tests/eval/test_config.json grades 7-9 as
  tool_trajectory_avg_score at threshold 1.0, so dropping any of them scores 0.00
  there while your score still looks like an improvement. Keep all four, verbatim
  in substance; reword around them instead.
- Do not renumber the remaining rules.
- Only change what's necessary to improve the score
- The agent must still produce bullet-point summaries
- Do not add tools or change the model

Respond with ONLY the new instructions text, nothing else. Do not include "Instructions:" or any wrapper."""

    if provider == "openrouter":
        import litellm

        litellm.api_key = os.environ.get("OPENROUTER_API_KEY")
        litellm.api_base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        model_name = get_optimizer_model_name(provider)
        response = litellm.completion(
            model=f"openrouter/{model_name}",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    else:
        import google.genai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        model_name = get_optimizer_model_name(provider)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text.strip()


def save_history(iteration, old_instructions, new_instructions, old_score, new_score, kept):
    """Save optimization history."""
    history = []
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())

    history.append(
        {
            "iteration": iteration,
            "old_score": old_score,
            "new_score": new_score,
            "kept": kept,
            "old_instructions": old_instructions,
            "new_instructions": new_instructions,
        }
    )

    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Auto-optimize agent instructions")
    parser.add_argument(
        "--max-iterations", type=int, default=10, help="Max optimization iterations"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Stop after N consecutive non-improvements"
    )
    parser.add_argument(
        "--optimizer-model",
        type=str,
        default=None,
        help="Override optimizer model (default: based on MODEL_PROVIDER)",
    )
    args = parser.parse_args()

    provider = os.environ.get("MODEL_PROVIDER", "gemini")
    optimizer_model = args.optimizer_model or get_optimizer_model_name(provider)

    print("=" * 60)
    print("AUTO-OPTIMIZATION LOOP")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Patience: {args.patience} consecutive non-improvements")
    print(f"Optimizer model: {optimizer_model}")

    # Baseline
    print("\n[Baseline] Running eval with current instructions...")
    baseline_score, _ = run_eval()
    if baseline_score is None:
        print("ERROR: Could not get baseline score")
        return
    print(f"  Baseline score: {baseline_score:.4f}")

    best_score = baseline_score
    best_instructions = get_current_instructions()
    no_improvement_count = 0

    for i in range(args.max_iterations):
        print(f"\n--- Iteration {i + 1}/{args.max_iterations} ---")

        current_instructions = get_current_instructions()
        current_score = best_score

        # Get the expected output from the test file for context
        with open(EVAL_FILE) as f:
            test_data = json.load(f)
        test_case = test_data["eval_cases"][0]
        test_input = test_case["conversation"][0]["user_content"]["parts"][0]["text"]
        expected_output = test_case["conversation"][0]["final_response"]["parts"][0]["text"]

        # Critique and rewrite
        print("  Optimizer is analyzing and rewriting instructions...")
        try:
            new_instructions = critique_and_rewrite(
                current_instructions, current_score, test_input, expected_output, "", provider
            )
        except Exception as e:
            print(f"  ERROR calling optimizer: {e}")
            break

        # Apply new instructions
        set_instructions(new_instructions)
        print(f"  New instructions:\n    {new_instructions[:100]}...")

        # Refuse a rewrite that deleted the rules the eval grades, BEFORE
        # spending a second eval on it. See REQUIRED_RULES: ROUGE-1 rewards the
        # trim that breaks persistence, so the loop would otherwise keep it.
        dropped = missing_required_rules(new_instructions)
        if dropped:
            print(
                "  REJECTED: the rewrite dropped instruction rule(s) "
                f"{', '.join(str(n) for n in dropped)}, which "
                "tests/eval/test_config.json grades as tool_trajectory_avg_score "
                "(threshold 1.0). Reverting and aborting this iteration."
            )
            print(
                "  The score would probably have gone UP -- that is the trap. "
                "Rewrite around those rules instead of deleting them."
            )
            set_instructions(current_instructions)
            save_history(
                i + 1, current_instructions, new_instructions, current_score, None, kept=False
            )
            break

        # Re-evaluate
        print("  Running eval with new instructions...")
        new_score, _ = run_eval()
        if new_score is None:
            print("  ERROR: Could not get new score, reverting")
            set_instructions(current_instructions)
            break

        print(
            f"  Score: {current_score:.4f} -> {new_score:.4f} (delta: {new_score - current_score:+.4f})"
        )

        # Decide whether to keep
        if new_score > best_score:
            best_score = new_score
            best_instructions = new_instructions
            no_improvement_count = 0
            kept = True
            print("  [+] KEPT (improved)")
        else:
            no_improvement_count += 1
            kept = False
            print(f"  [-] REVERTED (no improvement, {no_improvement_count}/{args.patience})")
            set_instructions(current_instructions)

        save_history(i + 1, current_instructions, new_instructions, current_score, new_score, kept)

        # Early stop
        if no_improvement_count >= args.patience:
            print(f"\n  Stopping: no improvement for {args.patience} consecutive iterations")
            break

    # Final summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"  Starting score: {baseline_score:.4f}")
    print(f"  Best score:     {best_score:.4f}")
    print(f"  Improvement:    {best_score - baseline_score:+.4f}")
    print(f"\n  Best instructions:")
    print(f"  {best_instructions}")
    print(f"\n  History saved to: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
