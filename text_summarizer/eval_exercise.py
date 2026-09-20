"""
EVALUATION LOOP EXERCISE
========================

This script shows you exactly how to iterate on prompts and measure improvement.

WORKFLOW:
  1. Run eval with current instructions → note the score
  2. Edit agent.py instructions
  3. Re-run eval → compare scores
  4. Repeat until satisfied

Run this from the text_summarizer directory:
  python eval_exercise.py
"""

import subprocess
import json
import os
import re

AGENT_DIR = os.path.join(os.path.dirname(__file__))
EVAL_FILE = os.path.join(AGENT_DIR, "tests", "eval", "simple_test.test.json")
CONFIG_FILE = os.path.join(AGENT_DIR, "tests", "eval", "test_config.json")
AGENT_FILE = os.path.join(AGENT_DIR, "agent.py")


def run_eval():
    """Run adk eval and return the response_match_score."""
    result = subprocess.run(
        [
            "python", "-m", "google.adk.cli", "eval",
            "text_summarizer",
            EVAL_FILE,
            "--config_file_path", CONFIG_FILE,
            "--print_detailed_results"
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(AGENT_DIR),
        timeout=60
    )
    output = result.stdout + result.stderr

    # Extract response_match_score from output
    match = re.search(r"response_match_score.*?Score:\s*([\d.]+)", output)
    if match:
        return float(match.group(1)), output
    return None, output


def get_current_instructions():
    """Read current agent instructions."""
    with open(AGENT_FILE, "r") as f:
        content = f.read()
    match = re.search(r'instruction="""(.*?)"""', content, re.DOTALL)
    return match.group(1) if match else "NOT FOUND"


def set_instructions(new_instructions):
    """Update agent instructions."""
    with open(AGENT_FILE, "r") as f:
        content = f.read()
    new_content = re.sub(
        r'instruction=""".*?"""',
        f'instruction="""{new_instructions}"""',
        content,
        flags=re.DOTALL
    )
    with open(AGENT_FILE, "w") as f:
        f.write(new_content)
    print(f"\n  Updated instructions to:\n    {new_instructions[:80]}...")


def main():
    print("=" * 60)
    print("EVALUATION LOOP EXERCISE")
    print("=" * 60)

    print("\nCurrent instructions:")
    print("-" * 40)
    print(get_current_instructions())
    print("-" * 40)

    print("\nRunning evaluation with CURRENT instructions...")
    score, output = run_eval()

    if score is not None:
        print(f"\n  >>> CURRENT SCORE: {score:.4f} <<<")
        print(f"  >>> Threshold: 0.5")
        print(f"  >>> Status: {'PASSED' if score >= 0.5 else 'FAILED'}")
    else:
        print("\n  Could not parse score. Raw output:")
        print(output[-500:])
        return

    print("\n" + "=" * 60)
    print("NOW TRY MODIFYING THE INSTRUCTIONS")
    print("=" * 60)
    print("""
Try these experiments:

EXPERIMENT 1: Remove the bullet point rule
  - Remove: "1. Always respond with bullet points (use - or * prefix)."
  - Run: python eval_exercise.py
  - Expected: Score might change since expected output uses bullets

EXPERIMENT 2: Make it more verbose
  - Add: "7. Include as much detail as possible in each bullet point."
  - Run: python eval_exercise.py
  - Expected: Longer responses, might lower ROUGE score if too verbose

EXPERIMENT 3: Make it more concise
  - Change rule 2 to: "2. Each bullet point should be 5-10 words maximum."
  - Run: python eval_exercise.py
  - Expected: Shorter responses, might change score

EXPERIMENT 4: Add format constraint
  - Add: "7. Never use the word 'the' at the start of bullet points."
  - Run: python eval_exercise.py
  - Expected: Forces different phrasing, affects word overlap

To modify instructions, edit: agent.py
Then run: python eval_exercise.py
""")

    # Show how to run eval manually
    print("=" * 60)
    print("MANUAL EVAL COMMAND")
    print("=" * 60)
    print("""
To run eval manually anytime:

  cd C:\\Users\\carlo\\Desktop\\agents
  python -m google.adk.cli eval text_summarizer \\
    text_summarizer\\tests\\eval\\simple_test.test.json \\
    --config_file_path text_summarizer\\tests\\eval\\test_config.json \\
    --print_detailed_results

Look for this line in the output:
  Metric: response_match_score, Status: PASSED, Score: 0.XXXX, Threshold: 0.5

The Score (0.XXXX) is what changes when you modify instructions.
""")


if __name__ == "__main__":
    main()
