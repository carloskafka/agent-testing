# Evaluation Loop

The evaluation loop is the core learning objective of this project: **edit instructions in `agent.py` → run `adk eval` → compare scores → keep or revert**.

```
  +-----------------------------------------------------+
  |                                                     |
  |   1. DEFINE   -> Create test cases with expected     |
  |                  inputs and outputs                  |
  |                                                     |
  |   2. MEASURE  -> Run `adk eval` to score the agent   |
  |                  against test cases                  |
  |                                                     |
  |   3. ANALYZE  -> Compare actual vs expected output    |
  |                  to find weaknesses                  |
  |                                                     |
  |   4. IMPROVE  -> Update agent instructions in        |
  |                  agent.py to address failures        |
  |                                                     |
  |   5. REPEAT   -> Re-run evals to verify improvement  |
  |                                                     |
  +-----------------------------------------------------+
```

## Evaluation Criteria Used

| Criterion | What It Measures | Threshold |
|---|---|---|
| `tool_trajectory_avg_score` | Did the agent use the right tools in the right order? | 1.0 (exact) |
| `response_match_score` | How similar is the output to the expected response? (ROUGE-1) | 0.5 |

## Running Evaluations

Evaluations call the **real agent** and write into the vault, so they must bypass
the vault cache — always run them with `CACHE_ENABLED=false`:

```bash
# Single eval case
CACHE_ENABLED=false uv run adk eval text_summarizer \
  text_summarizer/tests/eval/simple_test.test.json \
  --config_file_path text_summarizer/tests/eval/test_config.json \
  --print_detailed_results

# Full eval set
CACHE_ENABLED=false uv run adk eval text_summarizer \
  text_summarizer/tests/eval/summarizer_eval_set.evalset.json \
  --config_file_path text_summarizer/tests/eval/test_config.json \
  --print_detailed_results
```

Without `CACHE_ENABLED=false`, a second run hits the vault cache, skips the model
*and* the tool calls, and reports a high score while testing nothing.

## Interpreting Results

```
Metric: response_match_score, Status: PASSED, Score: 0.679, Threshold: 0.5
```

- **Score: 0.679** — the agent's response had 67.9% word overlap with the expected response
- **Threshold: 0.5** — minimum acceptable score is 50%
- **Status: PASSED** — score meets or exceeds threshold

If a test fails, the output shows a side-by-side comparison of expected vs actual
response, so you can see exactly what went wrong.

> **Important:** `response_match_score` is ROUGE-1 **word overlap**, not quality.
> Higher scores come from *matching the expected words*, not from being objectively
> better. Don't use it to judge semantic quality.

## Hands-On: Learning the Evaluation Loop

Learn by doing: **edit instructions → run eval → compare scores → learn**.

### Quick Start

```bash
cd text_summarizer
python eval_exercise.py
```

This script shows your current instructions, runs the eval, and displays the score.

### The Baseline

With the original instructions, the baseline score is:

```
>>> CURRENT SCORE: 0.6600 <<<
>>> Threshold: 0.5
>>> Status: PASSED
```

### Experiment 1: Make Instructions More Restrictive

Add rules that force different wording than the expected output:

```python
# In agent.py, change instruction to:
instruction="""You are a text summarization agent...

Rules:
1. Always respond with bullet points using the - prefix (dash followed by space).
2. Each bullet point should be a single concise sentence.
3. Capture the main ideas, not minor details.
4. Aim for exactly 3-5 bullet points.
5. Do not add information that is not present in the original text.
6. Use clear, professional language.
7. Start each bullet with a subject or action word, not "The" or "It".
8. Keep each bullet point under 20 words.""",
```

Run eval, then check the result:

```
>>> CURRENT SCORE: 0.5116 <<<
```

**What happened:** the score dropped from 0.6600 to 0.5116. Rules 7 and 8 forced
different words than the expected output, reducing ROUGE-1 word overlap.

**Lesson:** restrictive rules that force different phrasing can hurt a score that
measures word overlap.

### Experiment 2: Make Instructions Simpler

Change the bullet count rule from `3-7` to `3-5`:

```bash
python eval_exercise.py
```

```
>>> CURRENT SCORE: 0.7312 <<<
```

**What happened:** the score increased from 0.6600 to 0.7312. Fewer bullet points
happened to match the expected output more closely.

**Lesson:** small changes can have surprising effects. Test everything.

### Experiment Results Summary

| Run | Instruction Change | Score | Delta |
|---|---|---|---|
| 1 | Original baseline | 0.6600 | — |
| 2 | Added "no The/It" + "under 20 words" | 0.5116 | -0.1484 |
| 3 | Reverted + changed "3-7" to "3-5" | 0.7312 | +0.0712 |

### What to Try Next

| Experiment | What to Change | Expected Effect |
|---|---|---|
| Remove bullets | Delete rule 1 entirely | Score may drop (expected output uses bullets) |
| More verbose | Add "Include as much detail as possible" | Score may drop (too wordy) |
| More concise | Change rule 2 to "5-10 words maximum" | Score may drop (too short) |
| Simple vocabulary | Add "Use only common words" | Score may rise (more word overlap) |
| Different format | Change "- " to "* " prefix | Score may drop (expected uses "- ") |

### The Full Loop

```
  +-----------------------------------------------------+
  |                                                     |
  |   1. RUN     -> python eval_exercise.py              |
  |                Note the score                        |
  |                                                     |
  |   2. EDIT    -> Change instructions in agent.py      |
  |                (one change at a time)                |
  |                                                     |
  |   3. RE-RUN  -> python eval_exercise.py              |
  |                Compare new score to previous         |
  |                                                     |
  |   4. LEARN   -> What caused the score change?        |
  |                Keep changes that improve score       |
  |                Revert changes that hurt score        |
  |                                                     |
  |   5. REPEAT  -> Try next experiment                   |
  |                                                     |
  +-----------------------------------------------------+
```

## Auto-Optimization

Instead of manually editing instructions, use `auto_optimize.py` to let an LLM
automatically improve the agent's instructions through the eval loop.

### How It Works

```
  +-----------------------------------------------------+
  |                                                     |
  |   1. RUN EVAL  -> Get baseline score                |
  |                                                     |
  |   2. CRITIQUE   -> LLM analyzes why score is low    |
  |                    and suggests instruction changes  |
  |                                                     |
  |   3. REWRITE    -> LLM rewrites the instructions    |
  |                                                     |
  |   4. RE-EVAL    -> Score the new instructions       |
  |                                                     |
  |   5. KEEP/REVERT -> Keep if improved, revert if not |
  |                                                     |
  |   6. REPEAT     -> Until patience exhausted or max  |
  |                    iterations reached                |
  |                                                     |
  +-----------------------------------------------------+
```

### Usage

```bash
# Uses the same MODEL_PROVIDER as agent.py
model_provider=gemini python text_summarizer/auto_optimize.py

# With options
model_provider=gemini python text_summarizer/auto_optimize.py --max-iterations 5 --patience 3

# Force a specific optimizer model
model_provider=gemini python text_summarizer/auto_optimize.py --optimizer-model gemini-3.5-flash-lite
```

### Example Run

```
============================================================
AUTO-OPTIMIZATION LOOP
============================================================
Provider: gemini
Max iterations: 2
Patience: 3 consecutive non-improvements
Optimizer model: gemini-3.5-flash-lite

[Baseline] Running eval with current instructions...
  Baseline score: 0.7423

--- Iteration 1/2 ---
  Optimizer is analyzing and rewriting instructions...
  Score: 0.7423 -> 0.6517 (delta: -0.0906)
  [-] REVERTED (no improvement, 1/3)

--- Iteration 2/2 ---
  Optimizer is analyzing and rewriting instructions...
  Score: 0.7423 -> 0.7917 (delta: +0.0494)
  [+] KEPT (improved)

============================================================
OPTIMIZATION COMPLETE
============================================================
  Starting score: 0.7423
  Best score:     0.7917
  Improvement:    +0.0494
```

The optimizer learned to add: *"closely mirroring the key terms, phrasing, and
sentence structures found in the source text"* — which increased ROUGE-1 word overlap.

### Optimization History

Each run saves history to `optimization_history.json` (git-ignored, regenerated per run):

```json
[
  {
    "iteration": 1,
    "old_score": 0.7423,
    "new_score": 0.6517,
    "kept": false,
    "old_instructions": "...",
    "new_instructions": "..."
  },
  {
    "iteration": 2,
    "old_score": 0.7423,
    "new_score": 0.7917,
    "kept": true,
    "old_instructions": "...",
    "new_instructions": "..."
  }
]
```

> `auto_optimize.py` locates the main agent's instructions via the regex
> `instruction="""..."""` and rewrites them in place, reverting to the
> best-known instructions when a rewrite doesn't improve the score. Keep that
> exact `instruction="""..."""` block intact in `agent.py`.

## Creating Your Own Test Cases

To add a new test case, append to an existing `.evalset.json` file:

```json
{
  "eval_id": "my_new_test",
  "conversation": [
    {
      "invocation_id": "my-test-001",
      "user_content": {
        "parts": [{ "text": "Summarize: Your input text here..." }],
        "role": "user"
      },
      "final_response": {
        "parts": [{ "text": "- Expected bullet point 1\n- Expected bullet point 2" }],
        "role": "model"
      },
      "intermediate_data": {
        "tool_uses": [],
        "intermediate_responses": []
      }
    }
  ],
  "session_input": {
    "app_name": "text_summarizer",
    "user_id": "test_user",
    "state": {}
  }
}
```

## Beyond Free Tier

With a Google Cloud project + billing enabled, you unlock:

- **`adk optimize`** — automatically improve agent instructions using AI
- **`rubric_based_final_response_quality_v1`** — custom quality rubrics
- **`hallucinations_v1`** — check if the agent makes unsupported claims
- **`safety_v1`** — ensure responses are safe and harmless
- **`final_response_match_v2`** — LLM-judged semantic matching

These measure **quality** instead of word overlap, unlike the free-tier `response_match_score`.