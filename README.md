# Text Summarizer Agent — ADK Evaluation Loop Demo

A text summarization agent built with Google's **Agent Development Kit (ADK)** that demonstrates how to use evaluation loops to systematically improve agent prompts and outputs.

## What This Project Does

This agent takes long text and converts it into concise bullet-point summaries. More importantly, it shows you how to **evaluate** and **improve** agent quality using ADK's built-in evaluation framework.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- `uv` (package manager)
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- (Optional) A free OpenRouter API key from [OpenRouter](https://openrouter.ai/keys)

### 2. Install ADK

```bash
uvx google-agents-cli setup
pip install "google-adk[eval]"
```

### 3. Configure API Keys

Edit `.env` and set your keys:

```env
# Gemini (default)
GEMINI_API_KEY=your_actual_key_here

# OpenRouter (optional, for free models)
OPENROUTER_API_KEY=your_actual_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# Switch provider: "gemini" or "openrouter"
MODEL_PROVIDER=gemini

# OpenRouter model alias (gemma, qwen, nvidia)
MODEL_ALIAS=gemma
```

### 4. Run the Agent

**Option A: Web UI (recommended for testing)**

```bash
adk web .
```

Open http://127.0.0.1:8000 in your browser, select `text_summarizer`, and start chatting.

**Option B: CLI (quick test)**

```bash
adk run text_summarizer "Summarize: Your long text here..."
```

### 5. Run Evaluations

```bash
adk eval text_summarizer tests/eval/simple_test.test.json --config_file_path tests/eval/test_config.json --print_detailed_results
```

Or run the full evaluation set:

```bash
adk eval text_summarizer tests/eval/summarizer_eval_set.evalset.json --config_file_path tests/eval/test_config.json --print_detailed_results
```

## Model Switching (Dependency Injection)

The agent supports switching between Gemini and OpenRouter via environment variables. The same `MODEL_PROVIDER` env var controls both the agent and the auto-optimizer.

### Available Models

| Provider | Model | Alias | Cost |
|----------|-------|-------|------|
| Gemini | `gemini-3.5-flash-lite` | — | Free tier |
| OpenRouter | `google/gemma-4-26b-a4b-it:free` | `gemma` | Free |
| OpenRouter | `qwen/qwen3.8-27b:free` | `qwen` | Free |
| OpenRouter | `nvidia/nemotron-3.5-lightning:free` | `nvidia` | Free |

### How It Works

In `agent.py`:

```python
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "gemini")

def get_model():
    if MODEL_PROVIDER == "openrouter":
        model_name = OPENROUTER_MODELS.get(MODEL_ALIAS, MODELS["openrouter"])
        return LiteLlm(model=f"openrouter/{model_name}")
    return MODELS.get("gemini")
```

To switch models, just change the env var:

```bash
# Use Gemini
set MODEL_PROVIDER=gemini
adk web .

# Use OpenRouter free models
set MODEL_PROVIDER=openrouter
set MODEL_ALIAS=gemma
adk web .
```

## Testing in the ADK Web UI

Once the agent is running at http://127.0.0.1:8000, try these prompts to see how the agent performs.

### Good Examples (Expected Behavior)

These inputs should produce clean, accurate bullet-point summaries:

**Example 1: Simple factual text**
```
Summarize: Dogs are domesticated mammals known for their loyalty and companionship. They are often called man's best friend. Dogs come in many breeds, varying in size, color, and temperament. They have been bred for various purposes including hunting, herding, guarding, and companionship. Dogs are social animals that thrive on interaction with humans and other dogs.
```

**Expected output:**
```
- Dogs are domesticated mammals valued for loyalty and companionship.
- They come in many breeds with varying size, color, and temperament.
- Dogs have been bred for hunting, herding, guarding, and companionship.
- They are social animals that thrive on human and canine interaction.
```

**Example 2: Technical text**
```
Summarize the following text:

Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.
```

**Example 3: News article**
```
Summarize the following text:

Scientists at CERN have announced the discovery of a new subatomic particle that could reshape our understanding of fundamental physics. The particle, tentatively named the Xi-cc-double-plus, is a type of baryon containing two charm quarks and one up quark. Unlike most known baryons, which contain at most one heavy quark, this particle's unique composition makes it an ideal laboratory for testing quantum chromodynamics, the theory describing the strong nuclear force. The discovery was made using the Large Hadron Collider's beauty experiment, which analyzed data from proton-proton collisions at energies of up to 13 teraelectronvolts. Researchers expect this finding to open new avenues for understanding how quarks bind together to form matter.
```

---

### Bad Examples (What NOT to Input)

These inputs will produce poor results or violate the agent's rules:

**Bad 1: No text to summarize**
```
Summarize this.
```

The agent has nothing to work with. It will either ask for text or produce a meaningless response.

**Bad 2: Contradictory instructions**
```
Summarize this text but don't use bullet points and make it as long as possible:
Dogs are friendly animals.
```

This violates the agent's core rule: always respond with bullet points. The agent may produce inconsistent output.

**Bad 3: Asking for information not in the text**
```
Summarize: The weather today is sunny. What time does the store close?
```

The text contains no information about store hours. A good summary should only reflect what's in the source text, but this input confuses the agent's purpose.

**Bad 4: Vague with no clear source text**
```
Tell me about dogs.
```

This isn't a summarization task — it's a general knowledge question. The agent is designed to summarize provided text, not answer questions from its training data.

**Bad 5: Extremely long input with no structure**
```
[paste 10,000 words of unstructured text with no clear topic]
```

While the agent can handle long text, very long inputs without clear structure may produce summaries that miss key points or become too generic.

---

### Edge Cases (Interesting to Test)

These test the agent's boundaries:

**Edge case 1: Very short text**
```
Summarize: It rained today.
```

How concise can the agent get while still producing a valid bullet point?

**Edge case 2: Multiple topics**
```
Summarize: The stock market rose 2% today. Meanwhile, scientists discovered a new species of frog in the Amazon. The president signed a new trade agreement with Canada. Local schools announced a new lunch program.
```

Can the agent capture all four distinct topics in separate bullet points?

**Edge case 3: Text with numbers and data**
```
Summarize: In Q3 2024, Acme Corp reported revenue of $4.2 billion, up 12% from $3.75 billion in Q3 2023. Net income was $890 million compared to $720 million the prior year. Operating margins expanded to 21.2% from 19.1%. The company added 15,000 new enterprise customers, bringing the total to 340,000.
```

Does the agent correctly preserve the numbers in the summary?

## How the Evaluation Loop Works

The evaluation loop is the core learning objective of this project:

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

### Evaluation Criteria Used

| Criterion | What It Measures | Threshold |
|---|---|---|
| `tool_trajectory_avg_score` | Did the agent use the right tools in the right order? | 1.0 (exact) |
| `response_match_score` | How similar is the output to the expected response? (ROUGE-1) | 0.5 |

### Running the Evaluation Loop

```bash
# Step 1: Run evaluation
adk eval text_summarizer tests/eval/summarizer_eval_set.evalset.json \
  --config_file_path tests/eval/test_config.json \
  --print_detailed_results

# Step 2: Check results — look for FAILED tests

# Step 3: Edit agent.py instructions to fix failures

# Step 4: Re-run evaluation to verify improvement
```

### Interpreting Results

```
Metric: response_match_score, Status: PASSED, Score: 0.679, Threshold: 0.5
```

- **Score: 0.679** — The agent's response had 67.9% word overlap with the expected response
- **Threshold: 0.5** — Minimum acceptable score is 50%
- **Status: PASSED** — Score meets or exceeds threshold

If a test fails, the output shows a side-by-side comparison of expected vs actual response, so you can see exactly what went wrong.

## Hands-On: Learning the Evaluation Loop

This is where you learn by doing. The process is: **edit instructions -> run eval -> compare scores -> learn**.

### Quick Start

```bash
cd C:\Users\carlo\Desktop\agents\text_summarizer
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

**Change:** Add rules that force different wording than expected output.

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

**Run eval:**
```bash
python eval_exercise.py
```

**Result:**
```
>>> CURRENT SCORE: 0.5116 <<<
```

**What happened:** Score dropped from 0.6600 to 0.5116. Rules 7 and 8 forced the model to use different words than the expected output, reducing ROUGE-1 word overlap.

**Lesson:** Restrictive rules that force different phrasing can hurt scores that measure word overlap.

---

### Experiment 2: Make Instructions Simpler

**Change:** Simplify the bullet count rule.

```python
# In agent.py, change rule 4 from:
#   "4. Aim for 3-7 bullet points depending on the length and complexity of the input."
# To:
#   "4. Aim for 3-5 bullet points depending on the length and complexity of the input."
```

**Run eval:**
```bash
python eval_exercise.py
```

**Result:**
```
>>> CURRENT SCORE: 0.7312 <<<
```

**What happened:** Score increased from 0.6600 to 0.7312. The model produces slightly fewer bullet points, which happens to match the expected output more closely.

**Lesson:** Small changes can have surprising effects. Test everything.

---

### Experiment Results Summary

| Run | Instruction Change | Score | Delta |
|---|---|---|---|
| 1 | Original baseline | 0.6600 | — |
| 2 | Added "no The/It" + "under 20 words" | 0.5116 | -0.1484 |
| 3 | Reverted + changed "3-7" to "3-5" | 0.7312 | +0.0712 |

### What to Try Next

Edit `agent.py` instructions, then run `python eval_exercise.py`:

| Experiment | What to Change | Expected Effect |
|---|---|---|
| Remove bullets | Delete rule 1 entirely | Score may drop (expected output uses bullets) |
| More verbose | Add "Include as much detail as possible" | Score may drop (too wordy) |
| More concise | Change rule 2 to "5-10 words maximum" | Score may drop (too short) |
| Simple vocabulary | Add "Use only common words" | Score may rise (more word overlap) |
| Different format | Change "- " to "* " prefix | Score may drop (expected uses "- ") |

### Important Insight

The `response_match_score` uses **ROUGE-1**, which measures **word overlap**, not quality. This means:

- Using the **same words** as expected -> higher score
- Using **different words** (even if better) -> lower score
- The "best" instructions depend on what you're optimizing for

If you want to measure **quality** instead of word overlap, you need the free-tier incompatible criteria (requires Google Cloud billing):
- `final_response_match_v2` — LLM judges if responses mean the same thing
- `rubric_based_final_response_quality_v1` — Custom quality rubrics

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

Instead of manually editing instructions, you can use `auto_optimize.py` to let an LLM automatically improve your agent's instructions through the eval loop.

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
# Uses same MODEL_PROVIDER as agent.py
set MODEL_PROVIDER=gemini
python auto_optimize.py

# With options
python auto_optimize.py --max-iterations 5 --patience 3

# Force specific optimizer model
python auto_optimize.py --optimizer-model gemini-3.5-flash-lite
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

The optimizer learned to add: "closely mirroring the key terms, phrasing, and sentence structures found in the source text" — which increased ROUGE-1 word overlap.

### Optimization History

Each run saves history to `optimization_history.json`:

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

## Project Structure

```
text_summarizer/
|-- agent.py                    # Agent definition (instructions + model)
|-- __init__.py                 # Exposes root_agent
|-- .env                        # API key configuration
|-- pyproject.toml              # Project dependencies
|-- eval_exercise.py            # Run this to test instruction changes
|-- auto_optimize.py            # Auto-optimize instructions via eval loop
|-- README.md                   # This file
|-- optimization_history.json   # History from auto_optimize.py runs
|-- check_free_models.py        # List available OpenRouter free models
|-- tests/
    |-- eval/
        |-- simple_test.test.json               # Unit test (1 case)
        |-- summarizer_eval_set.evalset.json    # Full eval set (3 cases)
        |-- test_config.json                    # Evaluation criteria
```

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

- **`adk optimize`** — Automatically improve agent instructions using AI
- **`rubric_based_final_response_quality_v1`** — Custom quality rubrics
- **`hallucinations_v1`** — Check if the agent makes unsupported claims
- **`safety_v1`** — Ensure responses are safe and harmless
- **`final_response_match_v2`** — LLM-judged semantic matching

## License

This project is for learning purposes. Built with [Google ADK](https://adk.dev).
