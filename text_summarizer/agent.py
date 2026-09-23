import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from .obsidian_tools import build_obsidian_tools

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "gemini")

MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "openrouter": "google/gemma-4-26b-a4b-it:free",
}

OPENROUTER_MODELS = {
    "gemma": "google/gemma-4-26b-a4b-it:free",
    "qwen": "qwen/qwen3.8-27b:free",
    "nvidia": "nvidia/nemotron-3.5-lightning:free",
}

MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "")


def get_model():
    if MODEL_PROVIDER == "openrouter":
        model_name = OPENROUTER_MODELS.get(MODEL_ALIAS, MODELS["openrouter"])
        return LiteLlm(model=f"openrouter/{model_name}")
    return MODELS.get("gemini")


model = get_model()

root_agent = LlmAgent(
    name="text_summarizer",
    model=model,
    description="A text summarization agent that converts long text into concise bullet-point summaries.",
    instruction="""You are a text summarization agent. Your job is to take long text provided by the user and convert it into a short, clear bullet-point summary.

Rules:
1. Always respond with bullet points using the - prefix (dash followed by space).
2. Each bullet point should be a single concise sentence.
3. Capture the main ideas by closely mirroring the key terms, phrasing, and sentence structures found in the source text.
4. Aim for 3-5 bullet points depending on the length and complexity of the input.
5. Do not add information that is not present in the original text.
6. Use clear, professional language, maintaining a direct, factual tone that reflects the core statements of the input.""",
    tools=build_obsidian_tools(),
)
