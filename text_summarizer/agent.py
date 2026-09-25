import os

from google.adk.agents import LlmAgent
from google.adk.models import FallbackModel, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.genai.types import Content, Part

from .obsidian_tools import build_obsidian_tools
from .second_brain import find_cached_summary, log_conversation, save_summary_to_second_brain

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "gemini")

GEMINI_MODEL = "gemini-3.5-flash-lite"

# Only models with a ":free" suffix are real free-tier models on OpenRouter.
OPENROUTER_MODELS = {
    "gemma": "google/gemma-4-26b-a4b-it:free",
    "qwen": "qwen/qwen3.8-27b:free",
    "nvidia": "nvidia/nemotron-3.5-lightning:free",
}

MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "")


def _free_openrouter_models() -> list[str]:
    return [
        name for name in OPENROUTER_MODELS.values() if name.endswith(":free")
    ]


def _openrouter_llm(model_name: str) -> LiteLlm:
    return LiteLlm(model=f"openrouter/{model_name}")


def get_model():
    if MODEL_PROVIDER == "openrouter":
        primary_name = OPENROUTER_MODELS.get(MODEL_ALIAS) or _free_openrouter_models()[0]
        fallback_names = [
            name for name in _free_openrouter_models() if name != primary_name
        ]
        return FallbackModel(
            models=[
                _openrouter_llm(primary_name),
                *(_openrouter_llm(name) for name in fallback_names),
            ]
        )
    # Default provider is Gemini; fall back to free OpenRouter models on quota
    # exhaustion (HTTP 429) or transient 5xx errors.
    return FallbackModel(
        models=[
            GEMINI_MODEL,
            *(_openrouter_llm(name) for name in _free_openrouter_models()),
        ]
    )


model = get_model()


def _last_user_text(llm_request) -> str:
    """Extract the text of the most recent user message from the LLM request."""
    contents = llm_request.contents or []
    for content in reversed(contents):
        if getattr(content, "role", None) == "user":
            return "".join(part.text or "" for part in content.parts or [])
    return ""


def cache_hit_before_model(callback_context, llm_request):
    """If the vault already has a summary for this exact text, return it with zero LLM calls.

    Returns an LlmResponse (which skips the model invocation) on a fingerprint
    match, otherwise None to let the model run normally.
    """
    user_text = _last_user_text(llm_request)
    if not user_text:
        return None
    cached = find_cached_summary(user_text)
    if not cached:
        return None
    return LlmResponse(
        content=Content(role="model", parts=[Part(text=cached)]),
        turn_complete=True,
    )


root_agent = LlmAgent(
    name="text_summarizer",
    model=model,
    description="A text summarization agent that converts long text into concise bullet-point summaries and stores them in an Obsidian vault (second brain).",
    before_model_callback=cache_hit_before_model,
    instruction="""You are a text summarization agent backed by an Obsidian vault that acts as a second brain. Your job is to take long text provided by the user, convert it into a short, clear bullet-point summary, and persist it in the vault so the knowledge is graph-aware and reusable.

Rules:
1. Always respond with bullet points using the - prefix (dash followed by space).
2. Each bullet point should be a single concise sentence.
3. Capture the main ideas by closely mirroring the key terms, phrasing, and sentence structures found in the source text.
4. Aim for 3-5 bullet points depending on the length and complexity of the input.
5. Do not add information that is not present in the original text.
6. Use clear, professional language, maintaining a direct, factual tone that reflects the core statements of the input.
7. USE THE SECOND BRAIN - RETRIEVE FIRST: Before summarizing, identify the 1-3 main topics of the input. Use the vault search tools (search_text) and note_read to look up existing notes on those topics and on the Second Brain Index. If relevant related notes exist, incorporate their key facts into your summary: add a "## From the vault" section after the bullets that lists the related notes you consulted (with [[wikilinks]]) and briefly how they connect. If nothing relevant exists, skip this and say so in one line.
8. ALWAYS persist the summary: after producing the summary, call save_summary_to_second_brain with a short descriptive title, the bullet-point content, a comma-separated list of the 2-5 main topics it covers, and source_text set to the FULL original text the user provided (verbatim). This keeps the vault graph connected and enables zero-cost dedup on repeated requests.
9. ALWAYS log the conversation: after producing and persisting your final answer, call log_conversation with the user's exact message and your final answer, so every exchange is recorded in the vault's chat log for later recall.
10. Mention in your final answer that the summary was saved to the second brain and its title.""",
    tools=[
        FunctionTool(save_summary_to_second_brain),
        FunctionTool(log_conversation),
        *build_obsidian_tools(),
    ],
)