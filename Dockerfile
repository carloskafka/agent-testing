FROM python:3.14-slim

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

COPY text_summarizer ./text_summarizer
COPY patch-adk-devui-mobile.py ./patch-adk-devui-mobile.py

WORKDIR /workspace/text_summarizer
RUN uv sync --frozen

# Patch the bundled ADK dev UI for phones. Must run AFTER uv sync, which is what
# puts google-adk in the venv, and before anything starts the server. The script
# is idempotent and fails the build if a future google-adk changes the bundle
# underneath it, so an upgrade cannot silently ship a patch that does nothing.
RUN python /workspace/patch-adk-devui-mobile.py

WORKDIR /workspace

EXPOSE 8000

ENTRYPOINT []
CMD ["text_summarizer/.venv/bin/adk", "web", "--host", "0.0.0.0", "--port", "8000", "text_summarizer"]