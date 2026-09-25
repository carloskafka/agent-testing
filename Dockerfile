FROM python:3.14-slim

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

COPY text_summarizer ./text_summarizer

WORKDIR /workspace/text_summarizer
RUN uv sync --frozen

WORKDIR /workspace

EXPOSE 8000

ENTRYPOINT []
CMD ["text_summarizer/.venv/bin/adk", "web", "--host", "0.0.0.0", "--port", "8000", "text_summarizer"]