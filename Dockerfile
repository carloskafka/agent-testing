FROM python:3.14-slim

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

COPY text_summarizer ./text_summarizer
COPY patch-adk-devui-mobile.py ./patch-adk-devui-mobile.py
COPY patch-adk-devui-confirm.py adk-confirm-options.js ./

WORKDIR /workspace/text_summarizer
RUN uv sync --frozen

# Patch the bundled ADK dev UI for phones. Must run AFTER uv sync, which is what
# puts google-adk in the venv, and before anything starts the server. The script
# is idempotent and fails the build if a future google-adk changes the bundle
# underneath it, so an upgrade cannot silently ship a patch that does nothing.
RUN python /workspace/patch-adk-devui-mobile.py

# Give tool confirmations clickable options. `ask_user` pauses a turn by
# requesting one, and the stock UI draws that request as a checkbox plus a raw
# JSON dump of the options -- so a menu reaches the user as text. Two halves,
# both required: this rewrites the one expression that builds the confirmation
# response, and the shim beside it renders the buttons. Same idempotence, and the
# same fail-the-build guard on vendor drift, as the patch above.
RUN python /workspace/patch-adk-devui-confirm.py

WORKDIR /workspace

EXPOSE 8000

ENTRYPOINT []
# `adk web` with one addition: the active vault served read-only at /vault, so
# the note titles in every **Sources** line are clickable. Same app, same port,
# still one process -- see text_summarizer/serve.py.
CMD ["text_summarizer/.venv/bin/python", "-m", "text_summarizer.serve", "--host", "0.0.0.0", "--port", "8000", "text_summarizer"]