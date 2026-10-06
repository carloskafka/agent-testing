"""``adk web`` plus a read-only HTTP view of the active vault.

Why this exists
---------------
Every ``**Sources**`` line ends in a note title, and a title is only useful if
clicking it takes you to the note. Obsidian's own ``[[wikilink]]`` syntax is
meaningless to a browser -- the dev UI renders it as literal ``[Email]`` -- so
the renderer emits a real markdown link to ``/vault/<path>``
(see :mod:`text_summarizer.sources`). This module is what serves that path.

It replaces the container's ``adk web`` command with the same FastAPI app plus
two ``StaticFiles`` mounts -- the vault at ``/vault`` and the hourly curves at
``/chart`` -- so there is still a single process and a single port.
Two alternatives were rejected:

* ``obsidian://`` deep links -- the canonical Obsidian answer, but the protocol
  handler only exists on a machine with Obsidian installed. This deployment runs
  Obsidian itself in a container, and the browser is on the host, so the link
  would resolve to nothing.
* Serving from ``obsidian-mcp`` -- it is already an HTTP server, but it speaks
  MCP and has no file-serving route; adding one would mean a second app to keep
  alive, defeating the point of the sidecar being a separate process.

The mount is read-only and rooted at the **resolved** vault (the same
``VAULT_ROOT`` the tools write to), so ``/vault`` cannot be walked out of with
``..``: ``StaticFiles`` resolves and rejects paths that escape its directory.
It serves markdown as ``text/plain`` -- the browser shows the note's source,
which is what a link in a chat transcript should do. Rendering it as a vault
page would need an Obsidian instance, which is a different product.

The ``/chart`` mount is the same idea for a graph rather than a note: the weather
answer ends in an image, and an image needs a URL. See :mod:`text_summarizer.chart`
for why it is drawn as SVG and served from here rather than rendered by a charting
library.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys

# Importing the package loads .env and initialises observability, which is what
# the `adk web` command would do anyway via the agent module.
from .second_brain import VAULT_ROOT
from .sources import VAULT_WEB_PREFIX


def build_app(agents_dir: str, host: str, port: int):
    """The stock ADK web app, with the vault mounted at ``/vault``."""
    from fastapi.staticfiles import StaticFiles
    from google.adk.cli.fast_api import get_fast_api_app

    app = get_fast_api_app(
        agents_dir=agents_dir,
        web=True,
        host=host,
        bind_host=host,
        port=port,
        use_local_storage=True,
    )

    if VAULT_ROOT and os.path.isdir(VAULT_ROOT):
        app.mount(
            VAULT_WEB_PREFIX,
            StaticFiles(directory=VAULT_ROOT, html=False, follow_symlink=False),
            name="vault",
        )
        logging.getLogger(__name__).info(
            "serving vault read-only at %s -> %s", VAULT_WEB_PREFIX, VAULT_ROOT
        )
    else:
        # Not fatal: the agent still answers, the Sources block just renders
        # with unclickable titles (note_href returns None for a missing root).
        logging.getLogger(__name__).warning(
            "vault root %r is not a directory; %s will not be served, so source "
            "titles render as plain wikilinks",
            VAULT_ROOT,
            VAULT_WEB_PREFIX,
        )

    _mount_charts(app, StaticFiles)
    return app


def _mount_charts(app, static_files) -> None:
    """Mount the hourly curves at ``/chart``, when there is a directory to serve.

    Same shape as the vault mount, and for the same reason: an ``<img>`` in the answer
    has to resolve to something, and there is no other HTTP server in this process to
    resolve it. ``StaticFiles`` brings the traversal defence with it, which matters more
    here than it does for the vault only because the chart directory holds generated
    files -- though those are content-addressed names this process wrote, so there is
    nothing to walk out of either way.

    **The directory is created empty if absent, rather than skipped.** The chart is drawn
    lazily, by the first weather turn, which is typically long after this app is built;
    skipping the mount because nothing has been drawn yet would mean every image in
    every answer is a broken one until the server is restarted. Creating it is a
    no-op on a deployment that has already drawn one.

    ``follow_symlink=False`` matches the vault. Nothing legitimate here is a symlink,
    and a chart directory under ``.adk`` is exactly the kind of path where an operator
    might have left one.
    """
    from .chart import CHART_WEB_PREFIX, chart_dir

    directory = chart_dir()
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        # Not fatal, and the message names the cause rather than just the path: a
        # read-only `.adk` is the likely one, and "chart_dir is not a directory"
        # would send whoever reads the log looking in the wrong place.
        logging.getLogger(__name__).warning(
            "chart directory %r is not writable (%s); %s will not be served, so "
            "hourly curves render as a broken image",
            directory,
            exc,
            CHART_WEB_PREFIX,
        )
        return
    app.mount(
        CHART_WEB_PREFIX,
        static_files(directory=directory, html=False, follow_symlink=False),
        name="chart",
    )
    logging.getLogger(__name__).info(
        "serving hourly curves at %s -> %s", CHART_WEB_PREFIX, directory
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ADK web UI with the active vault served read-only."
    )
    parser.add_argument("agents_dir", nargs="?", default=os.getcwd())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    agents_dir = os.path.abspath(args.agents_dir)
    app = build_app(agents_dir, args.host, args.port)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        print(
            f"\n+{'-' * 77}+\n"
            f"| ADK web server (with {VAULT_WEB_PREFIX}/ -> the active vault){' ' * 21}|\n"
            f"| Open http://{args.host}:{args.port}{' ' * (50 - len(str(args.port)))}|\n"
            f"+{'-' * 77}+",
            flush=True,
        )
        yield

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
