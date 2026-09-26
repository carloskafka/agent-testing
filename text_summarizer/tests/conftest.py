"""Test bootstrap.

Importing anything under ``text_summarizer`` runs the package ``__init__``,
which calls ``setup_observability()``. With a real ``LANGFUSE_PUBLIC_KEY``
that performs a network auth check (tens of seconds against an unreachable
host), so the unit tests blank the key first -- ``load_dotenv`` uses
``setdefault`` and will not put it back.

Only the variables the tests need are touched; everything else is untouched.
"""

import os

os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ.setdefault("CACHE_ENABLED", "true")
