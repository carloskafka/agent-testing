from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from .observability import setup_observability

setup_observability()

from .agent import root_agent

__all__ = ["root_agent"]
