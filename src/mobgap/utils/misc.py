"""Miscellaneous utility functions."""

import os
from pathlib import Path
from typing import Any

from mobgap import PROJECT_ROOT

_NONE = object()


def get_env_var(name: str, default: Any = _NONE) -> str:
    """Get an environment variable.

    We first check if it exists, then load the project-root `.env` file if needed.
    """
    if name not in os.environ:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(Path(PROJECT_ROOT) / ".env")

    if name not in os.environ and default is _NONE:
        raise ValueError(
            f"The environment variable {name} is not set. Please set it in your environment.\n\n"
            "If you are developing mobgap, you can alternatively place a `.env` file in the project root."
        )
    return os.environ.get(name, default)
