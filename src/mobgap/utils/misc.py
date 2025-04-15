"""Miscellaneous utility functions."""

import os
from typing import Any

_NONE = object()


def get_env_var(name: str, default: Any = _NONE) -> str:
    """Get an environment variable.

    We first check if it exists, if not, we attempt to load a `.env` file, which might be present during development.
    """
    if name not in os.environ:
        from dotenv import load_dotenv

        load_dotenv()

    if name not in os.environ and default is _NONE:
        raise ValueError(
            f"The environment variable {name} is not set. Please set it in your environment.\n\n"
            "If you are developing mobgap, you can alternatively place a `.env` file in the project root."
        )
    return os.environ.get(name, default)
