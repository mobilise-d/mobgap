"""Some small utilities to improve writing docstrings.

For now, this just exports some functions from scipy._lib.doccer, to have only one place to import from.
While, the ``doccer`` submodule of scip[y is not part of the public API, it seems to be stable enough to use it here.
"""
from scipy._lib.doccer import filldoc

__all__ = ["filldoc"]
