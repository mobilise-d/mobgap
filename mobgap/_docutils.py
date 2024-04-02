"""Some small utilities to improve writing docstrings.

For now, this just exports some functions from scipy._lib.doccer, to have only one place to import from.
While, the ``doccer`` submodule of scip[y is not part of the public API, it seems to be stable enough to use it here.
"""

from typing import Callable, Optional

from scipy._lib.doccer import filldoc, inherit_docstring_from


def filldoc_with_better_error(docdict: dict[str, str]) -> filldoc:
    def inner(func: Callable) -> str:
        try:
            return filldoc(docdict)(func)
        except ValueError as e:
            if "unsupported format character" in str(e):
                raise ValueError(
                    "Your docstring contains a single '%' character. "
                    "This is not supported by this decorator. "
                    "If you want to type a '%' character, you need to type '%%'."
                ) from e
            raise

    return inner


def make_filldoc(docdict: dict[str, str], *, doc_summary: Optional[str] = None) -> filldoc:
    """Create a new doc-filler from a dictionary.

    This can be applied to a function, method, or class to substitute ``%(key)s`` occurrences in its docstring.

    Parameters
    ----------
    docdict
        Dictionary with docstring keys and values.
        These can be multiline strings and will be dedented before substitution.
    doc_summary
        An optional summary line for the docstring of the returned decorator function.
        This is only relevant for inclusion in the documentation.

    Note
    ----
    If you use this decorator, your docstrings can not contain a single ``%`` character.
    If you want to type a ``%`` character, you need to type ``%%``.
    """
    inner = filldoc_with_better_error(docdict)

    if doc_summary is None:
        doc_summary = "Fill docstring from dictionary."

    full_doc = f"""{doc_summary}

    Available keys for the substitution::

    {list(docdict.keys())}

    """
    inner.__doc__ = full_doc
    inner._dict = docdict
    return inner


__all__ = ["make_filldoc", "inherit_docstring_from"]
