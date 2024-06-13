"""Evaluation utils for final pipeline outputs.

Note, that this just provides some reexports of the evaluation functions from some of the other modules.

"""

__all__ = [
    "categorize_intervals_per_sample",
    "categorize_intervals",
    "get_matching_intervals",
    "get_default_error_transformations",
    "get_default_error_aggregations",
    "icc",
    "error",
    "rel_error",
    "abs_error",
    "abs_rel_error",
]

from mobgap.gsd.evaluation import (
    abs_error,
    abs_rel_error,
    categorize_intervals,
    categorize_intervals_per_sample,
    error,
    get_default_error_aggregations,
    get_default_error_transformations,
    get_matching_intervals,
    icc,
    rel_error,
)
