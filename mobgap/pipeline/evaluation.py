"""Evaluation utils for final pipeline outputs.

Note, that this just provides some reexports of the evaluation functions from some of the other modules.

"""

__all__ = [
    "categorize_intervals_per_sample",
    "categorize_intervals",
    "get_matching_intervals",
    "ErrorTransformFuncs",
    "error",
    "rel_error",
    "abs_error",
    "abs_rel_error",
    "get_default_error_transformations",
    "get_default_error_aggregations",
    "CustomErrorAggregations",
    "icc",
    "loa",
    "quantiles",
]

from mobgap.gait_sequences.evaluation import (
    categorize_intervals,
    categorize_intervals_per_sample,
    get_matching_intervals,
)
from mobgap.pipeline._error_metrics import (
    CustomErrorAggregations,
    ErrorTransformFuncs,
    abs_error,
    abs_rel_error,
    error,
    get_default_error_aggregations,
    get_default_error_transformations,
    icc,
    loa,
    quantiles,
    rel_error,
)
