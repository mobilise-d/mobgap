"""Evaluation utils for final pipeline outputs.

Note, that this just provides some reexports of the evaluation functions from some of the other modules.

"""

import pandas as pd
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.gait_sequences.evaluation import (
    categorize_intervals,
    categorize_intervals_per_sample,
    get_matching_intervals,
)
from mobgap.pipeline import MobilisedPipelineUniversal
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
from mobgap.utils.df_operations import apply_transformations

E = ErrorTransformFuncs

_errors = [
    ("walking_speed_mps", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
    ("stride_length_m", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
    ("cadence_spm", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
]
_error_aggregates = []


def pipeline_per_datapoint_score(
    pipeline: MobilisedPipelineUniversal, datapoint: BaseGaitDatasetWithReference
) -> dict:  # TODO: correct type for argument pipeline?
    """Evaluate the performance of the Mobilise-D pipeline for primary DMOs estimation on a single datapoint.

    .. warning:: This function is not meant to be called directly, but as a scoring function in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function calculates the median DMO reference and detected values and errors on a per datapoint level
    (per trial/recording).

    The following metrics are calculated:

    - The error, absolute error, relative error, and absolute relative error for each WB. The WB-level metrics are
      calculated as the average values of the DMO for each WB in the algorithm output.
      Both the detected and reference values of the DMOs are taken directly from the wb-level reference data.
      (``wb_level_<DMO>_values_with_errors``). These are returned as a dataframe wrapped in ``no_agg``.
      The dataframe also contains the average walking speed for each WB extracted from the reference system to provide
      context for further analysis.
    - The average WB-level error metrics on a per-data-point level. These are returned as ``combined__<DMO>_<metric>`` and
      will be averaged over all datapoints in the Scorer.

    Parameters
    ----------
    pipeline
        An instance of :class:`~mobgab.pipeline.MobilisedPipelineUniversal`that wraps the pipeline that should #TODO: correct type for argument pipeline?
        be evaluated.
    datapoint
        The datapoint to be evaluated.

    Returns
    -------
    dict
        A dictionary containing the performance metrics.
        Note, that some results are wrapped in a ``no_agg`` object or other aggregators.
        The results of this function are not expected to be parsed manually, but rather the function is expected to be
        used in the context of the :func:`~tpcp.validate.validate`/:func:`~tpcp.validate.cross_validate` functions or
        similar as scorer.
        This functions will aggregate the results and provide a summary of the performance metrics.

    """
    pipeline.safe_run(datapoint)
    # Extracting main results (average values per WB)
    # We don't drop NaNs here, as we want to keep Cadence values even if other values are missing
    calculated_per_wb = pipeline.per_wb_parameters_[
        ["walking_speed_mps", "stride_length_m", "cadence_spm"]
    ]
    reference_per_wb = datapoint.reference_parameters_.wb_list[
        ["avg_walking_speed_mps", "avg_stride_length_m", "avg_cadence_spm"]
    ]
    reference_per_wb.columns = reference_per_wb.columns.str.removeprefix("avg_")

    # Combined evaluation
    # Agg/Combined Evaluation
    median_parameters = (
        pd.concat(
            {"detected": calculated_per_wb.median(), "reference": reference_per_wb.median()}
        )
        .to_frame()
        .swaplevel()
        .T
    )
    median_parameters_errors = apply_transformations(median_parameters, _errors)
    median_parameters_with_errors = pd.concat([median_parameters_errors, median_parameters], axis=1)
    median_parameters_with_errors.columns = ["__".join(levels) for levels in median_parameters_with_errors.columns]
    assert len(median_parameters_with_errors) == 1
    # There is only one value per datapoint, so we can just take the first row
    median_parameters_with_errors = median_parameters_with_errors.add_prefix("combined__").iloc[0]

    return {
        # We also pass the results of the combined analysis directly as single values.
        # This way there medians are available directly as a single value for optimization.
        **median_parameters_with_errors.to_dict(),
        "combined_error": no_agg(median_parameters_with_errors),
    }


def pipeline_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    _: MobilisedPipelineUniversal,  # TODO: is it the correct object type?
    dataset: BaseGaitDatasetWithReference,
) -> tuple[dict[str, any], dict[str, list[any]]]:
    """Aggregate the results of the pipeline evaluation.

    .. warning:: This function is not meant to be called directly, but as ``final_aggregator`` in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function aggregates the performance metrics as follows:

    - The raw WB level values are combined into one dataframe each across the entire dataset.
    - All other values are passed through unchanged.

    """
    data_labels = [d.group_label for d in dataset]
    data_label_names = data_labels[0]._fields

    combined_errors = single_results.pop("combined_error")
    combined_errors = pd.concat(combined_errors, keys=data_labels, names=data_label_names, axis=1).dropna()

    aggregated_single_results = {
        "raw__combined_errors": combined_errors,
    }

    return agg_results, {**single_results, **aggregated_single_results}


#: :data:: pipeline_score
#: Scorer class instance for the Mobilise-D pipeline.
pipeline_score = Scorer(pipeline_per_datapoint_score, final_aggregator=pipeline_final_agg)
pipeline_score.__doc__ = """Scorer for the Mobilise-D pipeline evaluation.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using the :func:`pipeline_per_datapoint_score` function as
per-datapoint scorer and the :func:`pipeline_final_agg` function as final aggregator.
For more information about Scorer, head to the tpcp documentation (:class:`~tpcp.validate.Scorer`).
For usage information in the context of mobgap, have a look at the :ref:`evaluation example <sl_evaluation>` for GSD.

The following metrics are calculated:

Raw metrics (part of the single results):

- ``single__raw__wb_level_<dmo>_values_with_errors``: A dataframe containing the WB level values and errors for each WB in the
  dataset.
  In addition, to the DMO values and errors, the dataframe also contains the average walking speed for each WB
  extracted from the reference system to provide context for further analysis.

Metrics per datapoint (single results):
*These values are all provided as a list of values, one per datapoint.*

- ``single__wb__{metric}``: The median of WB level values for each datapoint.

Aggregated metrics (agg results):

- ``agg__wb__{metric}``: The average over the per-datapoint-averaged WB level metrics.
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
    "pipeline_score",
    "pipeline_per_datapoint_score",
    "pipeline_final_agg",
]
