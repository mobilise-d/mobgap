"""Evaluation tools and functions for final pipeline outputs."""

import pandas as pd
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
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
from mobgap.pipeline.base import BaseMobilisedPipeline
from mobgap.utils.df_operations import apply_transformations

E = ErrorTransformFuncs

_errors = [
    ("walking_speed_mps", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
    ("stride_length_m", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
    ("cadence_spm", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
]
_error_aggregates = []


def pipeline_per_datapoint_score(pipeline: BaseMobilisedPipeline, datapoint: BaseGaitDatasetWithReference) -> dict:
    """Evaluate the performance of the Mobilise-D pipeline for primary DMOs estimation on a single datapoint.

    .. warning:: This function is not meant to be called directly, but as a scoring function in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function calculates and evaluates DMO values in a two-ways fashion:

    - Aggregate analysis ("originally referred to as "Combined Evaluation"): Median DMO reference and estimated values
      are calculated on a per-datapoint level. As a result, a pair of reference and estimated values is returned for
      each datapoint (datapoint = trial in the case of the laboratory dataset, datapoint = recording in the case of the
      free-living dataset).
    - Matched analysis (originally referred to as "True Positive Evaluation"): Estimated walking bouts (WB) are matched
      with the reference WB (overlap: 80%). Average reference and estimated DMO values are calculated on a per-WB level.
      As a result, a pair of reference and estimated values is returned for each matched WB.

    The following metrics are calculated:

    - The error, absolute error, relative error, and absolute relative error of DMO estimation for each matched WB
      (``matched_parameters_with_errors``). These are returned as a dataframe wrapped in ``no_agg``. The dataframe also
      contains the average walking speed for each WB extracted from the reference system to provide context for further
      analysis.
    - The average matched WB-level error metrics on a per-datapoint level. These are returned as
      ``matched__<DMO>__<metric>`` and will be averaged over all datapoints in the Scorer.
    - The median error metrics on a per-datapoint level. These are returned as ``combined__<DMO>__<metric>`` and will be
      averaged over all datapoints in the Scorer.

    Parameters
    ----------
    pipeline
        An instance of :class:`~mobgab.pipeline.base.BaseMobilisedPipeline` that provides at least `per_wb_parameters_`
        as result.
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

    Notes
    -----
    The aggregated analysis takes the median of a DMO per recording (gait test in lab, full recording in free-living)
    and then calculates the error metrics based on the median values.
    This results in a single "row" of metrics per recording.

    The "matched" or "true positive" analysis is based on the WB level.
    We first categorize the WBs into true positives, false positives, and false negatives based on the overlap of the
    detected and reference WBs.
    A bidirectional overlap of 80% is considered a match.
    For the matched WBs, we calculate the error metrics based on the detected and reference values.
    The errors are then provided on two granularities:
    - The error dataframe where each row corresponds to a matched WB.
    - The aggregated error metrics based on the mean of all errors of all matched WBs in a recording.

    """
    pipeline.safe_run(datapoint)
    # Extracting main results (average values per WB)
    # We don't drop NaNs here, as we want to keep Cadence values even if other values are missing
    calculated_per_wb = pipeline.per_wb_parameters_[
        ["start", "end", "walking_speed_mps", "stride_length_m", "cadence_spm"]
    ]
    reference_per_wb = datapoint.reference_parameters_.wb_list[
        ["start", "end", "avg_walking_speed_mps", "avg_stride_length_m", "avg_cadence_spm"]
    ]
    reference_per_wb.columns = reference_per_wb.columns.str.removeprefix("avg_")

    # Aggregate analysis (Combined Evaluation)
    median_parameters = (
        pd.concat({"detected": calculated_per_wb.median(), "reference": reference_per_wb.median()})
        .to_frame()
        .swaplevel()
        .T
    )
    median_parameters_errors = apply_transformations(median_parameters, _errors)
    median_parameters_with_errors = pd.concat([median_parameters_errors, median_parameters], axis=1)
    median_parameters_with_errors.columns = ["__".join(levels) for levels in median_parameters_with_errors.columns]
    assert len(median_parameters_with_errors) == 1
    median_parameters_with_errors = median_parameters_with_errors.add_prefix("combined__").iloc[0]

    # Matched analysis (True positive Evaluation)
    wb_tp_fp_fn = categorize_intervals(
        gsd_list_detected=calculated_per_wb,
        gsd_list_reference=reference_per_wb,
        overlap_threshold=0.8,
        multiindex_warning=False,
    )
    wb_matches = get_matching_intervals(
        metrics_detected=calculated_per_wb,
        metrics_reference=reference_per_wb,
        matches=wb_tp_fp_fn,
    )

    matched_errors = apply_transformations(wb_matches, _errors)
    matched_parameters_with_errors = pd.concat([matched_errors, wb_matches], axis=1)
    # Compared to the median values, we have multiple values per datapoint here.
    # Each matched WB has its own row.
    matched_parameters_with_errors.columns = ["__".join(levels) for levels in matched_parameters_with_errors.columns]

    # We calculate the mean error across all errors of all WBs
    matched_parameters_with_errors_agg = matched_parameters_with_errors.mean()
    matched_parameters_with_errors_agg["n_matched_wbs"] = len(matched_parameters_with_errors)
    matched_parameters_with_errors_agg = matched_parameters_with_errors_agg.add_prefix("matched__")

    return {
        **median_parameters_with_errors.to_dict(),
        **matched_parameters_with_errors_agg.to_dict(),
        # We also pass the raw matched parameters to allow for detailed analysis on the WB level.
        "matched_error": no_agg(matched_parameters_with_errors),
    }


def pipeline_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    _: BaseMobilisedPipeline,
    dataset: BaseGaitDatasetWithReference,
) -> tuple[dict[str, any], dict[str, list[any]]]:
    """Aggregate the results of the pipeline evaluation.

    .. warning:: This function is not meant to be called directly, but as ``final_aggregator`` in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function aggregates the performance metrics as follows:

    - The raw per-datapoint level values are combined into one dataframe each across the entire dataset.
    - All other values are passed through unchanged.

    """
    data_labels = [d.group_label for d in dataset]
    data_label_names = data_labels[0]._fields

    matched_errors = single_results.pop("matched_error")
    matched_errors = pd.concat(matched_errors, keys=data_labels, names=data_label_names, axis=0)

    aggregated_single_results = {
        "raw__matched_errors": matched_errors,
    }

    return agg_results, {**single_results, **aggregated_single_results}


#: :data:: pipeline_score
#: Scorer class instance for the Mobilise-D pipeline.
pipeline_score = Scorer(pipeline_per_datapoint_score, final_aggregator=pipeline_final_agg)
pipeline_score.__doc__ = """Scorer for the Mobilise-D pipeline evaluation.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using the :func:`pipeline_per_datapoint_score` function
as per-datapoint scorer and the :func:`pipeline_final_agg` function as final aggregator.
For more information about Scorer, head to the tpcp documentation (:class:`~tpcp.validate.Scorer`).
For usage information in the context of mobgap, have a look at the :ref:`evaluation example <sl_evaluation>` for GSD.

The following metrics are calculated:

Raw metrics (part of the single results):

- ``single__raw__matched_errors``: A dataframe containing the WB level values and errors for each WB
  in the dataset.
  In addition, to the DMO values and errors, the dataframe also contains the average walking speed for each WB
  extracted from the reference system to provide context for further analysis.

Metrics per datapoint (single results):
*These values are all provided as a list of values, one per datapoint.*

- ``combined__<DMO>__{metric}``: The per-datapoint WB-level metrics calculated from median reference and estimated
  values.
- ``matched__<DMO>__{metric}``: The average over the per-WB-averaged matched WB-level metrics of each datapoint.

Aggregated metrics (agg results):

- ``combined__<DMO>__{metric}``: The average over the per-datapoint-averaged aggregate WB-level metrics.
- ``matched__<DMO>__{metric}``: The average over the per-datapoint-averaged matched WB-level metrics.
"""


__all__ = [
    "CustomErrorAggregations",
    "ErrorTransformFuncs",
    "abs_error",
    "abs_rel_error",
    "categorize_intervals",
    "categorize_intervals_per_sample",
    "error",
    "get_default_error_aggregations",
    "get_default_error_transformations",
    "get_matching_intervals",
    "icc",
    "loa",
    "pipeline_final_agg",
    "pipeline_per_datapoint_score",
    "pipeline_score",
    "quantiles",
    "rel_error",
]
