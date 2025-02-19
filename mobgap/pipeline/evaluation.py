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
"""
Evaluation tools and functions for the Mobilise-D pipeline for the estimation of primary DMOs: 
- Cadence
- Stride length 
- Walking speed
"""

import pandas as pd
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.pipeline import MobilisedPipelineUniversal
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals, ErrorTransformFuncs as E  # noqa: N814
from mobgap.utils.df_operations import apply_transformations

_errors = [("walking_speed_mps", [E.error, E.abs_error, E.rel_error, E.abs_rel_error])]
           # ("stride_length_m", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
           # ("cadence_spm", [E.error, E.abs_error, E.rel_error, E.abs_rel_error])]
_error_aggregates = []


def pipeline_per_datapoint_score(pipeline: MobilisedPipelineUniversal, datapoint: BaseGaitDatasetWithReference) -> dict: #TODO: correct type for argument pipeline?
    """Evaluate the performance of the Mobilise-D pipeline for primary DMOs estimation on a single datapoint.

    .. warning:: This function is not meant to be called directly, but as a scoring function in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function calculates the median DMO reference and detected values and errors on a per datapoint level (per trial/recording).

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

    # Combined evaluation
    #-------------------------------------------------------------------------------------------------------------------
    # Walking speed
    #-------------------------------------------------------------------------------------------------------------------

    # Here we extract the average walking speed per WB (as provided by the pipeline and the ground truth)
    calculated_walking_speed_raw = pipeline.per_wb_parameters_["walking_speed_mps"].dropna()
    reference_walking_speed_raw = datapoint.reference_parameters_.wb_list["avg_walking_speed_mps"].rename(
        "walking_speed_mps"
    ).dropna()

    # Here we concatenate into the same dataframe and calculate errors
    combined_ws_wb_level = pd.concat([calculated_walking_speed_raw, reference_walking_speed_raw], axis=1)
    combined_ws_wb_level.columns = pd.MultiIndex.from_tuples([("walking_speed_mps", "detected"), ("walking_speed_mps", "reference")])
    wb_level_ws_errors = apply_transformations(combined_ws_wb_level, _errors)
    wb_level_ws_values_with_errors = pd.concat([wb_level_ws_errors, combined_ws_wb_level], axis=1)

    # Here we calculate the median of the detected and reference walking speed across all WBs in the current datapoint
    median_ws = pd.DataFrame(
    [[calculated_walking_speed_raw.median(), reference_walking_speed_raw.median()]],
    columns=pd.MultiIndex.from_tuples([
        ("walking_speed_mps", "detected"),
        ("walking_speed_mps", "reference")
    ])
    )
    # Here we calculate pre-defined errors for the combined evaluation as reported in Kirk et al., (2024)
    median_ws_errors = apply_transformations(median_ws, _errors)
    # Here we concatenate detected and reference values with errors
    median_ws_values_with_errors = (pd.concat([median_ws_errors, median_ws], axis=1)["walking_speed_mps"]
                                    .add_prefix("combined__")
                                    .iloc[0]
                                    .to_dict())
    # Add information on average reference walking speed (not really necessary when walking speed is the DMO)
    no_agg_wb_level_ws_values_with_errors = wb_level_ws_values_with_errors["walking_speed_mps"].assign(
        reference_ws=reference_walking_speed_raw
    )
    # # -------------------------------------------------------------------------------------------------------------------
    # # Stride length
    # # -------------------------------------------------------------------------------------------------------------------
    #
    # # Here we extract the average stride length per WB (as provided by the pipeline and the ground truth)
    # calculated_stride_length_raw = pipeline.per_wb_parameters_["stride_length_m"].dropna()
    # reference_stride_length_raw = datapoint.reference_parameters_.wb_list["avg_stride_length_m"].rename(
    #     "stride_length_m"
    # ).dropna()
    #
    # # Here we concatenate into the same dataframe and calculate errors
    # combined_sl_wb_level = pd.concat([calculated_stride_length_raw, reference_stride_length_raw], axis=1)
    # combined_sl_wb_level.columns = pd.MultiIndex.from_tuples(
    #     [("stride_length_m", "detected"), ("stride_length_m", "reference")])
    # wb_level_sl_errors = apply_transformations(combined_sl_wb_level, _errors)
    # wb_level_sl_values_with_errors = pd.concat([wb_level_sl_errors, combined_sl_wb_level], axis=1)
    #
    # # Here we calculate the median of the detected and reference stride length across all WBs in the current datapoint
    # median_sl = pd.DataFrame(
    #     [[calculated_stride_length_raw.median(), reference_stride_length_raw.median()]],
    #     columns=pd.MultiIndex.from_tuples([
    #         ("stride_length_m", "detected"),
    #         ("stride_length_m", "reference")
    #     ])
    # )
    # # Here we calculate pre-defined errors for the combined evaluation as reported in Kirk et al., (2024)
    # median_sl_errors = apply_transformations(median_sl, _errors)
    # # Here we concatenate detected and reference values with errors
    # median_sl_values_with_errors = (pd.concat([median_sl_errors, median_sl], axis=1)["stride_length_m"]
    #                                 .add_prefix("combined__").
    #                                 to_dict())
    # # Add information on average reference stride length
    # no_agg_wb_level_sl_values_with_errors = wb_level_sl_values_with_errors["stride_length_m"].assign(
    #     reference_ws=reference_walking_speed_raw
    # )
    # # -------------------------------------------------------------------------------------------------------------------
    # # Cadence
    # # -------------------------------------------------------------------------------------------------------------------
    #
    # # Here we extract the average cadence per WB (as provided by the pipeline and the ground truth)
    # calculated_cadence_raw = pipeline.per_wb_parameters_["cadence_spm"].dropna()
    # reference_cadence_raw = datapoint.reference_parameters_.wb_list["avg_cadence_spm"].rename(
    #     "cadence_spm"
    # ).dropna()
    #
    # # Here we concatenate into the same dataframe and calculate errors
    # combined_cad_wb_level = pd.concat([calculated_cadence_raw, reference_cadence_raw], axis=1)
    # combined_cad_wb_level.columns = pd.MultiIndex.from_tuples(
    #     [("cadence_spm", "detected"), ("cadence_spm", "reference")])
    # wb_level_cad_errors = apply_transformations(combined_cad_wb_level, _errors)
    # wb_level_cad_values_with_errors = pd.concat([wb_level_cad_errors, combined_cad_wb_level], axis=1)
    #
    # # Here we calculate the median of the detected and reference cadence across all WBs in the current datapoint
    # median_cad = pd.DataFrame(
    #     [[calculated_cadence_raw.median(), reference_cadence_raw.median()]],
    #     columns=pd.MultiIndex.from_tuples([
    #         ("cadence_spm", "detected"),
    #         ("cadence_spm", "reference")
    #     ])
    # )
    # # Here we calculate pre-defined errors for the combined evaluation as reported in Kirk et al., (2024)
    # median_cad_errors = apply_transformations(median_cad, _errors)
    # # Here we concatenate detected and reference values with errors
    # median_cad_values_with_errors = (pd.concat([median_cad_errors, median_cad], axis=1)["stride_length_m"]
    #                                 .add_prefix("combined__").
    #                                 to_dict())
    # # Add information on average reference cadence
    # no_agg_wb_level_cad_values_with_errors = wb_level_cad_values_with_errors["cadence_spm"].assign(
    #     reference_ws=reference_walking_speed_raw
    # )
    return {
        **median_ws_values_with_errors,
        # **median_sl_values_with_errors,
        # **median_cad_values_with_errors,
        "wb_level_ws_values_with_errors": no_agg(no_agg_wb_level_ws_values_with_errors),
        # "wb_level_sl_values_with_errors": no_agg(no_agg_wb_level_sl_values_with_errors),
        # "wb_level_cad_values_with_errors": no_agg(no_agg_wb_level_sl_values_with_errors),
    }


def pipeline_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    _: MobilisedPipelineUniversal, # TODO: is it the correct object type?
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

    wb_level_ws_values_with_errors_list = single_results.pop("wb_level_ws_values_with_errors")
    wb_level_ws_values_with_errors = pd.concat(
        wb_level_ws_values_with_errors_list,
        keys=data_labels,
        names=[*data_label_names, *wb_level_ws_values_with_errors_list[0].index.names],
    )

    aggregated_single_results = {
        "raw__wb_level_ws_values_with_errors": wb_level_ws_values_with_errors,
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


__all__ = ["pipeline_score", "pipeline_per_datapoint_score", "pipeline_final_agg"]
