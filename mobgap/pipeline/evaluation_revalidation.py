"""Evaluation tools and functions for the Mobilise-D pipeline for the estimation of walking speed."""

import pandas as pd
from pandas.testing import assert_frame_equal
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.pipeline import MobilisedPipelineHealthy, MobilisedPipelineImpaired, MobilisedPipelineUniversal
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals, ErrorTransformFuncs as E  # noqa: N814
from mobgap.stride_length.pipeline import SlEmulationPipeline
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby

_errors = [("walking_speed_mps", [E.error, E.abs_error, E.rel_error, E.abs_rel_error])]
_error_aggregates = []


def pipeline_per_datapoint_score(pipeline: MobilisedPipelineUniversal, datapoint: BaseGaitDatasetWithReference) -> dict: #TODO: correct type for argument pipeline?
    """Evaluate the performance of the Mobilise-D pipeline for walking speed estimation on a single datapoint.

    .. warning:: This function is not meant to be called directly, but as a scoring function in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function calculates the median walking speed on a per datapoint level (per trial/recording).

    The following metrics are calculated:

    - The error, absolute error, relative error, and absolute relative error for each stride
      (``stride_level_values_with_errors``). These are returned as a dataframe wrapped in ``no_agg``.
    - The average stride-level error metrics on a per-data-point level. These are returned as ``stride__<metric>`` and
      will be averaged over all datapoints in the Scorer.
    - The error, absolute error, relative error, and absolute relative error for each WB. The WB-level metrics are
      calculated as the average stride-level stride-length for each WB in the algorithm output.
      For the reference system, the average stride length are taken directly from the wb-level reference data.
      (``wb_level_values_with_errors``). These are returned as a dataframe wrapped in ``no_agg``.
      The dataframe also contains the average walking speed for each WB extracted from the reference system to provide
      context for further analysis.
    - The average WB-level error metrics on a per-data-point level. These are returned as ``wb__<metric>`` and
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

    # Here we calculate the median of the average walking speed per WB (as provided by the pipeline and the ground truth)
    calculated_walking_speed = pipeline.per_wb_parameters_["walking_speed_mps"].dropna().median() #TODO: correct attribute?
    reference_walking_speed = datapoint.reference_parameters_.wb_list["avg_walking_speed_mps"].rename(
        columns={"avg_walking_speed_mps": "walking_speed_mps"}
    ).dropna().median()

    # Here we concatenate into the same dataframe
    combined_ws_wb_level = pd.concat(
        {"detected": calculated_walking_speed[["walking_speed_mps"]], "reference": reference_walking_speed[["walking_speed_mps"]]},
        names=["source"],
        axis=1,
    )
    # -------------------------------------------------------------------------------------------------------------------
    # Stride length
    # -------------------------------------------------------------------------------------------------------------------

    # Here we calculate the median of the average stride length per WB (as provided by the pipeline and the ground truth)
    calculated_stride_length = pipeline.per_wb_parameters_[
        "stride_length_m"].dropna().median()  # TODO: correct attribute?
    reference_stride_length = datapoint.reference_parameters_.wb_list["avg_walking_speed_mps"].rename(
        columns={"avg_stride_length_m": "stride_length_m"}
    ).dropna().median()

    # Here we concatenate into the same dataframe
    combined_sl_wb_level = pd.concat(
        {"detected": calculated_stride_length[["stride_length_m"]],
         "reference": reference_stride_length[["stride_length_m"]]},
        names=["source"],
        axis=1,
    )
    # -------------------------------------------------------------------------------------------------------------------
    # Cadence
    # -------------------------------------------------------------------------------------------------------------------

    # Here we calculate the median of the average cadence per WB (as provided by the pipeline and the ground truth)
    calculated_cadence = pipeline.per_wb_parameters_[
        "cadence_spm"].dropna().median()  # TODO: correct attribute?
    reference_cadence = datapoint.reference_parameters_.wb_list["avg_walking_speed_mps"].rename(
        columns={"avg_cadence_spm": "cadence_spm"}
    ).dropna().median()

    # Here we concatenate into the same dataframe
    combined_cad_wb_level = pd.concat(
        {"detected": calculated_cadence[["cadence_spm"]],
         "reference": reference_cadence[["cadence_spm"]]},
        names=["source"],
        axis=1,
    )
    return {
        "median_wb_values_walking_speed": no_agg(combined_ws_wb_level),
        "median_wb_values_stride_length": no_agg(combined_sl_wb_level),
        "median_wb_values_cadence": no_agg(combined_cad_wb_level),
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

    - The raw stride and WB level values are combined into one dataframe each across the entire dataset.
    - All other values are passed through unchanged.

    """
    data_labels = [d.group_label for d in dataset]
    data_label_names = data_labels[0]._fields

    stride_level_values_with_errors_list = single_results.pop("median_wb_values_without_errors")
    stride_level_values_with_errors = pd.concat(
        stride_level_values_with_errors_list,
        keys=data_labels,
        names=[*data_label_names, *stride_level_values_with_errors_list[0].index.names],
    )
    wb_level_values_with_errors_list = single_results.pop("wb_level_values_with_errors")
    wb_level_values_with_errors = pd.concat(
        wb_level_values_with_errors_list,
        keys=data_labels,
        names=[*data_label_names, *wb_level_values_with_errors_list[0].index.names],
    )

    aggregated_single_results = {
        "raw__stride_level_values_with_errors": stride_level_values_with_errors,
        "raw__wb_level_values_with_errors": wb_level_values_with_errors,
    }

    return agg_results, {**single_results, **aggregated_single_results}


#: :data:: sl_score
#: Scorer class instance for SL algorithms.
pipeline_score = Scorer(pipeline_per_datapoint_score, final_aggregator=pipeline_final_agg)
pipeline_score.__doc__ = """Scorer for the Mobilise-D pipeline evaluation.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using the :func:`pipeline_per_datapoint_score` function as
per-datapoint scorer and the :func:`pipeline_final_agg` function as final aggregator.
For more information about Scorer, head to the tpcp documentation (:class:`~tpcp.validate.Scorer`).
For usage information in the context of mobgap, have a look at the :ref:`evaluation example <sl_evaluation>` for GSD.

The following metrics are calculated:

Raw metrics (part of the single results):

- ``single__raw__stride_level_values_with_errors``: A dataframe containing the stride level values and errors for each
  stride in the dataset.
- ``single__raw__wb_level_values_with_errors``: A dataframe containing the WB level values and errors for each WB in the
  dataset.
  In addition, to the stride length values and errors, the dataframe also contains the average walking speed for each WB
  extracted from the reference system to provide context for further analysis.

Metrics per datapoint (single results):
*These values are all provided as a list of values, one per datapoint.*

- ``single__stride__{metric}``: The stride level values and errors averaged for each datapoint.
- ``single__wb__{metric}``: The WB level values and errors averaged for each datapoint.

Aggregated metrics (agg results):

- ``agg__stride__{metric}``: The average over the per-datapoint-averaged stride level metrics.
- ``agg__wb__{metric}``: The average over the per-datapoint-averaged WB level metrics.
"""


__all__ = ["pipeline_score", "pipeline_per_datapoint_score", "pipeline_final_agg"]
