"""Evaluation tools and functions for cadence."""

import pandas as pd
from pandas.testing import assert_frame_equal
from tpcp.validate import Scorer, no_agg

from mobgap.cadence.pipeline import CadEmulationPipeline
from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_transformations

_errors = [("cadence_spm", [E.error, E.abs_error, E.rel_error, E.abs_rel_error])]


def cad_per_datapoint_score(pipeline: CadEmulationPipeline, datapoint: BaseGaitDatasetWithReference) -> dict:
    """Evaluate the performance of the cadence pipeline on a single datapoint.

    .. warning:: This function is not meant to be called directly, but as a scoring function in a
       :class:`tpcp.validate.Scorer`.
       If you are writing custom scoring functions, you can use this function as a template or wrap it in a new
       function.

    This function calculates the cadence error on a per stride, per WB, and per datapoint level.

    The following metrics are calculated:

    - The error, absolute error, relative error, and absolute relative error for each stride
      (``stride_level_values_with_errors``). These are returned as a dataframe wrapped in ``no_agg``.
    - The average stride-level error metrics on a per-data-point level. These are returned as ``stride__<metric>`` and
      will be averaged over all datapoints in the Scorer.
    - The error, absolute error, relative error, and absolute relative error for each WB. The WB-level metrics are
      calculated as the average stride-level cadence for each WB in the algorithm output.
      For the reference system, the average cadence are taken directly from the wb-level reference data.
      (``wb_level_values_with_errors``). These are returned as a dataframe wrapped in ``no_agg``.
      The dataframe also contains the average walking speed for each WB extracted from the reference system to provide
      context for further analysis.
    - The average WB-level error metrics on a per-data-point level. These are returned as ``wb__<metric>`` and
      will be averaged over all datapoints in the Scorer.

    Parameters
    ----------
    pipeline
        An instance of :class:`~mobgab.cadence.pipeline.CadEmulationPipeline` that wraps the algorithm that should
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

    # The CadEmulation pipeline calculates the values for the exact same strides that are also in the reference data.
    # This means, we can calculate errors on the stride level and on the wb/gs level.
    ref_paras = datapoint.reference_parameters_
    ref_strides = ref_paras.stride_parameters
    ref_strides = ref_strides[~ref_strides.duration_s.isna()][["start", "end", "duration_s"]].rename(
        columns={"duration_s": "cadence_spm"}
    )
    # convert the duration values to stride per minute (gives reference cadence per stride)
    ref_strides["cadence_spm"] = (1 / ref_strides["cadence_spm"]) * 120

    # calculated cadence per stride
    calculated_cadence = pipeline.cadence_per_stride_

    # We check that we actually calculated the cadence for the same strides as in the reference data
    assert_frame_equal(ref_strides[["start", "end"]], calculated_cadence[["start", "end"]])

    combined_stride_level = pd.concat(
        {"detected": calculated_cadence[["cadence_spm"]], "reference": ref_strides[["cadence_spm"]]},
        names=["source"],
        axis=1,
    ).swaplevel(axis=1)

    # For the WB level, we calculate the average cadence for the algorithm, but we take the average values
    # provided by the reference system directly instead of recalculating them.
    # Because we are later interested in the dependency of the error on the gait speed, we also add this value from the
    # reference system.
    combined_wb_level = combined_stride_level.groupby("wb_id")[[("cadence_spm", "detected")]].mean()
    combined_wb_level[("cadence_spm", "reference")] = ref_paras.wb_list["avg_cadence_spm"]

    stride_level_errors = apply_transformations(combined_stride_level, _errors)
    stride_level_values_with_errors = pd.concat([stride_level_errors, combined_stride_level], axis=1)
    wb_level_errors = apply_transformations(combined_wb_level, _errors)
    wb_level_values_with_errors = pd.concat([wb_level_errors, combined_wb_level], axis=1)

    stride_level_performance_metrics = (
        stride_level_values_with_errors.mean()["cadence_spm"].add_prefix("stride__").to_dict()
    )
    wb_level_errors = wb_level_values_with_errors.mean()["cadence_spm"].add_prefix("wb__").to_dict()

    no_agg_wb_level_values_with_errors = wb_level_values_with_errors["cadence_spm"].assign(
        reference_ws=ref_paras.wb_list["avg_walking_speed_mps"]
    )

    return {
        **stride_level_performance_metrics,
        **wb_level_errors,
        "stride_level_values_with_errors": no_agg(stride_level_values_with_errors["cadence_spm"]),
        "wb_level_values_with_errors": no_agg(no_agg_wb_level_values_with_errors),
    }


def cad_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    _: CadEmulationPipeline,
    dataset: BaseGaitDatasetWithReference,
) -> tuple[dict[str, any], dict[str, list[any]]]:
    """Aggregate the results of the cadence evaluation.

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

    stride_level_values_with_errors_list = single_results.pop("stride_level_values_with_errors")
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


#: :data:: cad_score
#: Scorer class instance for Cadence algorithms.
cad_score = Scorer(cad_per_datapoint_score, final_aggregator=cad_final_agg)
cad_score.__doc__ = """Scorer for cadence algorithms.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using the :func:`cad_per_datapoint_score` function as
per-datapoint scorer and the :func:`cad_final_agg` function as final aggregator.
For more information about Scorer, head to the tpcp documentation (:class:`~tpcp.validate.Scorer`).
For usage information in the context of mobgap, have a look at the :ref:`evaluation example <cad_evaluation>` for GSD.

The following metrics are calculated:

Raw metrics (part of the single results):

- ``single__raw__stride_level_values_with_errors``: A dataframe containing the stride level values and errors for each
  stride in the dataset.
- ``single__raw__wb_level_values_with_errors``: A dataframe containing the WB level values and errors for each WB in the
  dataset.
  In addition, to the cadence values and errors, the dataframe also contains the average walking speed for each WB
  extracted from the reference system to provide context for further analysis.

Metrics per datapoint (single results):
*These values are all provided as a list of values, one per datapoint.*

- ``single__stride__{metric}``: The stride level values and errors averaged for each datapoint.
- ``single__wb__{metric}``: The WB level values and errors averaged for each datapoint.

Aggregated metrics (agg results):

- ``agg__stride__{metric}``: The average over the per-datapoint-averaged stride level metrics.
- ``agg__wb__{metric}``: The average over the per-datapoint-averaged WB level metrics.
"""


__all__ = ["cad_final_agg", "cad_per_datapoint_score", "cad_score"]
