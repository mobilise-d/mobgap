import pandas as pd
from statsmodels.compat.pandas import assert_frame_equal
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.stride_length.pipeline import SlEmulationPipeline
from mobgap.utils.df_operations import apply_transformations

_errors = [("stride_length_m", [E.error, E.abs_error, E.rel_error, E.abs_rel_error])]
_error_aggregates = []


def sl_per_datapoint_score(pipeline: SlEmulationPipeline, datapoint: BaseGaitDatasetWithReference):
    pipeline.safe_run(datapoint)

    # The SlEmulation pipeline calculates the values for the exact same strides that are also in the reference data.
    # This means, we can calculate errors on the stride level and on the wb/gs level.
    ref_strides = datapoint.reference_parameters_.stride_parameters
    ref_strides = ref_strides[~ref_strides.length_m.isna()][["start", "end", "length_m"]].rename(
        columns={"length_m": "stride_length_m"}
    )

    calculated_stride_length = pipeline.stride_length_per_stride_

    # We check that we actually calculated the stride length for the same strides as in the reference data
    assert_frame_equal(ref_strides[["start", "end"]], calculated_stride_length[["start", "end"]])

    combined_stride_level = pd.concat(
        {"detected": calculated_stride_length[["stride_length_m"]], "reference": ref_strides[["stride_length_m"]]},
        names=["source"],
        axis=1,
    ).swaplevel(axis=1)
    combined_wb_level = combined_stride_level.groupby("wb_id").mean()

    stride_level_errors = apply_transformations(combined_stride_level, _errors)
    stride_level_values_with_errors = pd.concat([stride_level_errors, combined_stride_level], axis=1)
    wb_level_errors = apply_transformations(combined_wb_level, _errors)
    wb_level_values_with_errors = pd.concat([wb_level_errors, combined_wb_level], axis=1)

    stride_level_performance_metrics = stride_level_errors.mean()["stride_length_m"].add_prefix("stride__").to_dict()
    wb_level_errors = wb_level_errors.mean()["stride_length_m"].add_prefix("wb__").to_dict()

    return {
        **stride_level_performance_metrics,
        **wb_level_errors,
        "stride_level_values_with_errors": no_agg(stride_level_values_with_errors["stride_length_m"]),
        "wb_level_values_with_errors": no_agg(wb_level_values_with_errors["stride_length_m"]),
    }


def sl_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    _: SlEmulationPipeline,
    dataset: BaseGaitDatasetWithReference,
) -> tuple[dict[str, any], dict[str, list[any]]]:
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


sl_score = Scorer(sl_per_datapoint_score, final_aggregator=sl_final_agg)
