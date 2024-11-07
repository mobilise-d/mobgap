from collections.abc import Sequence

import pandas as pd
from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.gait_sequences.evaluation import (
    categorize_intervals,
    get_matching_intervals,
)
from mobgap.pipeline import GenericMobilisedPipeline
from mobgap.pipeline.evaluation import (
    get_default_error_aggregations,
    get_default_error_transformations,
)
from mobgap.utils.df_operations import (
    apply_aggregations,
    apply_transformations,
)
from tpcp import Dataset
from tpcp.validate import no_agg


def _list_of_dfs_to_df(
    dfs: Sequence[pd.DataFrame], dataset: Dataset
) -> pd.DataFrame:
    return pd.concat(
        {k.group_label: v for k, v in zip(dataset, dfs)},
        axis=1,
        names=[*dataset[0].group_label.fields, dfs[0].index.names],
    )


def _calc_and_agg_error(
    matched_vals: pd.DataFrame,
    error_transforms: list[tuple],
    error_aggregations: list[tuple],
):
    wb_errors = apply_transformations(matched_vals, error_transforms)
    matched_vals_with_errors = pd.concat([matched_vals, wb_errors], axis=1)
    agg_results = (
        apply_aggregations(matched_vals_with_errors, error_aggregations)
        .rename_axis(index=["aggregation", "metric", "origin"])
        .reorder_levels(["metric", "origin", "aggregation"])
        .sort_index(level=0)
        .to_dict()
    )
    return agg_results


def full_pipeline_per_datapoint_score(
    pipeline: GenericMobilisedPipeline,
    datapoint: BaseGaitDatasetWithReference,
    agg_levels: list[str],
    agg_name_prefix: str,
):
    detected = pipeline.safe_run(datapoint).per_wb_parameters_
    reference = datapoint.reference_parameters_.wb_list

    agg_matches = (
        pd.concat([detected, reference], keys=["detected", "reference"], axis=1)
        .reorder_levels((1, 0), axis=1)
        .sort_index(axis=1)
        .groupby(level=agg_levels)
        .mean()
        .dropna()
    )

    wb_tp_fp_fn = categorize_intervals(
        gsd_list_detected=detected,
        gsd_list_reference=reference,
        overlap_threshold=0.8,
        multiindex_warning=False,
    )
    tp_matches = get_matching_intervals(
        metrics_detected=detected,
        metrics_reference=reference,
        matches=wb_tp_fp_fn,
    )

    return {
        **{
            f"{agg_name_prefix}__{k}": v
            for k, v in _calc_and_agg_error(
                agg_matches,
                get_default_error_transformations(),
                get_default_error_aggregations(),
            ).items()
        },
        **{
            f"tp__{k}": v
            for k, v in _calc_and_agg_error(
                tp_matches,
                get_default_error_transformations(),
                get_default_error_aggregations(),
            ).items()
        },
        "agg_matches": no_agg(agg_matches),
        "tp_matches": no_agg(tp_matches),
    }


def full_pipeline_final_agg(
    agg_results: dict[str, float],
    raw_results: dict[str, list],
    pipeline: GenericMobilisedPipeline,
    dataset: BaseGaitDatasetWithReference,
):
    tp_matches = _list_of_dfs_to_df(raw_results["tp_matches"], dataset)
