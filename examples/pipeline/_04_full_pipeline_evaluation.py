from typing import Sequence, Callable, Protocol, Union, TypeAlias, Any

import pandas as pd
from tpcp import Dataset
from tpcp.validate import Aggregator
from typing_extensions import Unpack

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.pipeline import GenericMobilisedPipeline
from mobgap.pipeline.evaluation import (
    get_default_error_aggregations,
    get_default_error_transformations,
)
from mobgap.utils.df_operations import apply_aggregations, apply_transformations

class AggFunc(Protocol):
    def __call__(self, *args: Unpack[pd.DataFrame], **kwargs: Unpack[dict[str, Any]]) -> dict[str, float]: ...


DelayedAggInput: TypeAlias = Union[pd.DataFrame], Sequence[pd.DataFrame, pd.DataFrame]

def _list_of_dfs_to_df(dfs: Sequence[pd.DataFrame], datapoints: Sequence[Dataset]) -> pd.DataFrame:
    return pd.concat({k.group_label: v for k, v in zip(datapoints, dfs)}, axis=1, names=[*datapoints[0].group_label.fields, dfs[0].index.names])

class DelayedDfAggregator(Aggregator[Union[pd.DataFrame], Sequence[pd.DataFrame, pd.DataFrame]]):

    def __init__(self, func: AggFunc, *, other_kwargs: dict[str, Any], return_raw_scores: bool = True) -> None:
        self.other_kwargs = other_kwargs
        self.func = func
        super().__init__(return_raw_scores=return_raw_scores)

    def aggregate(self, /, values: Sequence[DelayedAggInput], datapoints: Sequence[Dataset]) -> dict[str, float]:
        if isinstance(values, pd.DataFrame):
            return self.func(_list_of_dfs_to_df(values, datapoints), **self.other_kwargs)
        else:
            inverted_list = zip(*values)
            combined_dfs = [_list_of_dfs_to_df(v, datapoints) for v in inverted_list]
            return self.func(*combined_dfs, **self.other_kwargs)





def full_pipeline_evaluation_scorer(
    pipeline: GenericMobilisedPipeline, datapoint: BaseGaitDatasetWithReference
):
    calculated = pipeline.safe_run(datapoint)
    reference = datapoint.reference_parameters_

    # aggregated analysis


def _calc_and_agg_error(
    matched_vals: pd.DataFrame,
    error_transforms: list[tuple] = get_default_error_transformations(),
    error_aggregations: list[tuple] = get_default_error_aggregations(),
):
    wb_errors = apply_transformations(matched_vals, error_transforms)
    matched_vals_with_errors = pd.concat([matched_vals, wb_errors], axis=1)
    agg_results = (
        apply_aggregations(matched_vals_with_errors, error_aggregations)
        .rename_axis(index=["aggregation", "metric", "origin"])
        .reorder_levels(["metric", "origin", "aggregation"])
        .sort_index(level=0)
        .to_frame("values")
    )
    return agg_results


def agg_errors(
    detected: pd.DataFrame,
    reference: pd.DataFrame,
    agg_levels: list[str],
    error_transforms: list[tuple] = get_default_error_transformations(),
    error_aggregations: list[tuple] = get_default_error_aggregations(),
):
    combined_dmos = (
        pd.concat([detected, reference], keys=["detected", "reference"], axis=1)
        .reorder_levels((1, 0), axis=1)
        .sort_index(axis=1)
    )

    agg_dmos = combined_dmos.groupby(level=agg_levels).mean().dropna()
    return _calc_and_agg_error(
        agg_dmos,
        error_transforms=error_transforms,
        error_aggregations=error_aggregations,
    )


def matched_errors(detected: pd.DataFrame, reference: pd.DataFrame):
    matched_dmos = (
        pd.concat([detected, reference], keys=["detected", "reference"], axis=1)
        .reorder_levels((1, 0), axis=1)
        .sort_index(axis=1)
    )

    matched_dmos = matched_dmos.dropna()
    return _calc_and_agg_error(matched_dmos)
