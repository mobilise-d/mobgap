"""Known-sample accuracy for inner wear-time validation, without report tables."""

from typing import Any

import numpy as np
import pandas as pd
from tpcp.validate import FloatAggregator, Scorer, no_agg

from mobgap.data.base import BaseGaitDataset
from mobgap.weartime.pipeline import WtdEmulationPipeline


def _known_sample_counts(detected: pd.DataFrame, reference: pd.DataFrame, n_samples: int) -> tuple[int, int]:
    # Sweep half-open endpoints. Tied endpoints have zero distance, so only the
    # state after all events at a boundary contributes to the next segment.
    positions = [0, n_samples]
    changes = [[0, 0, 0], [0, 0, 0]]
    for column, intervals in enumerate(
        (detected, reference.loc[reference["label"] == "wear"], reference.loc[reference["label"] == "uncertain"])
    ):
        for start, end in intervals[["start", "end"]].itertuples(index=False, name=None):
            positions.extend((start, end))
            change = [0, 0, 0]
            change[column] = 1
            changes.extend((change, [-value for value in change]))
    order = np.argsort(positions)
    lengths = np.diff(np.asarray(positions)[order])
    active = np.cumsum(np.asarray(changes)[order], axis=0)[:-1] > 0
    known = ~active[:, 2]
    correct = known & (active[:, 0] == active[:, 1])
    return int(lengths[correct].sum()), int(lengths[known].sum())


def _accuracy_per_datapoint(pipeline: WtdEmulationPipeline, datapoint: BaseGaitDataset) -> dict[str, Any]:
    pipeline.safe_run(datapoint)
    correct, known = _known_sample_counts(
        pipeline.weartime_list_, datapoint.reference_weartime_, len(datapoint.data_ss)
    )
    return {"correct_samples": correct, "known_samples": known, "sampling_rate_hz": no_agg(datapoint.sampling_rate_hz)}


def _aggregate_accuracy(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    pipeline: WtdEmulationPipeline,  # noqa: ARG001
    dataset: BaseGaitDataset,  # noqa: ARG001
) -> tuple[dict[str, Any], dict[str, Any]]:
    sampling_rates = single_results.pop("sampling_rate_hz")
    if set(sampling_rates) != {sampling_rates[0]}:
        raise ValueError("Sampling rate is not the same for all datapoints in the dataset.")
    known = agg_results["known_samples"]
    return {"combined__accuracy": agg_results["correct_samples"] / known if known else np.nan}, single_results


_wtd_accuracy_score = Scorer(
    _accuracy_per_datapoint, final_aggregator=_aggregate_accuracy, default_aggregator=FloatAggregator(np.sum)
)
