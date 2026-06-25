"""Evaluation and scoring helpers for wear-time detection pipelines."""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from tpcp.validate import Scorer, no_agg

from mobgap.data.base import BaseGaitDataset
from mobgap.gait_sequences.evaluation import (
    calculate_matched_gsd_performance_metrics,
    categorize_intervals_per_sample,
)
from mobgap.utils.evaluation import count_samples_in_intervals
from mobgap.weartime.pipeline import WtdEmulationPipeline
from mobgap.weartime.utils import clip_intervals_to_waking_hours


def _empty_weartime_df(index_name: str = "wt_id") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.Series(dtype="int64"),
            "end": pd.Series(dtype="int64"),
        }
    ).rename_axis(index_name)


def _only_start_end(intervals: pd.DataFrame, *, index_name: str = "wt_id") -> pd.DataFrame:
    if intervals.empty:
        return _empty_weartime_df(index_name)
    return intervals[["start", "end"]].astype({"start": "int64", "end": "int64"})


def _duration_metrics(
    *,
    reference_weartime: pd.DataFrame,
    detected_weartime_min: float,
    sampling_rate_hz: float,
    prefix: str = "",
) -> dict[str, float]:
    # Keep duration accounting aligned with the GSD evaluation helpers used for TP/FP/FN/TN. These helpers count both
    # interval bounds for the reference intervals, so the reference minute values intentionally follow that convention.
    reference_weartime_min = count_samples_in_intervals(reference_weartime) / (sampling_rate_hz * 60)
    return {
        f"{prefix}reference_weartime_min": reference_weartime_min,
        f"{prefix}detected_weartime_min": detected_weartime_min,
        f"{prefix}weartime_error_min": detected_weartime_min - reference_weartime_min,
    }


def _get_waking_hours_min(pipeline: WtdEmulationPipeline) -> tuple[int, int]:
    return pipeline.algo_.waking_hours_min


def wtd_per_datapoint_score(
    pipeline: WtdEmulationPipeline,
    datapoint: BaseGaitDataset,
    *,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> dict[str, Any]:
    """Evaluate a wear-time detector on a single datapoint.

    This scorer intentionally uses the existing gait-sequence interval helpers for interval matching and sample
    counting. The confusion-matrix counts are returned in samples. Overall wear-time duration metrics are returned in
    minutes as floats.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Zero division", category=UserWarning)
        warnings.filterwarnings("ignore", message="`matches_df` does not contain `tn` matches", category=UserWarning)

        pipeline.safe_run(datapoint)

        detected_weartime = _only_start_end(pipeline.weartime_list_)
        reference_weartime = _only_start_end(datapoint.reference_weartime_, index_name="weartime_id")
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz
        waking_hours_min = _get_waking_hours_min(pipeline)

        matches = categorize_intervals_per_sample(
            gsd_list_detected=detected_weartime,
            gsd_list_reference=reference_weartime,
            n_overall_samples=len(data),
        )

        reference_waking_weartime = clip_intervals_to_waking_hours(
            reference_weartime, data=data, sampling_rate_hz=sampling_rate_hz, waking_hours_min=waking_hours_min
        )
        detected_weartime_min = pipeline.total_weartime_min_
        detected_waking_weartime_min = pipeline.total_weartime_during_waking_min_

        return {
            **calculate_matched_gsd_performance_metrics(matches, zero_division=zero_division),
            **_duration_metrics(
                reference_weartime=reference_weartime,
                detected_weartime_min=detected_weartime_min,
                sampling_rate_hz=sampling_rate_hz,
            ),
            **_duration_metrics(
                reference_weartime=reference_waking_weartime,
                detected_weartime_min=detected_waking_weartime_min,
                sampling_rate_hz=sampling_rate_hz,
                prefix="waking_",
            ),
            "matches": no_agg(matches),
            "detected": no_agg(detected_weartime),
            "reference": no_agg(reference_weartime),
            "reference_waking": no_agg(reference_waking_weartime),
            "sampling_rate_hz": no_agg(sampling_rate_hz),
            "runtime_s": getattr(pipeline.algo_, "perf_", {}).get("runtime_s", np.nan),
        }


def wtd_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    pipeline: WtdEmulationPipeline,  # noqa: ARG001
    dataset: BaseGaitDataset,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Aggregate wear-time scoring results over multiple datapoints."""
    data_labels = [d.group_label for d in dataset]
    data_label_names = data_labels[0]._fields

    matches = single_results.pop("matches")
    matches = pd.concat(matches, keys=data_labels, names=[*data_label_names, *matches[0].index.names])
    detected = single_results.pop("detected")
    detected = pd.concat(detected, keys=data_labels, names=[*data_label_names, *detected[0].index.names])
    reference = single_results.pop("reference")
    reference = pd.concat(reference, keys=data_labels, names=[*data_label_names, *reference[0].index.names])
    reference_waking = single_results.pop("reference_waking")
    reference_waking = pd.concat(
        reference_waking, keys=data_labels, names=[*data_label_names, *reference_waking[0].index.names]
    )

    sampling_rate_hz = single_results.pop("sampling_rate_hz")
    if set(sampling_rate_hz) != {sampling_rate_hz[0]}:
        raise ValueError(
            "Sampling rate is not the same for all datapoints in the dataset. "
            "This is not supported by this scorer. "
            "Provide a custom scorer that can handle this case."
        )

    combined_matched = {f"combined__{k}": v for k, v in calculate_matched_gsd_performance_metrics(matches).items()}
    combined_duration = {
        f"combined__{k}": v
        for k, v in _duration_metrics(
            reference_weartime=reference,
            detected_weartime_min=sum(single_results["detected_weartime_min"]),
            sampling_rate_hz=sampling_rate_hz[0],
        ).items()
    }
    combined_waking_duration = {
        f"combined__{k}": v
        for k, v in _duration_metrics(
            reference_weartime=reference_waking,
            detected_weartime_min=sum(single_results["waking_detected_weartime_min"]),
            sampling_rate_hz=sampling_rate_hz[0],
            prefix="waking_",
        ).items()
    }

    aggregated_single_results = {
        "raw__matches": matches,
        "raw__detected": detected,
        "raw__reference": reference,
        "raw__reference_waking": reference_waking,
    }

    return (
        {**agg_results, **combined_matched, **combined_duration, **combined_waking_duration},
        {**single_results, **aggregated_single_results},
    )


wtd_score = Scorer(wtd_per_datapoint_score, final_aggregator=wtd_final_agg)
wtd_score.__doc__ = """Scorer for wear-time detection algorithms.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using :func:`wtd_per_datapoint_score` as
per-datapoint scorer and :func:`wtd_final_agg` as final aggregator.
"""


__all__ = ["wtd_final_agg", "wtd_per_datapoint_score", "wtd_score"]
