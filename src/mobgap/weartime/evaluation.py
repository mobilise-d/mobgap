"""Evaluation and scoring helpers for wear-time detection pipelines."""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import t
from tpcp.validate import FloatAggregator, Scorer, no_agg

from mobgap.data.base import BaseGaitDataset
from mobgap.gait_sequences.evaluation import calculate_matched_gsd_performance_metrics
from mobgap.weartime.pipeline import WtdEmulationPipeline
from mobgap.weartime.utils import clip_intervals_to_waking_hours
from mobgap.weartime.utils._intervals import _only_start_end


def _categorize_weartime_samples(
    detected: pd.DataFrame, reference: pd.DataFrame, n_samples: int, uncertain: pd.DataFrame
) -> pd.DataFrame:
    """Categorize samples and return half-open runs of equal classification."""
    labels = np.zeros(n_samples, dtype=np.uint8)
    for start, end in detected[["start", "end"]].itertuples(index=False):
        labels[start:end] |= 1
    for start, end in reference[["start", "end"]].itertuples(index=False):
        labels[start:end] |= 2
    for start, end in uncertain[["start", "end"]].itertuples(index=False):
        labels[start:end] |= 4

    if n_samples == 0:
        return pd.DataFrame(columns=["start", "end", "match_type"])
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, n_samples]
    match_types = np.array(["tn", "fp", "fn", "tp", "uncertain", "uncertain", "uncertain", "uncertain"])[
        labels[boundaries[:-1]]
    ]
    return pd.DataFrame({"start": boundaries[:-1], "end": boundaries[1:], "match_type": match_types}).query(
        "match_type != 'uncertain'"
    )


def _exclude_uncertain(intervals: pd.DataFrame, uncertain: pd.DataFrame) -> pd.DataFrame:
    """Subtract uncertain samples while retaining each source interval's index value."""
    positions: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for position, (start, end) in enumerate(intervals[["start", "end"]].itertuples(index=False)):
        fragments = [(start, end)]
        for uncertain_start, uncertain_end in uncertain[["start", "end"]].itertuples(index=False):
            fragments = [
                (fragment_start, fragment_end)
                for left, right in fragments
                for fragment_start, fragment_end in (
                    (left, min(right, uncertain_start)),
                    (max(left, uncertain_end), right),
                )
                if fragment_end > fragment_start
            ]
        for fragment_start, fragment_end in fragments:
            positions.append(position)
            starts.append(fragment_start)
            ends.append(fragment_end)
    result = intervals.iloc[positions][["start", "end"]].copy()
    result["start"] = starts
    result["end"] = ends
    return result


def _gsd_metric_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Adapt half-open wear-time runs to the inclusive GSD metric convention."""
    return matches.assign(end=matches["end"] - 1)


def _duration_metrics(
    *,
    reference_weartime: pd.DataFrame,
    detected_weartime_min: float,
    sampling_rate_hz: float,
    prefix: str = "",
) -> dict[str, float]:
    reference_weartime_min = (reference_weartime["end"] - reference_weartime["start"]).sum() / (sampling_rate_hz * 60)
    return {
        f"{prefix}reference_weartime_min": reference_weartime_min,
        f"{prefix}detected_weartime_min": detected_weartime_min,
        f"{prefix}weartime_error_min": detected_weartime_min - reference_weartime_min,
    }


def wtd_per_datapoint_score(
    pipeline: WtdEmulationPipeline,
    datapoint: BaseGaitDataset,
    *,
    zero_division: Literal["warn", 0, 1] = "warn",
) -> dict[str, Any]:
    """Evaluate a wear-time detector on a single datapoint.

    Intervals are half-open. Confusion-matrix counts are returned in samples; wear-time durations are returned in
    minutes.

    Parameters
    ----------
    pipeline : WtdEmulationPipeline
        Pipeline with a detector that provides ``waking_hours`` and wear-time results.
    datapoint : BaseGaitDataset
        Single-day datapoint with ``data_ss``, ``sampling_rate_hz`` and ``reference_weartime_``. The reference is a
        DataFrame with sample-based ``start`` and exclusive ``end`` columns. If the datapoint provides
        ``reference_uncertain_``, those samples are excluded from all scores and duration errors.
    zero_division : {"warn", 0, 1}
        Value passed to classification metrics when a denominator is zero on labeled data. It does not turn
        entirely uncertain datapoints into measured zero scores.

    Returns
    -------
    dict[str, Any]
        Classification metrics and sample counts, wear-time durations in minutes, and runtime in seconds. All
        scalar rates and durations are NaN when no labeled samples remain. Waking durations are NaN when the
        recorded waking window contains only uncertain samples. ``detected`` and ``reference`` retain the original
        interval IDs; ``detected_scored`` and ``reference_scored`` contain the fragments used for scoring after
        uncertain samples are removed. These tables, ``matches``, ``reference_waking``, and ``sampling_rate_hz`` use
        :func:`~tpcp.validate.no_agg` for the final aggregator.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Zero division", category=UserWarning)
        warnings.filterwarnings("ignore", message="`matches_df` does not contain `tn` matches", category=UserWarning)

        pipeline.safe_run(datapoint)

        detected_weartime = _only_start_end(pipeline.weartime_list_)
        reference_weartime = _only_start_end(datapoint.reference_weartime_, index_name="weartime_id")
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz
        waking_hours = pipeline.algo_.waking_hours
        uncertain = getattr(datapoint, "reference_uncertain_", pd.DataFrame(columns=["start", "end"]))

        matches = _categorize_weartime_samples(detected_weartime, reference_weartime, len(data), uncertain)
        if not uncertain.empty:
            detected_scored = _exclude_uncertain(detected_weartime, uncertain)
            reference_scored = _exclude_uncertain(reference_weartime, uncertain)
        else:
            detected_scored = detected_weartime
            reference_scored = reference_weartime

        reference_waking_weartime = clip_intervals_to_waking_hours(
            reference_scored, data=data, sampling_rate_hz=sampling_rate_hz, waking_hours=waking_hours
        )
        if uncertain.empty:
            detected_weartime_min = pipeline.total_weartime_min_
            detected_waking_weartime_min = pipeline.total_weartime_during_waking_min_
        else:
            detected_weartime_min = (detected_scored["end"] - detected_scored["start"]).sum() / (sampling_rate_hz * 60)
            detected_waking = clip_intervals_to_waking_hours(
                detected_scored, data=data, sampling_rate_hz=sampling_rate_hz, waking_hours=waking_hours
            )
            detected_waking_weartime_min = (detected_waking["end"] - detected_waking["start"]).sum() / (
                sampling_rate_hz * 60
            )

        classification = calculate_matched_gsd_performance_metrics(
            _gsd_metric_matches(matches), zero_division=zero_division
        )
        duration = _duration_metrics(
            reference_weartime=reference_scored,
            detected_weartime_min=detected_weartime_min,
            sampling_rate_hz=sampling_rate_hz,
        )
        waking_duration = _duration_metrics(
            reference_weartime=reference_waking_weartime,
            detected_weartime_min=detected_waking_weartime_min,
            sampling_rate_hz=sampling_rate_hz,
            prefix="waking_",
        )
        if matches.empty:
            classification.update({key: np.nan for key in classification if not key.endswith("_samples")})
            duration = dict.fromkeys(duration, np.nan)
            waking_duration = dict.fromkeys(waking_duration, np.nan)
        elif not uncertain.empty:
            recorded_waking = clip_intervals_to_waking_hours(
                pd.DataFrame({"start": [0], "end": [len(data)]}),
                data=data,
                sampling_rate_hz=sampling_rate_hz,
                waking_hours=waking_hours,
            )
            known_waking = clip_intervals_to_waking_hours(
                matches[["start", "end"]], data=data, sampling_rate_hz=sampling_rate_hz, waking_hours=waking_hours
            )
            if not recorded_waking.empty and known_waking.empty:
                waking_duration = dict.fromkeys(waking_duration, np.nan)

        return {
            **classification,
            **duration,
            **waking_duration,
            "matches": no_agg(matches),
            "detected": no_agg(detected_weartime),
            "detected_scored": no_agg(detected_scored),
            "reference": no_agg(reference_weartime),
            "reference_scored": no_agg(reference_scored),
            "reference_waking": no_agg(reference_waking_weartime),
            "sampling_rate_hz": no_agg(sampling_rate_hz),
            "runtime_s": getattr(pipeline.algo_, "perf_", {}).get("runtime_s", np.nan),
        }


def wtd_final_agg(
    agg_results: dict[str, float],
    single_results: dict[str, list],
    pipeline: WtdEmulationPipeline,  # noqa: ARG001
    dataset: BaseGaitDataset,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate wear-time scoring results over multiple datapoints.

    Parameters
    ----------
    agg_results : dict[str, float]
        Values already aggregated by the tpcp scorer.
    single_results : dict[str, list]
        Per-datapoint values from :func:`wtd_per_datapoint_score` in dataset order. This function removes the raw
        interval and sampling-rate entries from the dictionary.
    pipeline : WtdEmulationPipeline
        Pipeline passed by the scorer. The aggregation does not inspect it.
    dataset : BaseGaitDataset
        Scored single-day datapoints, each with a named ``group_label``. All must share one sampling rate.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Combined sample-based classification and minute-based duration metrics, followed by per-datapoint metrics and
        raw interval tables indexed by dataset group labels. Entirely uncertain datapoints contribute no labeled
        samples to combined metrics. Combined rates and durations are NaN if no labeled samples remain; combined
        waking durations are NaN if no datapoint has waking-hours ground truth.

    Raises
    ------
    ValueError
        If datapoints have different sampling rates.
    """
    data_labels = [d.group_label for d in dataset]
    data_label_names = data_labels[0]._fields

    matches = single_results.pop("matches")
    matches = pd.concat(matches, keys=data_labels, names=[*data_label_names, *matches[0].index.names])
    detected = single_results.pop("detected")
    detected = pd.concat(detected, keys=data_labels, names=[*data_label_names, *detected[0].index.names])
    detected_scored = single_results.pop("detected_scored")
    detected_scored = pd.concat(
        detected_scored, keys=data_labels, names=[*data_label_names, *detected_scored[0].index.names]
    )
    reference = single_results.pop("reference")
    reference = pd.concat(reference, keys=data_labels, names=[*data_label_names, *reference[0].index.names])
    reference_scored = single_results.pop("reference_scored")
    reference_scored = pd.concat(
        reference_scored, keys=data_labels, names=[*data_label_names, *reference_scored[0].index.names]
    )
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

    combined_matched = {
        f"combined__{k}": v for k, v in calculate_matched_gsd_performance_metrics(_gsd_metric_matches(matches)).items()
    }
    if matches.empty:
        combined_matched.update({key: np.nan for key in combined_matched if not key.endswith("_samples")})
    combined_duration = {
        f"combined__{k}": v
        for k, v in _duration_metrics(
            reference_weartime=reference_scored,
            detected_weartime_min=np.nansum(single_results["detected_weartime_min"]),
            sampling_rate_hz=sampling_rate_hz[0],
        ).items()
    }
    combined_waking_duration = {
        f"combined__{k}": v
        for k, v in _duration_metrics(
            reference_weartime=reference_waking,
            detected_weartime_min=np.nansum(single_results["waking_detected_weartime_min"]),
            sampling_rate_hz=sampling_rate_hz[0],
            prefix="waking_",
        ).items()
    }
    if matches.empty:
        combined_duration = dict.fromkeys(combined_duration, np.nan)
    if np.isnan(single_results["waking_reference_weartime_min"]).all():
        combined_waking_duration = dict.fromkeys(combined_waking_duration, np.nan)

    aggregated_single_results = {
        "raw__matches": matches,
        "raw__detected": detected,
        "raw__detected_scored": detected_scored,
        "raw__reference": reference,
        "raw__reference_scored": reference_scored,
        "raw__reference_waking": reference_waking,
    }

    return (
        {**agg_results, **combined_matched, **combined_duration, **combined_waking_duration},
        {**single_results, **aggregated_single_results},
    )


def calculate_wtd_classification_summary(daily_results: pd.DataFrame) -> pd.Series:
    """Summarize held-out classification at day, participant and fold levels.

    Parameters
    ----------
    daily_results
        One row per held-out day for one algorithm, with ``fold``, ``participant_id``,
        ``tp_samples``, ``fp_samples``, ``fn_samples`` and ``tn_samples`` columns.
        Samples with uncertain ground truth must already be excluded from the counts.
        A participant's held-out days must belong to one fold.

    Returns
    -------
    pd.Series
        Precision, recall, F1, specificity, accuracy and NPV under six prefixes:

        - ``single_mean__day__combined__``
        - ``fold_mean__day__combined__``
        - ``single_mean__participant__combined__``
        - ``fold_mean__participant__combined__``
        - ``single_mean__participant__day_mean__combined__``
        - ``fold_mean__participant__day_mean__combined__``

        ``combined`` pools sample counts before calculating rates, within each day or participant.
        ``participant__day_mean`` averages daily rates within each participant instead.
        ``single_mean`` averages all scoring units across folds. ``fold_mean`` first averages the
        scoring units within each fold, then averages folds. These are means of rates, not a
        single rate calculated from samples pooled across all folds.

        A zero denominator yields zero, matching the default per-day scorer convention. A day or
        participant with no labeled samples has undefined rates and is excluded from means.
        Entirely undefined folds are also excluded from the fold mean.

        Each mean has ``__ci95_lower`` and ``__ci95_upper`` siblings. These are symmetric
        Student's t confidence intervals using the same averaging units as the mean.
        Bounds are NaN for fewer than two labeled units and are not clipped to [0, 1].
        Day-level intervals treat days as independent; they do not account for correlation
        between days of the same participant.
    """
    counts = daily_results.set_index(["fold", "participant_id"])[
        ["tp_samples", "fp_samples", "fn_samples", "tn_samples"]
    ]

    def rates(sample_counts: pd.DataFrame) -> pd.DataFrame:
        tp, fp, fn, tn = (sample_counts[column] for column in counts.columns)
        labeled = tp + fp + fn + tn
        scores = pd.DataFrame(
            {
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "f1_score": 2 * tp / (2 * tp + fp + fn),
                "specificity": tn / (tn + fp),
                "accuracy": (tp + tn) / labeled,
                "npv": tn / (tn + fn),
            }
        )
        return scores.fillna(0).where(labeled > 0, axis=0)

    day_scores = rates(counts)
    participant_scores = rates(counts.groupby(level=["fold", "participant_id"]).sum())
    participant_day_scores = day_scores.groupby(level=["fold", "participant_id"]).mean()
    summaries = []
    for name, scores in (
        ("day__combined", day_scores),
        ("participant__combined", participant_scores),
        ("participant__day_mean__combined", participant_day_scores),
    ):
        for prefix, units in (
            ("single_mean", scores),
            ("fold_mean", scores.groupby(level="fold").mean()),
        ):
            mean = units.mean().add_prefix(f"{prefix}__{name}__")
            half_width = (units.sem() * t.ppf(0.975, units.count() - 1)).set_axis(mean.index)
            summaries.extend(
                [mean, (mean - half_width).add_suffix("__ci95_lower"), (mean + half_width).add_suffix("__ci95_upper")]
            )
    return pd.concat(summaries)


def calculate_wtd_simulated_non_wear_summary(daily_results: pd.DataFrame) -> pd.Series:
    """Summarize repeated held-out predictions using unique simulated non-wear recordings.

    Parameters
    ----------
    daily_results
        One row per evaluated day and outer fold for one algorithm, containing only known non-wear recordings.
        Columns are ``fold``, ``recording_id``, the four ``*_samples`` confusion counts, and
        ``detected_weartime_min``. Uncertain samples must already be excluded. ``recording_id`` must identify
        the original recording globally across days and folds; SUSTAIN IDs include type, participant and CWA stem.

    Returns
    -------
    pd.Series
        ``single_mean__recording__fold_mean__combined__specificity`` and ``...__accuracy`` pool day counts
        within each recording/fold before calculating rates. For entirely non-wear ground truth, both rates
        are equal. ``single_mean__recording__fold_mean__day_mean__false_wear_min`` averages detected wear
        minutes over the evaluated days of each recording/fold. This is per evaluated day, not normalized
        to 24 hours. Each recording's metrics are then averaged across outer-fold models, followed by an
        equal mean over unique recordings. No positive-class rates or relative wear-duration errors are reported.

        Each mean has symmetric Student's t ``__ci95_lower`` and ``__ci95_upper`` bounds based on those
        unique recording values after fold-model averaging. Repeated recording/fold predictions are not
        independent CI observations. Undefined values are excluded; fewer than two measured recordings
        yield NaN bounds. Bounds are not clipped. ``n_recordings`` counts all provided recording IDs.
    """
    grouped = daily_results.groupby(["fold", "recording_id"])
    counts = grouped[["tp_samples", "fp_samples", "fn_samples", "tn_samples"]].sum()
    recording_fold_scores = pd.DataFrame(
        {
            "combined__specificity": counts["tn_samples"] / (counts["tn_samples"] + counts["fp_samples"]),
            "combined__accuracy": (counts["tp_samples"] + counts["tn_samples"]) / counts.sum(axis=1),
            "day_mean__false_wear_min": grouped["detected_weartime_min"].mean(),
        }
    )
    recording_scores = recording_fold_scores.groupby(level="recording_id").mean()
    mean = recording_scores.mean().add_prefix("single_mean__recording__fold_mean__")
    half_width = (recording_scores.sem() * t.ppf(0.975, recording_scores.count() - 1)).set_axis(mean.index)
    return pd.concat(
        [
            pd.Series({"n_recordings": len(recording_scores)}),
            mean,
            (mean - half_width).add_suffix("__ci95_lower"),
            (mean + half_width).add_suffix("__ci95_upper"),
        ]
    )


wtd_score = Scorer(
    wtd_per_datapoint_score, final_aggregator=wtd_final_agg, default_aggregator=FloatAggregator(np.nanmean)
)
wtd_score.__doc__ = """Scorer for wear-time detection algorithms.

This is a pre-configured :class:`~tpcp.validate.Scorer` object using :func:`wtd_per_datapoint_score` as
per-datapoint scorer and :func:`wtd_final_agg` as final aggregator. Pass single-day datapoints with
``reference_weartime_`` intervals in ``[start, end)`` sample coordinates and a common sampling rate. If a datapoint
provides ``reference_uncertain_``, its samples are excluded from scoring while the detector still receives the full
signal. Undefined per-day rates and durations are NaN and excluded from mean scores. Combined metrics use only
labeled samples; they are NaN when no applicable labels exist. ``raw__detected`` and ``raw__reference`` retain
original interval IDs, while ``raw__detected_scored`` and ``raw__reference_scored`` show fragments after masking.
"""


__all__ = [
    "calculate_wtd_classification_summary",
    "calculate_wtd_simulated_non_wear_summary",
    "wtd_final_agg",
    "wtd_per_datapoint_score",
    "wtd_score",
]
