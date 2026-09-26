"""Utilities for window voting and interval-based wear-time post-processing."""

import numpy as np

from mobgap.utils.array_handling import merge_intervals
from mobgap.weartime.utils._intervals import remove_short_interior_intervals


def overlapping_window_predictions_to_sample_labels(
    predictions: list[int],
    data_length: int,
    window_samples: int,
    step_samples: int,
    *,
    extend_tail: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert binary overlapping window predictions to sample labels using majority voting.

    Ties are treated as wear, matching the conservative behavior of the original ML wear-time post-processing.
    """
    if data_length < 0:
        raise ValueError("`data_length` must be non-negative.")
    if window_samples <= 0:
        raise ValueError("`window_samples` must be positive.")
    if step_samples <= 0:
        raise ValueError("`step_samples` must be positive.")

    predictions = np.asarray(predictions)
    if predictions.ndim != 1:
        raise ValueError("`predictions` must be one-dimensional.")
    if np.any((predictions != 0) & (predictions != 1)):
        raise ValueError("`predictions` must only contain binary labels 0 and 1.")

    predictions = predictions.astype(np.intp, copy=False)
    window_starts = np.arange(len(predictions), dtype=np.int64) * step_samples
    window_ends = np.minimum(window_starts + window_samples, data_length)
    valid_windows = window_starts < window_ends

    vote_count_diffs = np.zeros((data_length + 1, 2), dtype=np.int32)
    if not np.any(valid_windows):
        return np.zeros(data_length, dtype=np.int32), vote_count_diffs[:-1]

    np.add.at(vote_count_diffs, (window_starts[valid_windows], predictions[valid_windows]), 1)
    np.add.at(vote_count_diffs, (window_ends[valid_windows], predictions[valid_windows]), -1)
    last_covered_idx = int(window_ends[valid_windows].max() - 1)

    vote_counts = np.cumsum(vote_count_diffs[:-1], axis=0)
    # Keep the original conservative tie-breaking behavior: equal votes are treated as wear.
    sample_labels = (vote_counts[:, 1] >= vote_counts[:, 0]).astype(np.int32)

    uncovered_samples = data_length - (last_covered_idx + 1)
    if extend_tail and uncovered_samples > 0 and len(predictions) > 0:
        sample_labels[last_covered_idx + 1 :] = predictions[-1]

    return sample_labels, vote_counts


def remove_isolated_short_periods_from_intervals(
    wear_intervals: np.ndarray,
    *,
    data_length: int,
    min_period_sec: float = 15.0,
    sampling_rate_hz: float = 100.0,
) -> np.ndarray:
    """
    Remove isolated wear/non-wear periods shorter than minimum duration from wear intervals.

    It removes short interior wear intervals and then merges wear intervals separated by
    short interior non-wear gaps. Boundary periods are kept.
    """
    if data_length < 0:
        raise ValueError("`data_length` must be non-negative.")

    wear_intervals = np.asarray(wear_intervals, dtype=np.int64)
    if wear_intervals.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    wear_intervals = wear_intervals.reshape(-1, 2)
    min_samples = int(min_period_sec * sampling_rate_hz)
    if min_samples <= 0:
        return wear_intervals.copy()

    wear_intervals = remove_short_interior_intervals(wear_intervals, min_samples, data_length)

    if len(wear_intervals) > 0:
        wear_intervals = merge_intervals(wear_intervals, gap_size=min_samples - 1)

    return wear_intervals


def filter_short_wear_bouts_by_confidence(
    *,
    wear_intervals: np.ndarray,
    vote_counts: np.ndarray,
    data_length: int,
    sampling_rate_hz: float,
    min_confidence_short_bouts: float,
    short_bout_threshold_minutes: float,
    min_bout_duration_seconds: float,
) -> np.ndarray:
    """Remove short interior wear bouts unless overlapping windows agree strongly enough."""
    if data_length < 0:
        raise ValueError("`data_length` must be non-negative.")

    wear_intervals = np.asarray(wear_intervals, dtype=np.int64)
    if wear_intervals.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    wear_intervals = wear_intervals.reshape(-1, 2)
    vote_counts = np.asarray(vote_counts)
    if vote_counts.shape != (data_length, 2):
        raise ValueError("`vote_counts` must have shape `(data_length, 2)`.")

    short_bout_threshold_samples = short_bout_threshold_minutes * 60 * sampling_rate_hz
    min_bout_samples = min_bout_duration_seconds * sampling_rate_hz

    durations = wear_intervals[:, 1] - wear_intervals[:, 0]
    at_boundary = (wear_intervals[:, 0] == 0) | (wear_intervals[:, 1] == data_length)
    keep = durations >= min_bout_samples

    confidence_filter_candidates = keep & (durations < short_bout_threshold_samples) & ~at_boundary
    if np.any(confidence_filter_candidates):
        vote_prefix = np.vstack([np.zeros((1, 2), dtype=np.int64), np.cumsum(vote_counts, axis=0, dtype=np.int64)])
        for interval_index in np.flatnonzero(confidence_filter_candidates):
            start, end = wear_intervals[interval_index]
            bout_vote_counts = vote_prefix[end] - vote_prefix[start]
            if _wear_vote_proportion(bout_vote_counts) < min_confidence_short_bouts:
                keep[interval_index] = False

    return wear_intervals[keep]


def _wear_vote_proportion(bout_vote_counts_by_class: np.ndarray) -> float:
    total_votes = bout_vote_counts_by_class.sum()
    if total_votes <= 0:
        return 0.0
    return float(bout_vote_counts_by_class[1] / total_votes)


def remove_short_wear_bouts_by_ratio_from_intervals(
    wear_intervals: np.ndarray,
    *,
    data_length: int,
    max_bout_minutes: float = 20.0,
    min_ratio: float = 0.3,
    sampling_rate_hz: float = 100.0,
) -> np.ndarray:
    """Remove short wear bouts surrounded by disproportionately long non-wear periods from wear intervals."""
    if data_length < 0:
        raise ValueError("`data_length` must be non-negative.")

    wear_intervals = np.asarray(wear_intervals, dtype=np.int64)
    if wear_intervals.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    wear_intervals = wear_intervals.reshape(-1, 2)
    max_bout_samples = int(max_bout_minutes * 60 * sampling_rate_hz)

    keep = np.ones(len(wear_intervals), dtype=bool)
    for i, (start, end) in enumerate(wear_intervals):
        wear_duration_samples = end - start
        if wear_duration_samples > max_bout_samples:
            continue

        previous_wear_end = wear_intervals[i - 1, 1] if i > 0 else 0
        next_wear_start = wear_intervals[i + 1, 0] if i < len(wear_intervals) - 1 else data_length
        surrounding_nonwear_samples = (start - previous_wear_end) + (next_wear_start - end)

        if surrounding_nonwear_samples > 0 and wear_duration_samples / surrounding_nonwear_samples < min_ratio:
            keep[i] = False

    return wear_intervals[keep]
