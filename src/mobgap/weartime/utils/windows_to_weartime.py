"""Utilities for interval-based wear-time post-processing."""

import numpy as np

from mobgap.utils.array_handling import merge_intervals
from mobgap.weartime.utils._intervals import remove_short_interior_intervals


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


def remove_short_wear_bouts_by_ratio_from_intervals(
    wear_intervals: np.ndarray,
    *,
    data_length: int,
    max_bout_minutes: float = 20.0,
    min_ratio: float = 0.3,
    sampling_rate_hz: float = 100.0,
) -> np.ndarray:
    """Remove short wear bouts surrounded by disproportionately long non-wear periods from wear intervals."""
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
