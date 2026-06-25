"""Utilities for sample-level wear-time flags."""

import numpy as np

from mobgap.utils.array_handling import merge_intervals
from mobgap.weartime.utils._intervals import flags_to_intervals, intervals_to_flags, remove_short_interior_intervals


def remove_isolated_short_periods(
    weartime_flags: np.ndarray, min_period_sec: float = 15.0, sampling_rate_hz: float = 100.0
) -> np.ndarray:
    """
    Remove isolated wear/non-wear periods shorter than minimum duration.

    Rationale: Device attachment/removal requires a minimum physical time. Periods shorter
    than this threshold (default 15 seconds) are likely sensor artifacts, voting edge effects,
    or brief environmental disturbances rather than true wear state changes.

    This post-processing step removes sequentially:
    1. Brief isolated WEAR periods (<15s) — removed first, as pooled model evaluation
       consistently showed FP > FN, indicating a systematic tendency to over-detect wear.
    2. Brief isolated NON-WEAR periods (<15s) — removed second, reflecting the same
       physical impossibility of device attachment/removal within this timeframe.

    Boundary periods (at the start or end of the recording) are exempt from removal
    in both stages, as these may represent genuine partial wear or non-wear periods
    truncated by the recording window.

    Parameters
    ----------
    weartime_flags : np.ndarray
        Binary flags (1=wear, 0=non-wear) from majority voting
    min_period_sec : float
        Minimum period duration in seconds (default: 15.0)
        Periods shorter than this will be removed
    sampling_rate_hz : float
        Sampling frequency in Hz (default: 100.0)

    Returns
    -------
    np.ndarray
        Flags with brief isolated periods removed
    """
    weartime_flags = np.asarray(weartime_flags).ravel()
    min_samples = int(min_period_sec * sampling_rate_hz)

    if min_samples <= 0 or len(weartime_flags) == 0:
        return weartime_flags.copy()

    wear_intervals = flags_to_intervals(weartime_flags)
    wear_intervals = remove_short_interior_intervals(wear_intervals, min_samples, len(weartime_flags))

    if len(wear_intervals) > 0:
        wear_intervals = merge_intervals(wear_intervals, gap_size=min_samples - 1)

    return intervals_to_flags(wear_intervals, len(weartime_flags), dtype=weartime_flags.dtype)
