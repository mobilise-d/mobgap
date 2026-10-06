"""Small interval adapters for wear-time utilities."""

import warnings
from datetime import time
from fractions import Fraction
from math import ceil
from typing import Optional

import numpy as np
import pandas as pd


def _only_start_end(intervals: pd.DataFrame, *, index_name: Optional[str] = None) -> pd.DataFrame:
    result = intervals[["start", "end"]].astype({"start": "int64", "end": "int64"})
    if index_name is not None:
        return result.rename_axis(index_name)
    return result


def _validate_waking_hours(waking_hours: tuple[time, time]) -> tuple[time, time]:
    start, end = waking_hours
    if end == time(0):
        return waking_hours
    if start >= end:
        raise ValueError("`waking_hours` must define a non-empty window within one day.")
    return waking_hours


def _timestamp_to_sample_boundary(timestamp: pd.Timestamp, data_index: pd.DatetimeIndex) -> int:
    return int(data_index.searchsorted(timestamp, side="left"))


def _waking_hours_sample_bounds(
    *,
    data: Optional[pd.DataFrame],
    sampling_rate_hz: float,
    waking_hours: tuple[time, time],
) -> tuple[int, int]:
    start, end = waking_hours
    if data is None or not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:

        def minutes_since_midnight(value: time) -> Fraction:
            microseconds = ((value.hour * 60 + value.minute) * 60 + value.second) * 1_000_000 + value.microsecond
            return Fraction(microseconds, 60_000_000)

        end_min = 24 * 60 if end == time(0) else minutes_since_midnight(end)
        # Recover rates such as 1/15 Hz from their floating-point representation before rounding to samples.
        samples_per_minute = 60 * Fraction(sampling_rate_hz).limit_denominator(1_000_000)
        return (
            ceil(minutes_since_midnight(start) * samples_per_minute),
            ceil(end_min * samples_per_minute),
        )

    if len(data.index) == 0:
        return 0, 0

    first_day = data.index[0].date()
    last_day = data.index[-1].date()
    if first_day != last_day:
        raise ValueError(
            "Waking-hours wear-time metrics require datapoints that do not cross midnight. "
            "Split recordings into individual days before scoring waking-hours wear-time."
        )

    def localize_boundary(clock_time: time, *, first_occurrence: bool, next_day: bool = False) -> pd.Timestamp:
        local_day = pd.Timestamp(first_day) + pd.Timedelta(days=int(next_day))
        local_time = pd.Timestamp.combine(local_day, clock_time)
        boundary = local_time.tz_localize(data.index.tz, ambiguous=first_occurrence, nonexistent="NaT")
        if pd.isna(boundary):
            # shift_backward lands at the last representable instant before the gap.
            boundary = local_time.tz_localize(data.index.tz, nonexistent="shift_backward")
            boundary += pd.Timedelta(1, boundary.unit)
        return boundary

    start_ts = localize_boundary(start, first_occurrence=True)
    end_ts = localize_boundary(end, first_occurrence=False, next_day=end == time(0))
    return (
        _timestamp_to_sample_boundary(start_ts, data.index),
        _timestamp_to_sample_boundary(end_ts, data.index),
    )


def clip_intervals_to_waking_hours(
    intervals: pd.DataFrame,
    *,
    sampling_rate_hz: float,
    waking_hours: tuple[time, time],
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Clip ``[start, end)`` intervals to local daily waking hours.

    A skipped clock time moves to the first valid time. For repeated times, the start uses the first occurrence and
    the end uses the second, so the repeated hour is included.

    Parameters
    ----------
    intervals : pd.DataFrame
        Intervals with sample-based ``start`` and ``end`` columns.
    sampling_rate_hz : float
        Sampling rate used when ``data`` has no timestamps.
    waking_hours : tuple[datetime.time, datetime.time]
        Start and end of the local daily window. An end time of midnight means the end of the day;
        ``(time(0), time(0))`` covers the full day.
    data : pd.DataFrame, optional
        Recording data. A timezone-aware ``DatetimeIndex`` sets the local dates and times of the window.
        Without one, sample zero represents midnight and daylight-saving transitions cannot be considered.

    Returns
    -------
    pd.DataFrame
        Clipped half-open intervals with the input index and ``start`` and ``end`` columns.

    Raises
    ------
    ValueError
        If the waking-hours window is invalid or timestamped data crosses local midnight.
    """
    waking_hours = _validate_waking_hours(waking_hours)
    if data is None or not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        warnings.warn(
            "The provided data does not have a localized DatetimeIndex; assuming the recording starts at midnight. "
            "DST and similar clock changes cannot be considered.",
            UserWarning,
            stacklevel=2,
        )
    start_sample, end_sample = _waking_hours_sample_bounds(
        data=data,
        sampling_rate_hz=sampling_rate_hz,
        waking_hours=waking_hours,
    )

    intervals = _only_start_end(intervals)
    if intervals.empty:
        return intervals

    clipped = intervals.assign(
        start=lambda df_: df_["start"].clip(lower=start_sample),
        end=lambda df_: df_["end"].clip(upper=end_sample),
    )
    clipped = clipped[clipped["end"] > clipped["start"]]
    return clipped.astype({"start": "int64", "end": "int64"})


def remove_short_interior_intervals(intervals: np.ndarray, min_samples: int, data_length: int) -> np.ndarray:
    """Remove intervals shorter than ``min_samples`` unless they touch a recording boundary."""
    intervals = np.asarray(intervals, dtype=np.int64)
    if intervals.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    intervals = intervals.reshape(-1, 2)
    if min_samples <= 0:
        return intervals.copy()

    durations = intervals[:, 1] - intervals[:, 0]
    at_boundary = (intervals[:, 0] == 0) | (intervals[:, 1] == data_length)
    return intervals[(durations >= min_samples) | at_boundary]
