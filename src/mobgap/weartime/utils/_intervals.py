"""Small interval adapters for wear-time utilities."""

from typing import Optional

import numpy as np
import pandas as pd

from mobgap.utils.array_handling import bool_array_to_start_end_array, start_end_array_to_bool_array


def _empty_interval_df(index_name: str = "wt_id") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.Series(dtype="int64"),
            "end": pd.Series(dtype="int64"),
        }
    ).rename_axis(index_name)


def _only_start_end(intervals: pd.DataFrame, *, index_name: str = "wt_id") -> pd.DataFrame:
    if intervals.empty:
        return _empty_interval_df(index_name)
    return intervals[["start", "end"]].astype({"start": "int64", "end": "int64"})


def _validate_waking_hours_min(waking_hours_min: tuple[int, int]) -> tuple[int, int]:
    start_min, end_min = waking_hours_min
    if not 0 <= start_min < end_min <= 24 * 60:
        raise ValueError(
            "`waking_hours_min` must define a non-empty window within one day using minutes since midnight."
        )
    return waking_hours_min


def _timestamp_to_sample_boundary(timestamp: pd.Timestamp, data_index: pd.DatetimeIndex) -> int:
    if timestamp < data_index[0]:
        return 0
    if timestamp > data_index[-1]:
        return len(data_index)
    return int(data_index.get_indexer([timestamp], method="nearest")[0])


def _waking_hours_sample_bounds(
    *,
    data: Optional[pd.DataFrame],
    sampling_rate_hz: float,
    waking_hours_min: tuple[int, int],
) -> tuple[int, int]:
    start_min, end_min = waking_hours_min
    if data is None or not isinstance(data.index, pd.DatetimeIndex):
        return int(start_min * 60 * sampling_rate_hz), int(end_min * 60 * sampling_rate_hz)

    if len(data.index) == 0:
        return 0, 0

    first_day = data.index[0].normalize()
    last_day = data.index[-1].normalize()
    if first_day != last_day:
        raise ValueError(
            "Waking-hours wear-time metrics require datapoints that do not cross midnight. "
            "Split recordings into individual days before scoring waking-hours wear-time."
        )

    start_ts = first_day + pd.Timedelta(minutes=start_min)
    end_ts = first_day + pd.Timedelta(minutes=end_min)
    return (
        _timestamp_to_sample_boundary(start_ts, data.index),
        _timestamp_to_sample_boundary(end_ts, data.index),
    )


def clip_intervals_to_waking_hours(
    intervals: pd.DataFrame,
    *,
    sampling_rate_hz: float,
    waking_hours_min: tuple[int, int],
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Clip ``[start, end)`` intervals to a daily waking-hours window."""
    start_min, end_min = _validate_waking_hours_min(waking_hours_min)
    start_sample, end_sample = _waking_hours_sample_bounds(
        data=data,
        sampling_rate_hz=sampling_rate_hz,
        waking_hours_min=(start_min, end_min),
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


def flags_to_intervals(flags: np.ndarray) -> np.ndarray:
    """Convert binary sample flags to ``[start, end)`` intervals."""
    flags = np.asarray(flags).ravel()
    if len(flags) == 0:
        return np.empty((0, 2), dtype=np.int64)

    intervals = bool_array_to_start_end_array(flags)
    if intervals.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return intervals.astype(np.int64, copy=False).reshape(-1, 2)


def intervals_to_flags(intervals: np.ndarray, length: int, *, dtype: np.dtype = bool) -> np.ndarray:
    """Convert ``[start, end)`` intervals to sample flags with the requested length."""
    if length < 0:
        raise ValueError("`length` must be non-negative.")

    intervals = np.asarray(intervals, dtype=np.int64)
    if intervals.size == 0:
        return np.zeros(length, dtype=dtype)

    return start_end_array_to_bool_array(intervals.reshape(-1, 2), pad_to_length=length).astype(dtype, copy=False)


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
