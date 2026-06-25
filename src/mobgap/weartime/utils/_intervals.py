"""Small interval adapters for wear-time utilities."""

import numpy as np
import pandas as pd

from mobgap.utils.array_handling import bool_array_to_start_end_array, start_end_array_to_bool_array


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


def intervals_to_weartime_df(intervals: np.ndarray) -> pd.DataFrame:
    """Convert ``[start, end)`` intervals to the canonical wear-time dataframe shape."""
    intervals = np.asarray(intervals, dtype=np.int64)
    intervals = np.empty((0, 2), dtype=np.int64) if intervals.size == 0 else intervals.reshape(-1, 2)

    df = pd.DataFrame(intervals, columns=["start", "end"])
    df.index.name = "wt_id"
    return df


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
