"""Helper functions for machine-learning wear-time detectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def window_start_end(n_samples: int, window_samples: int, step_samples: int) -> np.ndarray:
    """Return ``[start, end)`` sample boundaries for all full stepped windows."""
    if n_samples < 0:
        raise ValueError("`n_samples` must be non-negative.")
    if window_samples <= 0:
        raise ValueError("`window_samples` must be positive.")
    if step_samples <= 0:
        raise ValueError("`step_samples` must be positive.")
    if n_samples < window_samples:
        return np.empty((0, 2), dtype=np.int64)

    starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=np.int64)
    return np.column_stack([starts, starts + window_samples])


def window_count_from_sample_count(n_samples: int, window_samples: int, step_samples: int) -> int:
    """Count full stepped windows in one recording."""
    if n_samples < 0:
        raise ValueError("Recording sample counts must be non-negative.")
    if window_samples <= 0:
        raise ValueError("`window_samples` must be positive.")
    if step_samples <= 0:
        raise ValueError("`step_samples` must be positive.")
    if n_samples < window_samples:
        return 0
    return (n_samples - window_samples) // step_samples + 1


def reference_weartime_interval_arrays(reference_weartime: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted reference wear-time interval starts and ends."""
    if len(reference_weartime) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    intervals = reference_weartime[["start", "end"]].to_numpy(dtype=np.int64, copy=False)
    intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
    return intervals[:, 0], intervals[:, 1]


def labels_from_interval_arrays(
    centers: np.ndarray, interval_starts: np.ndarray, interval_ends: np.ndarray
) -> np.ndarray:
    """Label window centers as wear when they fall into any reference interval."""
    if len(interval_starts) == 0:
        return np.zeros(len(centers), dtype=np.int32)

    interval_indices = np.searchsorted(interval_starts, centers, side="right") - 1
    labels = np.zeros(len(centers), dtype=np.int32)
    valid = interval_indices >= 0
    labels[valid] = centers[valid] < interval_ends[interval_indices[valid]]
    return labels


def labels_from_interval_centers(centers: np.ndarray, reference_weartime: pd.DataFrame) -> np.ndarray:
    """Label window centers from a reference wear-time DataFrame."""
    return labels_from_interval_arrays(centers, *reference_weartime_interval_arrays(reference_weartime))
