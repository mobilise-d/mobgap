"""Helper functions to perform interpolation and resampling tasks."""
import numpy as np


def interval_mean(
    measurement_samples: np.ndarray,
    measurements: np.ndarray,
    interval_start_ends: np.ndarray,
) -> np.ndarray:
    """Calculate the average over all measurements within the intervals.

    .. warning :: This assumes that the measurement samples are sorted!

    The method considers all measurements that are within the interval, including the start and end values.

    The calculation is performed using "nanmean", so that NaN values in the measurement are ignored.
    If no measurements are within the interval, the result will be NaN.

    Parameters
    ----------
    measurement_samples
        The samples at which the measurements were taken.
    measurements
        The measurements.
    interval_start_ends
        The start and end indices of the intervals.
        The first column ([:, 0]) contains the start indices and the second column ([:, 1]) contains the end indices.
        The unit of the indices must be the same as the unit of the measurement samples.

    Returns
    -------
    The average over all measurements within the intervals.

    """
    if len(measurement_samples) == 0:
        return np.full(len(interval_start_ends), np.nan)

    if len(interval_start_ends) == 0:
        return np.array([])

    # We use searchsorted to find the start and end indices of the intervals
    interval_start_indices = np.searchsorted(measurement_samples, interval_start_ends[:, 0], side="left")
    interval_end_indices = np.searchsorted(measurement_samples, interval_start_ends[:, 1], side="right")

    # We calculate the average over all measurements within the intervals
    interval_means = np.array(
        [np.nanmean(measurements[s:e]) for s, e in zip(interval_start_indices, interval_end_indices)]
    )
    return interval_means
