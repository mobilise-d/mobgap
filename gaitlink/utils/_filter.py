"""Package internal filter implementations."""
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation


def hampel_filter_vectorized(data: np.ndarray, window_size: int, n_sigmas: float = 3.0) -> np.ndarray:
    """Apply the Hampel filter to a time-series.

    Parameters
    ----------
    data
        The series to filter.
    window_size
        The size of the window to use for the median filter.
        Must be an odd number.
    n_sigmas
        The number of standard deviations to use for the outlier detection.

    Returns
    -------
    The filtered series.

    """

    k = 1.4826  # Scale factor for Gaussian distribution
    new_series = data.copy()

    # Create the median filtered series
    median_series = median_filter(data, size=window_size, mode="reflect")
    # Calculate the median absolute deviation with the corrected function
    scaled_mad = k * median_filter(median_abs_deviation(data, scale="normal"), size=window_size, mode="reflect")

    # Detect outliers
    outliers = np.abs(data - median_series) > n_sigmas * scaled_mad

    # Replace outliers
    new_series[outliers] = median_series[outliers]

    return new_series
