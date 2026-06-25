"""Helper functions for machine learning-based wear-time detection."""

import numpy as np
import pandas as pd
from scipy.signal import welch

from mobgap.weartime.utils._intervals import flags_to_intervals, intervals_to_flags


def rolling_window_indices(n_samples: int, win_samples: int, step: int) -> tuple[int, int]:
    """
    Generate rolling window start and end indices.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the data
    win_samples : int
        Window size in samples
    step : int
        Step size in samples between windows

    Yields
    ------
    tuple[int, int]
        Start and end indices for each window
    """
    for start in range(0, n_samples - win_samples + 1, step):
        yield start, start + win_samples


def extract_features_from_windows(window: pd.DataFrame, sampling_rate: float = 100.0) -> dict:
    """
    Extract features from a window of data.

    This function matches the original feature extraction used for XGBoost training,
    using Welch's method for PSD estimation (without DC removal).

    Parameters
    ----------
    window : pd.DataFrame
        A micro-window with columns including 'acc_is', 'acc_ml', 'acc_pa', 'gyr_is', 'gyr_ml', 'gyr_pa'
    sampling_rate : float
        Sampling frequency in Hz (default: 100.0)

    Returns
    -------
    features : dict
        Dictionary containing:
        - gyr_ml_spectral_centroid: Spectral centroid of mediolateral gyroscope (Hz)
        - gyr_is_spectral_centroid: Spectral centroid of inferior-superior gyroscope (Hz)
        - acc_pa_std: Standard deviation of anteroposterior acceleration
    """
    features = {}

    # Feature 1: acc_pa_std (accelerometer PA standard deviation)
    if "acc_pa" in window.columns:
        col = window["acc_pa"].to_numpy()
        features["acc_pa_std"] = np.std(col, ddof=1)
    else:
        features["acc_pa_std"] = np.nan

    # Feature 2 & 3: Gyroscope spectral centroids (ML and IS)
    # Using Welch's method to exactly match original XGBoost training features
    for axis in ["gyr_ml", "gyr_is"]:
        if axis in window.columns:
            col = window[axis].to_numpy()

            # Compute PSD using Welch's method (nperseg=len(col) matches original)
            f, pxx = welch(col, fs=sampling_rate, nperseg=len(col))

            # Spectral centroid (weighted mean of frequencies)
            total_power = np.sum(pxx)
            if total_power > 0:
                psd_norm = pxx / total_power
                spectral_centroid = np.sum(f * psd_norm)
            else:
                spectral_centroid = 0.0

            features[f"{axis}_spectral_centroid"] = spectral_centroid
        else:
            features[f"{axis}_spectral_centroid"] = np.nan

    return features


def remove_short_wear_bouts_by_ratio(
    weartime_flags: np.ndarray, max_bout_minutes: float = 20.0, min_ratio: float = 0.3, sampling_rate_hz: float = 100.0
) -> np.ndarray:
    """
    Remove short wear bouts surrounded by disproportionately long non-wear periods.

    Applies mild filtering to remove suspicious short wear periods that are likely
    artifacts from device handling rather than true wear events.

    Rule: Wear periods ≤20 minutes with ratio <0.3 are removed.
    Ratio = wear_duration / (before_nonwear_duration + after_nonwear_duration)

    Example removals:
    - 10 min wear surrounded by 40+ min total non-wear (ratio <0.3)
    - 15 min wear surrounded by 50+ min total non-wear (ratio <0.3)

    Example kept:
    - 20 min wear surrounded by 50 min total non-wear (ratio 0.4 ≥0.3)
    - Any wear >20 minutes (rule doesn't apply)

    Rationale: Brief wear periods surrounded by much longer non-wear are likely
    device handling, table bumps, or transfer movements rather than true wear.

    Parameters
    ----------
    weartime_flags : np.ndarray
        Binary flags (1=wear, 0=non-wear)
    max_bout_minutes : float
        Maximum wear bout duration to consider for filtering (default: 20.0 minutes)
        Wear periods longer than this are kept regardless of ratio
    min_ratio : float
        Minimum ratio of wear duration to surrounding non-wear (default: 0.3)
        Wear bouts with ratio < min_ratio are removed
    sampling_rate_hz : float
        Sampling frequency in Hz (default: 100.0)

    Returns
    -------
    np.ndarray
        Flags with suspicious short wear bouts removed
    """
    weartime_flags = np.asarray(weartime_flags).ravel()
    max_bout_samples = int(max_bout_minutes * 60 * sampling_rate_hz)
    wear_intervals = flags_to_intervals(weartime_flags)

    if len(wear_intervals) == 0:
        return weartime_flags.copy()

    keep = np.ones(len(wear_intervals), dtype=bool)
    for i, (start, end) in enumerate(wear_intervals):
        wear_duration_samples = end - start
        if wear_duration_samples > max_bout_samples:
            continue

        previous_wear_end = wear_intervals[i - 1, 1] if i > 0 else 0
        next_wear_start = wear_intervals[i + 1, 0] if i < len(wear_intervals) - 1 else len(weartime_flags)
        surrounding_nonwear_samples = (start - previous_wear_end) + (next_wear_start - end)

        if surrounding_nonwear_samples > 0 and wear_duration_samples / surrounding_nonwear_samples < min_ratio:
            keep[i] = False

    return intervals_to_flags(wear_intervals[keep], len(weartime_flags), dtype=weartime_flags.dtype)
