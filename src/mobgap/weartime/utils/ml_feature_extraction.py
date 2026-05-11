import numpy as np
import pandas as pd
from scipy.signal import welch

"""Rolling window function"""


def rolling_window_indices(n_samples, win_samples, step):
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
            f, Pxx = welch(col, fs=sampling_rate, nperseg=len(col))

            # Spectral centroid (weighted mean of frequencies)
            total_power = np.sum(Pxx)
            if total_power > 0:
                psd_norm = Pxx / total_power
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
    max_bout_samples = int(max_bout_minutes * 60 * sampling_rate_hz)

    # Find wear segments
    padded = np.pad(weartime_flags, (1, 1), constant_values=0)
    diff = np.diff(padded)

    wear_starts = np.where(diff == 1)[0]
    wear_ends = np.where(diff == -1)[0]

    filtered_flags = weartime_flags.copy()

    for start, end in zip(wear_starts, wear_ends):
        wear_duration_samples = end - start

        # Only apply rule to short wear bouts (≤20 minutes)
        if wear_duration_samples <= max_bout_samples:
            # Get surrounding non-wear durations
            before_nonwear_samples = 0
            after_nonwear_samples = 0

            # Before: Find start of preceding non-wear period
            if start > 0:
                nonwear_start = 0
                for i in range(start - 1, -1, -1):
                    if weartime_flags[i] == 1:  # Hit previous wear period
                        nonwear_start = i + 1
                        break
                before_nonwear_samples = start - nonwear_start

            # After: Find end of following non-wear period
            if end < len(weartime_flags):
                nonwear_end = len(weartime_flags)
                for i in range(end, len(weartime_flags)):
                    if weartime_flags[i] == 1:  # Hit next wear period
                        nonwear_end = i
                        break
                after_nonwear_samples = nonwear_end - end

            # Calculate ratio
            surrounding_nonwear_samples = before_nonwear_samples + after_nonwear_samples

            if surrounding_nonwear_samples > 0:
                ratio = wear_duration_samples / surrounding_nonwear_samples

                # Remove if ratio too low
                if ratio < min_ratio:
                    filtered_flags[start:end] = 0

    return filtered_flags
